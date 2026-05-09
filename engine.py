"""
Steam Engine — Small-State Thesis Demo
Core neural network, data generation, and three-axis compression engine.

Phase 2: dual-head architecture. The shared encoder feeds both a classification
head (5-class pattern recognition) and a prediction head (next-4-value forecast
from the first 12 tokens). Joint training tests how compression affects each
capability independently — pattern recognition vs world-model dynamics.

The model is trained with compression-awareness (random window masking, latent
dropout, weight noise injection) so it learns representations robust to all
three axes. Both heads share that encoder, so compression hits both.
"""

import torch
import torch.nn as nn
import numpy as np
import copy
import time
from typing import Optional

SEQ_LEN = 64
HIDDEN_DIM = 64
LATENT_DIM = 64
N_CLASSES = 10
PRED_CTX = 48   # first 48 tokens used as prediction context
PRED_LEN = 16   # predict the next 16 values
PATTERN_NAMES = [
    "Arithmetic",       # 0: linear  a + b*t
    "Geometric",        # 1: a * r^t
    "Quadratic",        # 2: a + b*t + c*t^2
    "Exponential",      # 3: a * exp(-k*t) + offset
    "Fibonacci",        # 4: s[n] = 0.5*(s[n-1] + s[n-2])
    "AR1",              # 5: s[n] = rho*s[n-1] + epsilon
    "Periodic",         # 6: sin(omega*t + phi)
    "Damped",           # 7: exp(-k*t) * sin(omega*t)
    "RandomWalk",       # 8: cumulative noise
    "Random",           # 9: pure noise
]

BIT_LEVELS = [32, 16, 8, 4, 2, 1.58, 1]
LATENT_DIMS = [64, 48, 32, 16, 8, 4, 2, 1]
WINDOW_SIZES = [64, 48, 32, 24, 16, 12, 8, 4, 2, 1]


def generate_sequences(n: int, seed: Optional[int] = None):
    rng = np.random.RandomState(seed)
    seqs, labels = [], []
    for _ in range(n):
        p = rng.randint(0, N_CLASSES)
        s = _make_pattern(p, rng)
        std = s.std()
        if std > 1e-8:
            s = (s - s.mean()) / std
        s += rng.normal(0, 0.08, SEQ_LEN)
        seqs.append(s.astype(np.float32))
        labels.append(p)
    return np.stack(seqs), np.array(labels, dtype=np.int64)


def _make_pattern(p: int, rng: np.random.RandomState):
    t = np.arange(SEQ_LEN, dtype=np.float64)
    t_norm = t / SEQ_LEN  # normalized time [0, 1) for stability across patterns

    if p == 0:  # Arithmetic: constant slope, identifiable from any 2 points
        return rng.uniform(-2, 2) + rng.uniform(-0.5, 0.5) * t

    elif p == 1:  # Geometric: constant ratio, identifiable from 2-3 points
        return rng.uniform(0.5, 1.5) * np.power(rng.uniform(0.94, 1.06), t)

    elif p == 2:  # Quadratic: a + b*t + c*t^2, smooth curvature
        a = rng.uniform(-1, 1)
        b = rng.uniform(-0.5, 0.5)
        c = rng.uniform(-0.02, 0.02)
        return a + b * t + c * t * t

    elif p == 3:  # Exponential decay: a * exp(-k*t) + offset
        k = rng.uniform(0.02, 0.1)
        a = rng.uniform(0.5, 2.0) * rng.choice([-1, 1])
        offset = rng.uniform(-0.5, 0.5)
        return a * np.exp(-k * t) + offset

    elif p == 4:  # Fibonacci-like: s[n] = 0.5 * (s[n-1] + s[n-2])
        s = [rng.uniform(-0.5, 0.5), rng.uniform(-0.5, 0.5)]
        for _ in range(SEQ_LEN - 2):
            s.append(0.5 * (s[-1] + s[-2]))
        return np.array(s)

    elif p == 5:  # AR(1): s[n] = rho * s[n-1] + epsilon
        rho = rng.uniform(0.6, 0.95)
        eps_scale = rng.uniform(0.05, 0.15)
        s = [rng.uniform(-1, 1)]
        for _ in range(SEQ_LEN - 1):
            s.append(rho * s[-1] + rng.normal(0, eps_scale))
        return np.array(s)

    elif p == 6:  # Periodic: sin(omega*t + phi), needs ~half-period to identify
        omega = rng.uniform(0.15, 0.6)  # full period ~10-40 tokens within length 64
        phi = rng.uniform(0, 2 * np.pi)
        return rng.uniform(0.7, 1.5) * np.sin(omega * t + phi)

    elif p == 7:  # Damped oscillator: exp(-k*t) * sin(omega*t)
        k = rng.uniform(0.015, 0.06)
        omega = rng.uniform(0.2, 0.7)
        phi = rng.uniform(0, 2 * np.pi)
        return rng.uniform(1.0, 2.0) * np.exp(-k * t) * np.sin(omega * t + phi)

    elif p == 8:  # Random walk: cumulative noise, structure but unpredictable detail
        steps = rng.normal(0, 0.15, SEQ_LEN)
        return np.cumsum(steps)

    else:  # Random: no structure
        return rng.uniform(-1, 1, SEQ_LEN)


class Attention(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)
        self.scale = dim ** -0.5

    def forward(self, x, window=None):
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, C).permute(2, 0, 1, 3)
        q, k, v = qkv.unbind(0)
        attn = (q @ k.transpose(-2, -1)) * self.scale

        if window is not None and window < T:
            mask = torch.full((T, T), float('-inf'), device=x.device)
            for i in range(T):
                lo = max(0, i - window // 2)
                hi = min(T, i + (window + 1) // 2)
                mask[i, lo:hi] = 0.0
            attn = attn + mask

        return self.proj(attn.softmax(-1) @ v)


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Linear(1, HIDDEN_DIM)
        self.attn1 = Attention(HIDDEN_DIM)
        self.norm1 = nn.LayerNorm(HIDDEN_DIM)
        self.attn2 = Attention(HIDDEN_DIM)
        self.norm2 = nn.LayerNorm(HIDDEN_DIM)
        self.enc = nn.Linear(HIDDEN_DIM, LATENT_DIM)
        self.dec = nn.Linear(LATENT_DIM, HIDDEN_DIM)
        self.act = nn.GELU()
        self.norm3 = nn.LayerNorm(HIDDEN_DIM)
        self.head = nn.Sequential(
            nn.Linear(HIDDEN_DIM, 32),
            nn.GELU(),
            nn.Linear(32, N_CLASSES),
        )
        self.pred_head = nn.Sequential(
            nn.Linear(HIDDEN_DIM, 32),
            nn.GELU(),
            nn.Linear(32, PRED_LEN),
        )
        # Phase 3: latent prediction head — symmetric architecture with raw head
        self.latent_pred_head = nn.Sequential(
            nn.Linear(HIDDEN_DIM, 32),
            nn.GELU(),
            nn.Linear(32, LATENT_DIM),
        )

    def encode(self, x, window=None, latent_mask=None, pool_range=None):
        """Encode a sequence to a pooled HIDDEN_DIM vector.

        pool_range: optional (start, end) tuple. When set, pools only over those
        positions instead of mean over the full sequence. Used for target latent
        extraction in Phase 3 (e.g., pool_range=(PRED_CTX, PRED_CTX+PRED_LEN)
        to encode only the future positions for the JEPA-style target).
        """
        x = self.embed(x.unsqueeze(-1))
        x = self.norm1(x + self.attn1(x, window))
        x = self.norm2(x + self.attn2(x, window))
        z = self.enc(x)
        if latent_mask is not None:
            z = z * latent_mask
        h = self.norm3(self.dec(self.act(z)))
        if pool_range is not None:
            start, end = pool_range
            return h[:, start:end].mean(1)
        return h.mean(1)

    def forward(self, x, window=None, latent_mask=None):
        pooled = self.encode(x, window, latent_mask)
        return self.head(pooled), self.pred_head(pooled), self.latent_pred_head(pooled)


def quantize(model: nn.Module, bits: float):
    """Post-training quantization with group-wise optimal scaling."""
    sd = model.state_dict()
    for k, v in sd.items():
        if v.dim() < 2:
            continue
        if bits <= 1.0:  # Binary {-1, +1}
            scale = v.abs().mean()
            q = torch.sign(v)
            q[q == 0] = 1.0
            sd[k] = q * scale
        elif bits <= 1.58:  # Ternary {-1, 0, +1} with group-wise scaling
            group_size = min(64, v.numel())
            v_flat = v.flatten()
            q_flat = torch.zeros_like(v_flat)
            for g in range(0, len(v_flat), group_size):
                group = v_flat[g:g + group_size]
                thr = group.abs().mean() * 0.4
                gq = torch.zeros_like(group)
                gq[group > thr] = 1.0
                gq[group < -thr] = -1.0
                nz = gq != 0
                if nz.any():
                    scale = (group[nz] * gq[nz]).sum() / gq[nz].abs().sum()
                    q_flat[g:g + group_size] = gq * scale.clamp(min=1e-6)
            sd[k] = q_flat.reshape(v.shape)
        else:  # Uniform quantization
            levels = 2 ** int(bits)
            mn, mx = v.min(), v.max()
            if mn == mx:
                continue
            step = (mx - mn) / (levels - 1)
            sd[k] = torch.round((v - mn) / step) * step + mn
    model.load_state_dict(sd)


class Engine:
    def __init__(self):
        self.model: Optional[Model] = None
        self.test_x: Optional[torch.Tensor] = None
        self.test_y: Optional[torch.Tensor] = None
        self.baseline = 0.0
        self.baseline_pred_mse = 1.0
        self.baseline_pred_mae = 1.0
        self.baseline_latent_mse = 1.0
        self.baseline_latent_mae = 1.0
        self.sweeps = {}
        self.pred_sweeps = {}
        self.latent_sweeps = {}
        self.ready = False
        self.n_params = 0

    def train(self, alpha: float = 1.0, beta: float = 0.5,
              model_seed: Optional[int] = None,
              aug_seed: int = 7, data_train_seed: int = 42, data_test_seed: int = 99,
              epochs: int = 120, n_train: int = 6000, n_test: int = 1000,
              warmup_epoch: int = 40):
        t0 = time.time()
        X_tr, y_tr = generate_sequences(n_train, seed=data_train_seed)
        X_te, y_te = generate_sequences(n_test, seed=data_test_seed)
        self.test_x = torch.from_numpy(X_te)
        self.test_y = torch.from_numpy(y_te)

        if model_seed is not None:
            torch.manual_seed(model_seed)
        m = Model()
        self.n_params = sum(p.numel() for p in m.parameters())
        opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=2e-3)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
        ce_fn = nn.CrossEntropyLoss()
        mse_fn = nn.MSELoss()
        tx = torch.from_numpy(X_tr)
        ty = torch.from_numpy(y_tr)

        m.train()
        rng = np.random.RandomState(aug_seed)
        for epoch in range(epochs):
            perm = torch.randperm(len(tx))
            for i in range(0, len(tx), 128):
                idx = perm[i:i + 128]
                bx = tx[idx]
                by = ty[idx]

                # Compression-aware training after warmup
                window = None
                latent_mask = None
                if epoch >= warmup_epoch:
                    if rng.random() < 0.4:
                        # Window choices scaled to SEQ_LEN
                        window = int(rng.choice([4, 8, 12, 16, 24, 32, 48]))
                    if rng.random() < 0.4:
                        keep = rng.randint(max(4, LATENT_DIM // 4), LATENT_DIM)
                        latent_mask = torch.zeros(LATENT_DIM)
                        latent_mask[:keep] = 1.0

                opt.zero_grad()

                # Classification: full sequence with compression-aware augmentation
                class_logits, _, _ = m(bx, window=window, latent_mask=latent_mask)
                ce_loss = ce_fn(class_logits, by)

                # Raw prediction: first PRED_CTX tokens, augmented; predict positions PRED_CTX:PRED_CTX+PRED_LEN
                ctx = bx[:, :PRED_CTX]
                target_raw = bx[:, PRED_CTX:PRED_CTX + PRED_LEN]
                pred_window = window if window is None else min(window, PRED_CTX)
                _, pred_raw, pred_latent = m(ctx, window=pred_window, latent_mask=latent_mask)
                mse_raw_loss = mse_fn(pred_raw, target_raw)

                # Phase 3: latent target from UNCOMPRESSED full-sequence forward
                # (target encoder = same weights, stop_grad, no compression)
                with torch.no_grad():
                    target_latent = m.encode(bx, pool_range=(PRED_CTX, PRED_CTX + PRED_LEN))
                mse_latent_loss = mse_fn(pred_latent, target_latent)

                (ce_loss + alpha * mse_raw_loss + beta * mse_latent_loss).backward()
                opt.step()

            # Weight noise injection (axis 1 robustness) — simulates quantization
            if epoch >= warmup_epoch and epoch % 5 == 0:
                with torch.no_grad():
                    for p in m.parameters():
                        if p.dim() >= 2:
                            noise_scale = p.abs().mean() * 0.05
                            p.add_(torch.randn_like(p) * noise_scale)

            sched.step()

        m.eval()
        self.model = m
        with torch.no_grad():
            class_logits, _, _ = m(self.test_x)
            self.baseline = (class_logits.argmax(1) == self.test_y).float().mean().item()

            ctx = self.test_x[:, :PRED_CTX]
            target = self.test_x[:, PRED_CTX:PRED_CTX + PRED_LEN]
            _, pred_raw, pred_latent = m(ctx)
            self.baseline_pred_mse = ((pred_raw - target) ** 2).mean().item()
            self.baseline_pred_mae = (pred_raw - target).abs().mean().item()

            # Phase 3: latent prediction baseline. Target = uncompressed teacher's latent.
            target_latent = m.encode(self.test_x, pool_range=(PRED_CTX, PRED_CTX + PRED_LEN))
            self.baseline_latent_mse = ((pred_latent - target_latent) ** 2).mean().item()
            self.baseline_latent_mae = (pred_latent - target_latent).abs().mean().item()

        elapsed = time.time() - t0
        print(
            f"  Trained in {elapsed:.1f}s | Class acc: {self.baseline:.1%} | "
            f"RawMSE: {self.baseline_pred_mse:.4f} | LatMSE: {self.baseline_latent_mse:.4f} | "
            f"{self.n_params:,} params"
        )

        self._compute_sweeps()
        self.ready = True
        return {
            "baseline": round(self.baseline, 4),
            "baseline_pred_mse": round(self.baseline_pred_mse, 4),
            "baseline_latent_mse": round(self.baseline_latent_mse, 4),
            "params": self.n_params,
        }

    def _apply_compression(self, bits, latent_ratio, state_ratio):
        m = copy.deepcopy(self.model)
        if bits < 32:
            quantize(m, bits)
        active = max(1, int(LATENT_DIM * latent_ratio))
        lm = torch.zeros(LATENT_DIM)
        lm[:active] = 1.0
        window = max(1, int(SEQ_LEN * state_ratio)) if state_ratio < 1.0 else SEQ_LEN
        return m, lm, window, active

    def evaluate(self, bits=32.0, latent_ratio=1.0, state_ratio=1.0, mode="both"):
        m, lm, window, active = self._apply_compression(bits, latent_ratio, state_ratio)
        m.eval()

        result = {}

        with torch.no_grad():
            # Compressed online forward
            class_logits, _, _ = m(self.test_x, window=window, latent_mask=lm)
            preds = class_logits.argmax(1)
            probs = class_logits.softmax(1)
            acc = (preds == self.test_y).float().mean().item()
            conf = probs.max(1).values.mean().item()

            ctx = self.test_x[:, :PRED_CTX]
            target = self.test_x[:, PRED_CTX:PRED_CTX + PRED_LEN]
            pred_window = min(window, PRED_CTX)
            _, pred_values, pred_latent = m(ctx, window=pred_window, latent_mask=lm)
            pred_err = pred_values - target
            mse = (pred_err ** 2).mean().item()
            mae = pred_err.abs().mean().item()

            # Phase 3: target latent comes from the UNCOMPRESSED original model
            # (asymmetric setup — see Phase 3 plan key design decisions)
            target_latent = self.model.encode(self.test_x, pool_range=(PRED_CTX, PRED_CTX + PRED_LEN))
            latent_err = pred_latent - target_latent
            latent_mse = (latent_err ** 2).mean().item()
            latent_mae = latent_err.abs().mean().item()

        per_class_acc = {}
        per_class_mse = {}
        per_class_latent_mse = {}
        for i, name in enumerate(PATTERN_NAMES):
            mask = self.test_y == i
            if mask.sum():
                per_class_acc[name] = round((preds[mask] == self.test_y[mask]).float().mean().item(), 4)
                per_class_mse[name] = round(((pred_values[mask] - target[mask]) ** 2).mean().item(), 4)
                per_class_latent_mse[name] = round(((pred_latent[mask] - target_latent[mask]) ** 2).mean().item(), 4)

        eff_bits = min(bits, 32)
        mem_kb = self.n_params * eff_bits / 8 / 1024
        base_kb = self.n_params * 4 / 1024
        bc = 32 / max(eff_bits, 0.5)
        lc = 1.0 / max(latent_ratio, 1 / LATENT_DIM)
        sc = SEQ_LEN / max(window, 1)

        weights = []
        for p in m.parameters():
            if p.dim() >= 2:
                weights.extend(p.flatten().tolist()[:300])
                if len(weights) >= 500:
                    break
        weights = weights[:500]

        samples = []
        for i in range(min(8, len(self.test_x))):
            samples.append({
                "seq": [round(v, 3) for v in self.test_x[i].tolist()],
                "pred": PATTERN_NAMES[preds[i]],
                "actual": PATTERN_NAMES[self.test_y[i]],
                "conf": round(probs[i].max().item(), 3),
                "ok": bool(preds[i] == self.test_y[i]),
                "predicted_values": [round(v, 3) for v in pred_values[i].tolist()],
                "actual_values": [round(v, 3) for v in target[i].tolist()],
                "pred_mse": round(((pred_values[i] - target[i]) ** 2).mean().item(), 4),
                "latent_mse": round(((pred_latent[i] - target_latent[i]) ** 2).mean().item(), 4),
            })

        result.update({
            "accuracy": round(acc, 4),
            "baseline": round(self.baseline, 4),
            "retained": round(acc / max(self.baseline, 0.01), 4),
            "confidence": round(conf, 4),
            "per_class": per_class_acc,
            "memory_kb": round(mem_kb, 2),
            "baseline_kb": round(base_kb, 2),
            "compression": {
                "bits": round(bc, 1),
                "latent": round(lc, 1),
                "state": round(sc, 1),
                "total": round(bc * lc * sc, 1),
            },
            "settings": {"bits": bits, "latent_dim": active, "window": window},
            "weights": weights,
            "samples": samples,
            "prediction": {
                "mse": round(mse, 4),
                "mae": round(mae, 4),
                "baseline_mse": round(self.baseline_pred_mse, 4),
                "baseline_mae": round(self.baseline_pred_mae, 4),
                "retained": round(self.baseline_pred_mse / max(mse, 1e-6), 4),
                "per_class_mse": per_class_mse,
                "ctx_len": PRED_CTX,
                "pred_len": PRED_LEN,
                # Phase 3: latent-space prediction (JEPA-style)
                "latent_mse": round(latent_mse, 4),
                "latent_mae": round(latent_mae, 4),
                "latent_baseline_mse": round(self.baseline_latent_mse, 4),
                "latent_retained": round(self.baseline_latent_mse / max(latent_mse, 1e-6), 4),
                "per_class_latent_mse": per_class_latent_mse,
                # Divergence index: positive = raw degraded more than latent under compression
                "divergence_index": round(
                    (self.baseline_latent_mse / max(latent_mse, 1e-6))
                    - (self.baseline_pred_mse / max(mse, 1e-6)),
                    4,
                ),
            },
            "mode": mode,
        })

        return result

    def predict(self, sequence, bits=32.0, latent_ratio=1.0, state_ratio=1.0):
        """Live prediction: given a partial sequence, return next PRED_LEN values."""
        if not sequence:
            return {"error": "empty sequence"}

        m, lm, window, _ = self._apply_compression(bits, latent_ratio, state_ratio)
        m.eval()

        x = torch.tensor(sequence, dtype=torch.float32).unsqueeze(0)
        T = x.shape[1]
        eff_window = min(window, T)

        with torch.no_grad():
            class_logits, pred_values, _ = m(x, window=eff_window, latent_mask=lm)
            probs = class_logits.softmax(1)
            cls = int(class_logits.argmax(1).item())

        return {
            "input": [round(v, 3) for v in sequence],
            "predicted": [round(v, 3) for v in pred_values[0].tolist()],
            "predicted_class": PATTERN_NAMES[cls],
            "confidence": round(probs.max().item(), 3),
            "ctx_len": T,
            "pred_len": PRED_LEN,
        }

    def _compute_sweeps(self):
        self.sweeps = {"bits": [], "latent": [], "state": []}
        self.pred_sweeps = {"bits": [], "latent": [], "state": []}
        self.latent_sweeps = {"bits": [], "latent": [], "state": []}
        for b in BIT_LEVELS:
            r = self.evaluate(bits=b)
            self.sweeps["bits"].append({"x": b, "acc": r["accuracy"]})
            self.pred_sweeps["bits"].append({"x": b, "mse": r["prediction"]["mse"]})
            self.latent_sweeps["bits"].append({"x": b, "mse": r["prediction"]["latent_mse"]})
        for d in LATENT_DIMS:
            r = self.evaluate(latent_ratio=d / LATENT_DIM)
            self.sweeps["latent"].append({"x": d, "acc": r["accuracy"]})
            self.pred_sweeps["latent"].append({"x": d, "mse": r["prediction"]["mse"]})
            self.latent_sweeps["latent"].append({"x": d, "mse": r["prediction"]["latent_mse"]})
        for w in WINDOW_SIZES:
            r = self.evaluate(state_ratio=w / SEQ_LEN)
            self.sweeps["state"].append({"x": w, "acc": r["accuracy"]})
            self.pred_sweeps["state"].append({"x": w, "mse": r["prediction"]["mse"]})
            self.latent_sweeps["state"].append({"x": w, "mse": r["prediction"]["latent_mse"]})
        print("  Sweeps computed (classification + raw prediction + latent prediction).")
