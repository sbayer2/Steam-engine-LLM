"""
Steam Engine — Small-State Thesis Demo
Core neural network, data generation, and three-axis compression engine.

Key design: the model is trained with compression-awareness (random window
masking, latent dropout, weight noise injection) so that it learns
representations robust to all three axes. This is the honest version of
the thesis: structure survives brutal cuts WHEN the system is designed for it.
"""

import torch
import torch.nn as nn
import numpy as np
import copy
import time
from typing import Optional

SEQ_LEN = 16
HIDDEN_DIM = 64
LATENT_DIM = 64
N_CLASSES = 5
PATTERN_NAMES = ["Arithmetic", "Geometric", "Periodic", "Fibonacci", "Random"]

BIT_LEVELS = [32, 16, 8, 4, 2, 1.58, 1]
LATENT_DIMS = [64, 48, 32, 16, 8, 4, 2, 1]
WINDOW_SIZES = [16, 12, 8, 6, 4, 3, 2, 1]


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
    if p == 0:  # Arithmetic: constant slope, identifiable from any 2 points
        return rng.uniform(-2, 2) + rng.uniform(-0.5, 0.5) * t
    elif p == 1:  # Geometric: constant ratio, identifiable from 2-3 points
        return rng.uniform(0.5, 1.5) * np.power(rng.uniform(0.85, 1.15), t)
    elif p == 2:  # Periodic: sinusoidal, needs ~half-period to identify
        return rng.uniform(0.5, 1.5) * np.sin(rng.uniform(0.5, 2.5) * t + rng.uniform(0, 2 * np.pi))
    elif p == 3:  # Fibonacci-like: each value is mean of previous two
        s = [rng.uniform(-0.5, 0.5), rng.uniform(-0.5, 0.5)]
        for _ in range(SEQ_LEN - 2):
            s.append(0.5 * (s[-1] + s[-2]))
        return np.array(s)
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

    def forward(self, x, window=None, latent_mask=None):
        x = self.embed(x.unsqueeze(-1))
        x = self.norm1(x + self.attn1(x, window))
        x = self.norm2(x + self.attn2(x, window))
        z = self.enc(x)
        if latent_mask is not None:
            z = z * latent_mask
        x = self.norm3(self.dec(self.act(z)))
        return self.head(x.mean(1))


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
        self.sweeps = {}
        self.ready = False
        self.n_params = 0

    def train(self):
        t0 = time.time()
        X_tr, y_tr = generate_sequences(4000, seed=42)
        X_te, y_te = generate_sequences(500, seed=99)
        self.test_x = torch.from_numpy(X_te)
        self.test_y = torch.from_numpy(y_te)

        m = Model()
        self.n_params = sum(p.numel() for p in m.parameters())
        opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=2e-3)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, 250)
        loss_fn = nn.CrossEntropyLoss()
        tx = torch.from_numpy(X_tr)
        ty = torch.from_numpy(y_tr)

        m.train()
        rng = np.random.RandomState(7)
        for epoch in range(250):
            perm = torch.randperm(len(tx))
            for i in range(0, len(tx), 128):
                idx = perm[i:i + 128]

                # Compression-aware training after warmup
                window = None
                latent_mask = None
                if epoch >= 80:
                    # Random window restriction (axis 3 robustness)
                    if rng.random() < 0.4:
                        window = int(rng.choice([3, 4, 6, 8, 12]))
                    # Random latent masking (axis 2 robustness)
                    if rng.random() < 0.4:
                        keep = rng.randint(max(4, LATENT_DIM // 4), LATENT_DIM)
                        latent_mask = torch.zeros(LATENT_DIM)
                        latent_mask[:keep] = 1.0

                opt.zero_grad()
                loss_fn(m(tx[idx], window=window, latent_mask=latent_mask), ty[idx]).backward()
                opt.step()

            # Weight noise injection (axis 1 robustness) — simulates quantization
            if epoch >= 80 and epoch % 5 == 0:
                with torch.no_grad():
                    for p in m.parameters():
                        if p.dim() >= 2:
                            noise_scale = p.abs().mean() * 0.05
                            p.add_(torch.randn_like(p) * noise_scale)

            sched.step()

        m.eval()
        self.model = m
        with torch.no_grad():
            self.baseline = (m(self.test_x).argmax(1) == self.test_y).float().mean().item()

        elapsed = time.time() - t0
        print(f"  Trained in {elapsed:.1f}s | Baseline accuracy: {self.baseline:.1%} | {self.n_params:,} params")

        self._compute_sweeps()
        self.ready = True
        return {"baseline": round(self.baseline, 4), "params": self.n_params}

    def evaluate(self, bits=32.0, latent_ratio=1.0, state_ratio=1.0):
        m = copy.deepcopy(self.model)

        if bits < 32:
            quantize(m, bits)

        active = max(1, int(LATENT_DIM * latent_ratio))
        lm = torch.zeros(LATENT_DIM)
        lm[:active] = 1.0

        window = max(1, int(SEQ_LEN * state_ratio)) if state_ratio < 1.0 else SEQ_LEN

        m.eval()
        with torch.no_grad():
            logits = m(self.test_x, window=window, latent_mask=lm)
            preds = logits.argmax(1)
            probs = logits.softmax(1)
            acc = (preds == self.test_y).float().mean().item()
            conf = probs.max(1).values.mean().item()

        per_class = {}
        for i, name in enumerate(PATTERN_NAMES):
            mask = self.test_y == i
            if mask.sum():
                per_class[name] = round((preds[mask] == self.test_y[mask]).float().mean().item(), 4)

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
            })

        return {
            "accuracy": round(acc, 4),
            "baseline": round(self.baseline, 4),
            "retained": round(acc / max(self.baseline, 0.01), 4),
            "confidence": round(conf, 4),
            "per_class": per_class,
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
        }

    def _compute_sweeps(self):
        self.sweeps = {"bits": [], "latent": [], "state": []}
        for b in BIT_LEVELS:
            r = self.evaluate(bits=b)
            self.sweeps["bits"].append({"x": b, "acc": r["accuracy"]})
        for d in LATENT_DIMS:
            r = self.evaluate(latent_ratio=d / LATENT_DIM)
            self.sweeps["latent"].append({"x": d, "acc": r["accuracy"]})
        for w in WINDOW_SIZES:
            r = self.evaluate(state_ratio=w / SEQ_LEN)
            self.sweeps["state"].append({"x": w, "acc": r["accuracy"]})
        print("  Sweeps computed.")
