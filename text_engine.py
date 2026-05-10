"""
Steam Engine — Phase 4 Text Model (Step 1)

A minimal char-level GPT trained on TinyShakespeare, instrumented with the same
three-axis compression engine as the synthetic toy. Architecture borrowed in
spirit from Karpathy's nanochat (single complexity dial: depth) but stripped
to bare essentials and scaled small for CPU tractability.

Phase 4 questions:
- Does the compression-aware training curriculum work on real text?
- Does Phase 3's H_D finding (latent prediction more compression-fragile than
  raw) replicate when raw is next-token cross-entropy on natural language?

Step 1 scope: LM-only (next-token prediction). Latent prediction head added in
Step 2.
"""

import copy
import math
import os
import time
import urllib.request
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------------------------------------------------------------
# Config — single complexity dial style, borrowed from nanochat

class TextConfig:
    """Inspired by nanochat.GPTConfig. Single primary dial: n_layer (depth)."""
    def __init__(self,
                 n_layer: int = 2,
                 n_embd: int = 64,
                 n_head: int = 4,
                 seq_len: int = 256,
                 vocab_size: int = 100,        # set after tokenizer build
                 dropout: float = 0.0):
        self.n_layer = n_layer
        self.n_embd = n_embd
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.dropout = dropout
        assert n_embd % n_head == 0, "n_embd must be divisible by n_head"


BIT_LEVELS = [32, 16, 8, 4, 2, 1.58, 1]
LATENT_DIMS = [64, 48, 32, 16, 8, 4, 2, 1]   # ratios of n_embd
WINDOW_SIZES = [256, 192, 128, 96, 64, 32, 16, 8, 4]  # ratios of seq_len


# -----------------------------------------------------------------------------
# Char-level tokenizer

class CharTokenizer:
    def __init__(self, text: str):
        chars = sorted(set(text))
        self.vocab = chars
        self.vocab_size = len(chars)
        self.stoi = {c: i for i, c in enumerate(chars)}
        self.itos = {i: c for i, c in enumerate(chars)}

    def encode(self, s: str) -> list[int]:
        return [self.stoi.get(c, 0) for c in s]

    def decode(self, ids) -> str:
        return "".join(self.itos.get(int(i), "?") for i in ids)


# -----------------------------------------------------------------------------
# Attention with optional sliding window (same idea as synthetic engine.py)

class CausalAttention(nn.Module):
    def __init__(self, cfg: TextConfig):
        super().__init__()
        self.cfg = cfg
        self.qkv = nn.Linear(cfg.n_embd, cfg.n_embd * 3, bias=False)
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd)
        self.scale = cfg.head_dim ** -0.5

    def forward(self, x, window: Optional[int] = None, latent_mask: Optional[torch.Tensor] = None):
        B, T, C = x.shape
        H, D = self.cfg.n_head, self.cfg.head_dim

        qkv = self.qkv(x).reshape(B, T, 3, H, D).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # each (B, H, T, D)

        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, H, T, T)

        # causal mask
        causal = torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool))
        attn = attn.masked_fill(~causal, float("-inf"))

        # sliding window: keep only attention to positions within [i - window + 1, i]
        if window is not None and window < T:
            positions = torch.arange(T, device=x.device)
            row = positions.unsqueeze(1)
            col = positions.unsqueeze(0)
            in_window = (row - col) < window  # within window in causal direction
            attn = attn.masked_fill(~in_window, float("-inf"))

        attn = F.softmax(attn, dim=-1)
        out = attn @ v  # (B, H, T, D)
        out = out.transpose(1, 2).reshape(B, T, C)

        # Optional latent-dim compression: zero out a contiguous tail of channels
        if latent_mask is not None:
            out = out * latent_mask

        return self.proj(out)


class MLP(nn.Module):
    def __init__(self, cfg: TextConfig):
        super().__init__()
        self.fc1 = nn.Linear(cfg.n_embd, cfg.n_embd * 4)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(cfg.n_embd * 4, cfg.n_embd)

    def forward(self, x, latent_mask: Optional[torch.Tensor] = None):
        x = self.fc2(self.act(self.fc1(x)))
        if latent_mask is not None:
            x = x * latent_mask
        return x


class Block(nn.Module):
    def __init__(self, cfg: TextConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.n_embd)
        self.attn = CausalAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.n_embd)
        self.mlp = MLP(cfg)

    def forward(self, x, window=None, latent_mask=None):
        x = x + self.attn(self.ln1(x), window=window, latent_mask=latent_mask)
        x = x + self.mlp(self.ln2(x), latent_mask=latent_mask)
        return x


class TextGPT(nn.Module):
    def __init__(self, cfg: TextConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.pos_emb = nn.Embedding(cfg.seq_len, cfg.n_embd)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.n_embd)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        # tie weights — saves params, matches nanoGPT/nanochat pattern
        self.lm_head.weight = self.tok_emb.weight

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, window=None, latent_mask=None, targets=None):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.tok_emb(idx) + self.pos_emb(pos)[None, :, :]
        for block in self.blocks:
            x = block(x, window=window, latent_mask=latent_mask)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        if targets is None:
            return logits, None
        loss = F.cross_entropy(logits.reshape(-1, self.cfg.vocab_size), targets.reshape(-1))
        return logits, loss


# -----------------------------------------------------------------------------
# Quantization (same recipe as synthetic engine.py)

def quantize(model: nn.Module, bits: float):
    """Post-training quantization with group-wise optimal scaling."""
    sd = model.state_dict()
    for k, v in sd.items():
        if v.dim() < 2:
            continue
        if bits <= 1.0:  # binary
            scale = v.abs().mean()
            q = torch.sign(v)
            q[q == 0] = 1.0
            sd[k] = q * scale
        elif bits <= 1.58:  # ternary group-wise
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
        else:
            levels = 2 ** int(bits)
            mn, mx = v.min(), v.max()
            if mn == mx:
                continue
            step = (mx - mn) / (levels - 1)
            sd[k] = torch.round((v - mn) / step) * step + mn
    model.load_state_dict(sd)


# -----------------------------------------------------------------------------
# Data

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
TINYSHAKESPEARE_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"


def load_tinyshakespeare() -> str:
    os.makedirs(DATA_DIR, exist_ok=True)
    path = os.path.join(DATA_DIR, "tinyshakespeare.txt")
    if not os.path.exists(path):
        print(f"  Fetching TinyShakespeare → {path}")
        urllib.request.urlretrieve(TINYSHAKESPEARE_URL, path)
    with open(path, "r") as f:
        return f.read()


def make_batches(data: torch.Tensor, batch_size: int, seq_len: int, rng: np.random.RandomState):
    """Random crops of length seq_len from a 1-D token tensor."""
    n = len(data) - seq_len - 1
    starts = rng.randint(0, n, size=batch_size)
    x = torch.stack([data[s:s + seq_len] for s in starts])
    y = torch.stack([data[s + 1:s + seq_len + 1] for s in starts])
    return x, y


# -----------------------------------------------------------------------------
# Engine — mirrors engine.py's API shape

class TextEngine:
    def __init__(self):
        self.model: Optional[TextGPT] = None
        self.cfg: Optional[TextConfig] = None
        self.tokenizer: Optional[CharTokenizer] = None
        self.train_data: Optional[torch.Tensor] = None
        self.val_data: Optional[torch.Tensor] = None
        self.baseline_loss = float("inf")
        self.baseline_perplexity = float("inf")
        self.sweeps = {}
        self.ready = False
        self.n_params = 0

    def train(self, n_layer: int = 2, n_embd: int = 64, n_head: int = 4,
              seq_len: int = 256, epochs: int = 200, batch_size: int = 32,
              steps_per_epoch: int = 32, warmup_frac: float = 0.30,
              lr: float = 3e-4, model_seed: Optional[int] = None,
              aug_seed: int = 7):
        t0 = time.time()
        if model_seed is not None:
            torch.manual_seed(model_seed)

        text = load_tinyshakespeare()
        self.tokenizer = CharTokenizer(text)
        n_chars = len(text)
        # 90/10 train/val split on contiguous text
        split = int(n_chars * 0.9)
        train_text, val_text = text[:split], text[split:]
        self.train_data = torch.tensor(self.tokenizer.encode(train_text), dtype=torch.long)
        self.val_data = torch.tensor(self.tokenizer.encode(val_text), dtype=torch.long)

        self.cfg = TextConfig(
            n_layer=n_layer, n_embd=n_embd, n_head=n_head,
            seq_len=seq_len, vocab_size=self.tokenizer.vocab_size,
        )
        self.model = TextGPT(self.cfg)
        self.n_params = sum(p.numel() for p in self.model.parameters())

        opt = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-2)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
        rng = np.random.RandomState(aug_seed)
        warmup_epoch = int(epochs * warmup_frac)

        self.model.train()
        for epoch in range(epochs):
            for _ in range(steps_per_epoch):
                window = None
                latent_mask = None
                if epoch >= warmup_epoch:
                    if rng.random() < 0.4:
                        # window choices proportional to seq_len, like Toy v2
                        window = int(rng.choice([8, 16, 32, 64, 128]))
                    if rng.random() < 0.4:
                        keep = rng.randint(max(8, n_embd // 4), n_embd)
                        latent_mask = torch.zeros(n_embd)
                        latent_mask[:keep] = 1.0

                x, y = make_batches(self.train_data, batch_size, seq_len, rng)
                opt.zero_grad()
                _, loss = self.model(x, window=window, latent_mask=latent_mask, targets=y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                opt.step()

            # weight noise injection (axis 1 robustness)
            if epoch >= warmup_epoch and epoch % 5 == 0:
                with torch.no_grad():
                    for p in self.model.parameters():
                        if p.dim() >= 2:
                            noise_scale = p.abs().mean() * 0.05
                            p.add_(torch.randn_like(p) * noise_scale)

            sched.step()

        self.model.eval()
        with torch.no_grad():
            val_loss = self._compute_val_loss(self.model)
            self.baseline_loss = val_loss
            self.baseline_perplexity = math.exp(val_loss)

        elapsed = time.time() - t0
        print(
            f"  Trained in {elapsed:.1f}s | val loss {self.baseline_loss:.4f} | "
            f"perplexity {self.baseline_perplexity:.2f} | {self.n_params:,} params"
        )

        self._compute_sweeps()
        self.ready = True
        return {
            "baseline_loss": round(self.baseline_loss, 4),
            "baseline_perplexity": round(self.baseline_perplexity, 4),
            "params": self.n_params,
            "vocab_size": self.tokenizer.vocab_size,
        }

    def _compute_val_loss(self, model: TextGPT, window=None, latent_mask=None,
                          n_batches: int = 16, batch_size: int = 32) -> float:
        rng = np.random.RandomState(123)
        seq_len = self.cfg.seq_len
        losses = []
        with torch.no_grad():
            for _ in range(n_batches):
                x, y = make_batches(self.val_data, batch_size, seq_len, rng)
                _, loss = model(x, window=window, latent_mask=latent_mask, targets=y)
                losses.append(loss.item())
        return sum(losses) / len(losses)

    def _apply_compression(self, bits, latent_ratio, state_ratio):
        m = copy.deepcopy(self.model)
        if bits < 32:
            quantize(m, bits)
        n_embd = self.cfg.n_embd
        active = max(1, int(n_embd * latent_ratio))
        lm = torch.zeros(n_embd)
        lm[:active] = 1.0
        seq_len = self.cfg.seq_len
        window = max(1, int(seq_len * state_ratio)) if state_ratio < 1.0 else seq_len
        return m, lm, window, active

    def evaluate(self, bits: float = 32.0, latent_ratio: float = 1.0, state_ratio: float = 1.0):
        m, lm, window, active = self._apply_compression(bits, latent_ratio, state_ratio)
        m.eval()
        loss = self._compute_val_loss(m, window=window, latent_mask=lm)
        perplexity = math.exp(loss)

        eff_bits = min(bits, 32)
        mem_kb = self.n_params * eff_bits / 8 / 1024
        base_kb = self.n_params * 4 / 1024
        bc = 32 / max(eff_bits, 0.5)
        lc = 1.0 / max(latent_ratio, 1 / self.cfg.n_embd)
        sc = self.cfg.seq_len / max(window, 1)

        return {
            "loss": round(loss, 4),
            "perplexity": round(perplexity, 4),
            "baseline_loss": round(self.baseline_loss, 4),
            "baseline_perplexity": round(self.baseline_perplexity, 4),
            "loss_retained": round(self.baseline_loss / max(loss, 1e-6), 4),
            "perplexity_retained": round(self.baseline_perplexity / max(perplexity, 1e-6), 4),
            "memory_kb": round(mem_kb, 2),
            "baseline_kb": round(base_kb, 2),
            "compression": {
                "bits": round(bc, 1),
                "latent": round(lc, 1),
                "state": round(sc, 1),
                "total": round(bc * lc * sc, 1),
            },
            "settings": {"bits": bits, "latent_dim": active, "window": window},
        }

    def generate(self, prompt: str, max_new: int = 200, temperature: float = 1.0,
                 bits: float = 32.0, latent_ratio: float = 1.0, state_ratio: float = 1.0):
        m, lm, window, _ = self._apply_compression(bits, latent_ratio, state_ratio)
        m.eval()
        idx = torch.tensor(self.tokenizer.encode(prompt), dtype=torch.long).unsqueeze(0)

        with torch.no_grad():
            for _ in range(max_new):
                # crop context to seq_len from the right
                ctx = idx[:, -self.cfg.seq_len:]
                eff_window = min(window, ctx.shape[1])
                logits, _ = m(ctx, window=eff_window, latent_mask=lm)
                logits = logits[:, -1, :] / max(temperature, 1e-6)
                probs = F.softmax(logits, dim=-1)
                next_tok = torch.multinomial(probs, num_samples=1)
                idx = torch.cat([idx, next_tok], dim=1)
        return self.tokenizer.decode(idx[0].tolist())

    def _compute_sweeps(self):
        n_embd = self.cfg.n_embd
        seq_len = self.cfg.seq_len
        self.sweeps = {"bits": [], "latent": [], "state": []}
        for b in BIT_LEVELS:
            r = self.evaluate(bits=b)
            self.sweeps["bits"].append({"x": b, "loss": r["loss"], "perplexity": r["perplexity"]})
        for d in LATENT_DIMS:
            r = self.evaluate(latent_ratio=d / n_embd)
            self.sweeps["latent"].append({"x": d, "loss": r["loss"], "perplexity": r["perplexity"]})
        for w in WINDOW_SIZES:
            r = self.evaluate(state_ratio=w / seq_len)
            self.sweeps["state"].append({"x": w, "loss": r["loss"], "perplexity": r["perplexity"]})
        print("  Sweeps computed (loss + perplexity over 3 axes).")


# -----------------------------------------------------------------------------
# CLI smoke test

if __name__ == "__main__":
    import sys
    epochs = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    e = TextEngine()
    e.train(epochs=epochs, model_seed=0, aug_seed=0)
    print()
    print("Generation sample (no compression):")
    print(e.generate("ROMEO:\n", max_new=200, temperature=0.8))
    print()
    print("Generation sample (4-bit, 32d latent, w=64):")
    print(e.generate("ROMEO:\n", max_new=200, temperature=0.8,
                     bits=4, latent_ratio=32 / e.cfg.n_embd,
                     state_ratio=64 / e.cfg.seq_len))
