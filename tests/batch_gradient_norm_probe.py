"""
Gradient-norm probe — weight-perturbation landscape measurement.

Replaces the deprecated input-perturbation probe (ADC-007, Batch D) which
measured input-output sensitivity rather than landscape jaggedness.

This probe measures three things at each compression point:

1. Per-head gradient norm: ||∇_θ L_head|| for each loss component.
   Does compression increase or decrease gradient magnitude?

2. Gradient alignment vs baseline: cosine similarity between the gradient
   vector at full precision and the gradient vector under compression.
   High cos-sim → smooth transition (optimization direction preserved).
   Low cos-sim → jagged cliff (compression redirected the loss landscape).

3. Weight-perturbation sensitivity: add N small random perturbations to the
   already-compressed model, recompute gradient norm, measure variance.
   High variance → jagged landscape → bad for gradient-based planning.
   Low variance → smooth → gradient-based planning still viable.

Dream-MPC's specific claim: latent-space prediction provides smoother
gradients than raw-space prediction. If true, the latent head's gradient
should show higher alignment and lower perturbation sensitivity than the
raw head's gradient under identical compression.

Runs on both engines. ~10-15 min total.
"""

import copy
import json
import os
import sys
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine import Engine, Model, quantize as synth_quantize, PRED_CTX, PRED_LEN, LATENT_DIM
from text_engine import TextEngine, TextGPT, quantize as text_quantize, make_batches
from text_engine import PRED_CTX as TEXT_PRED_CTX, PRED_LEN as TEXT_PRED_LEN

N_PERTURBATIONS = 16
EPSILONS = [0.01, 0.03, 0.05]


# --- Utilities ---

def flatten_grad(model: nn.Module) -> torch.Tensor:
    """Concatenate all parameter gradients into a single vector."""
    grads = []
    for p in model.parameters():
        if p.grad is not None:
            grads.append(p.grad.flatten())
        else:
            grads.append(torch.zeros(p.numel()))
    return torch.cat(grads)


def grad_norm(model: nn.Module) -> float:
    g = flatten_grad(model)
    return g.norm().item()


def cosine_sim(g1: torch.Tensor, g2: torch.Tensor) -> float:
    dot = (g1 * g2).sum()
    denom = g1.norm() * g2.norm()
    if denom < 1e-12:
        return 0.0
    return (dot / denom).item()


def perturb_weights(model: nn.Module, epsilon: float, rng: np.random.RandomState):
    """Add Gaussian noise to all 2D+ parameters, scaled by per-param magnitude."""
    with torch.no_grad():
        for p in model.parameters():
            if p.dim() >= 2:
                scale = p.abs().mean() * epsilon
                p.add_(torch.randn_like(p) * scale)


# --- Synthetic engine probe ---

def probe_synthetic(engine: Engine):
    """Compute gradient-norm measurements on the synthetic Toy v2 model."""
    model = engine.model
    test_x = engine.test_x
    test_y = engine.test_y
    ce_fn = nn.CrossEntropyLoss()
    mse_fn = nn.MSELoss()

    COMPRESSION_POINTS = [
        ("baseline", 32, 1.0, 1.0),
        ("bits_4", 4, 1.0, 1.0),
        ("bits_2", 2, 1.0, 1.0),
        ("bits_158", 1.58, 1.0, 1.0),
        ("latent_16", 32, 16 / 64, 1.0),
        ("latent_8", 32, 8 / 64, 1.0),
        ("cliff_171x", 2, 48 / 64, 8 / 64),
        ("max_131k", 1, 1 / 64, 1 / 64),
    ]

    # Sample a fixed batch for consistency
    rng = np.random.RandomState(42)
    idx = rng.choice(len(test_x), 256, replace=False)
    bx = test_x[idx]
    by = test_y[idx]

    def compute_per_head_grads(m, window, latent_mask):
        """Returns (grad_ce, grad_raw, grad_latent) as flattened vectors."""
        m.train()
        grads = {}
        for head_name, loss_fn in [("ce", "ce"), ("raw", "raw"), ("latent", "latent")]:
            m.zero_grad()
            class_logits, pred_raw, pred_latent = m(bx, window=window, latent_mask=latent_mask)

            if head_name == "ce":
                loss = ce_fn(class_logits, by)
            elif head_name == "raw":
                ctx = bx[:, :PRED_CTX]
                target_raw = bx[:, PRED_CTX:PRED_CTX + PRED_LEN]
                _, pred_r, _ = m(ctx, window=min(window, PRED_CTX) if window else None,
                                 latent_mask=latent_mask)
                loss = mse_fn(pred_r, target_raw)
            else:
                ctx = bx[:, :PRED_CTX]
                _, _, pred_l = m(ctx, window=min(window, PRED_CTX) if window else None,
                                 latent_mask=latent_mask)
                with torch.no_grad():
                    target_latent = model.encode(bx, pool_range=(PRED_CTX, PRED_CTX + PRED_LEN))
                loss = mse_fn(pred_l, target_latent)

            loss.backward()
            grads[head_name] = flatten_grad(m).clone()
        m.eval()
        return grads

    # Compute baseline gradients first
    baseline_grads = compute_per_head_grads(copy.deepcopy(model), None, None)

    results = []
    for name, bits, lr, sr in COMPRESSION_POINTS:
        m = copy.deepcopy(model)
        if bits < 32:
            synth_quantize(m, bits)
        active = max(1, int(LATENT_DIM * lr))
        lm = torch.zeros(LATENT_DIM)
        lm[:active] = 1.0
        window = max(1, int(64 * sr)) if sr < 1.0 else None

        # 1. Per-head gradient norms + alignment vs baseline
        grads = compute_per_head_grads(copy.deepcopy(m), window, lm if lr < 1.0 else None)
        point = {
            "name": name,
            "compression": {"bits": bits, "latent_ratio": lr, "state_ratio": sr},
        }
        for head in ["ce", "raw", "latent"]:
            point[f"{head}_grad_norm"] = round(grads[head].norm().item(), 6)
            point[f"{head}_cos_sim_vs_baseline"] = round(
                cosine_sim(grads[head], baseline_grads[head]), 6)

        # 2. Weight-perturbation sensitivity
        for eps in EPSILONS:
            perturbed_norms = {h: [] for h in ["ce", "raw", "latent"]}
            for _ in range(N_PERTURBATIONS):
                mp = copy.deepcopy(m)
                perturb_weights(mp, eps, np.random.RandomState())
                pg = compute_per_head_grads(mp, window, lm if lr < 1.0 else None)
                for h in ["ce", "raw", "latent"]:
                    perturbed_norms[h].append(pg[h].norm().item())
            for h in ["ce", "raw", "latent"]:
                norms = perturbed_norms[h]
                mean_n = sum(norms) / len(norms)
                std_n = (sum((n - mean_n) ** 2 for n in norms) / len(norms)) ** 0.5
                ref_norm = point[f"{h}_grad_norm"]
                cv = std_n / max(ref_norm, 1e-12)
                point[f"{h}_perturb_eps{eps}_mean"] = round(mean_n, 6)
                point[f"{h}_perturb_eps{eps}_std"] = round(std_n, 6)
                point[f"{h}_perturb_eps{eps}_cv"] = round(cv, 6)

        results.append(point)
    return results


# --- Text engine probe ---

def probe_text(engine: TextEngine):
    """Compute gradient-norm measurements on the text GPT model."""
    model = engine.model
    cfg = engine.cfg

    COMPRESSION_POINTS = [
        ("baseline", 32, 1.0, 1.0),
        ("bits_4", 4, 1.0, 1.0),
        ("bits_2", 2, 1.0, 1.0),
        ("bits_158", 1.58, 1.0, 1.0),
        ("latent_16", 32, 16 / 64, 1.0),
        ("latent_8", 32, 8 / 64, 1.0),
        ("cliff_171x", 2, 32 / 64, 32 / 256),
        ("max", 1, 1 / 64, 1 / 256),
    ]

    # Fixed val batch
    rng = np.random.RandomState(42)
    val_data = engine.val_data if not engine.multi_corpus else list(engine.corpora_val.values())[0]
    x, y = make_batches(val_data, 64, cfg.seq_len, rng)

    def compute_per_head_grads(m, window, latent_mask):
        """Returns (grad_lm, grad_latent) as flattened vectors."""
        m.train()
        grads = {}
        for head_name in ["lm", "latent"]:
            m.zero_grad()
            if head_name == "lm":
                _, _, loss = m(x, window=window, latent_mask=latent_mask, targets=y)
            else:
                _, hidden, _ = m(x, window=window, latent_mask=latent_mask, targets=y)
                h_ctx = hidden[:, TEXT_PRED_CTX - 1, :]
                pred_latent = m.latent_pred_head(h_ctx)
                with torch.no_grad():
                    _, target_hidden, _ = model(x)
                    z_target = target_hidden[:, TEXT_PRED_CTX:TEXT_PRED_CTX + TEXT_PRED_LEN, :].mean(1)
                loss = F.mse_loss(pred_latent, z_target)

            loss.backward()
            grads[head_name] = flatten_grad(m).clone()
        m.eval()
        return grads

    # Baseline gradients
    baseline_grads = compute_per_head_grads(copy.deepcopy(model), None, None)

    results = []
    for name, bits, lr, sr in COMPRESSION_POINTS:
        m = copy.deepcopy(model)
        if bits < 32:
            text_quantize(m, bits)
        n_embd = cfg.n_embd
        active = max(1, int(n_embd * lr))
        lm = torch.zeros(n_embd)
        lm[:active] = 1.0
        window = max(1, int(cfg.seq_len * sr)) if sr < 1.0 else None

        grads = compute_per_head_grads(copy.deepcopy(m), window, lm if lr < 1.0 else None)
        point = {
            "name": name,
            "compression": {"bits": bits, "latent_ratio": lr, "state_ratio": sr},
        }
        for head in ["lm", "latent"]:
            point[f"{head}_grad_norm"] = round(grads[head].norm().item(), 6)
            point[f"{head}_cos_sim_vs_baseline"] = round(
                cosine_sim(grads[head], baseline_grads[head]), 6)

        # Weight-perturbation sensitivity
        for eps in EPSILONS:
            perturbed_norms = {h: [] for h in ["lm", "latent"]}
            for _ in range(N_PERTURBATIONS):
                mp = copy.deepcopy(m)
                perturb_weights(mp, eps, np.random.RandomState())
                pg = compute_per_head_grads(mp, window, lm if lr < 1.0 else None)
                for h in ["lm", "latent"]:
                    perturbed_norms[h].append(pg[h].norm().item())
            for h in ["lm", "latent"]:
                norms = perturbed_norms[h]
                mean_n = sum(norms) / len(norms)
                std_n = (sum((n - mean_n) ** 2 for n in norms) / len(norms)) ** 0.5
                ref_norm = point[f"{h}_grad_norm"]
                cv = std_n / max(ref_norm, 1e-12)
                point[f"{h}_perturb_eps{eps}_mean"] = round(mean_n, 6)
                point[f"{h}_perturb_eps{eps}_std"] = round(std_n, 6)
                point[f"{h}_perturb_eps{eps}_cv"] = round(cv, 6)

        results.append(point)
    return results


# --- Main ---

def main():
    print("=" * 80)
    print("GRADIENT-NORM PROBE — Weight-Perturbation Landscape Measurement")
    print("=" * 80)
    print(f"\nEpsilons: {EPSILONS}")
    print(f"Perturbations per point: {N_PERTURBATIONS}")
    print(f"\nDream-MPC hypothesis: latent head has smoother gradient landscape")
    print(f"  → higher cos-sim vs baseline, lower perturbation CV\n")

    overall_t0 = time.time()
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "batch_gradient_norm_results.json")

    # 1. Train synthetic engine
    print("--- Training synthetic Toy v2 ---")
    synth_engine = Engine()
    synth_engine.train(model_seed=0, aug_seed=0)
    print(f"  Baseline: acc={synth_engine.baseline:.3f}, "
          f"raw_mse={synth_engine.baseline_pred_mse:.4f}, "
          f"lat_mse={synth_engine.baseline_latent_mse:.4f}")

    # 2. Train text engine
    print("\n--- Training text engine (single-corpus Shakespeare) ---")
    text_eng = TextEngine()
    text_eng.train(epochs=200, model_seed=0, aug_seed=0, beta=0.5, multi_corpus=False)

    # 3. Probe synthetic
    print("\n--- Probing synthetic model ---")
    t0 = time.time()
    synth_results = probe_synthetic(synth_engine)
    print(f"  Synthetic probe done in {time.time()-t0:.1f}s")

    # 4. Probe text
    print("\n--- Probing text model ---")
    t0 = time.time()
    text_results = probe_text(text_eng)
    print(f"  Text probe done in {time.time()-t0:.1f}s")

    # Save
    output = {
        "experiment": "gradient_norm_probe",
        "started": datetime.now().isoformat(),
        "elapsed_s": round(time.time() - overall_t0, 1),
        "config": {"epsilons": EPSILONS, "n_perturbations": N_PERTURBATIONS},
        "synthetic": synth_results,
        "text": text_results,
    }
    with open(out, "w") as f:
        json.dump(output, f, indent=2)

    # === Summary tables ===
    print("\n" + "=" * 90)
    print("SYNTHETIC — Gradient Alignment & Perturbation Sensitivity (ε=0.03)")
    print("=" * 90)
    eps_key = "0.03"
    print(f"\n{'point':>12} {'ce_norm':>9} {'raw_norm':>9} {'lat_norm':>9} "
          f"{'ce_cos':>8} {'raw_cos':>8} {'lat_cos':>8} "
          f"{'ce_cv':>7} {'raw_cv':>7} {'lat_cv':>7}")
    print("-" * 100)
    for pt in synth_results:
        print(f"{pt['name']:>12} "
              f"{pt['ce_grad_norm']:>9.4f} {pt['raw_grad_norm']:>9.4f} {pt['latent_grad_norm']:>9.4f} "
              f"{pt['ce_cos_sim_vs_baseline']:>8.4f} {pt['raw_cos_sim_vs_baseline']:>8.4f} {pt['latent_cos_sim_vs_baseline']:>8.4f} "
              f"{pt[f'ce_perturb_eps{eps_key}_cv']:>7.4f} {pt[f'raw_perturb_eps{eps_key}_cv']:>7.4f} {pt[f'latent_perturb_eps{eps_key}_cv']:>7.4f}")

    print("\n" + "=" * 90)
    print("TEXT — Gradient Alignment & Perturbation Sensitivity (ε=0.03)")
    print("=" * 90)
    print(f"\n{'point':>12} {'lm_norm':>9} {'lat_norm':>9} "
          f"{'lm_cos':>8} {'lat_cos':>8} "
          f"{'lm_cv':>7} {'lat_cv':>7}")
    print("-" * 70)
    for pt in text_results:
        print(f"{pt['name']:>12} "
              f"{pt['lm_grad_norm']:>9.4f} {pt['latent_grad_norm']:>9.4f} "
              f"{pt['lm_cos_sim_vs_baseline']:>8.4f} {pt['latent_cos_sim_vs_baseline']:>8.4f} "
              f"{pt[f'lm_perturb_eps{eps_key}_cv']:>7.4f} {pt[f'latent_perturb_eps{eps_key}_cv']:>7.4f}")

    # Dream-MPC verdict
    print("\n" + "=" * 90)
    print("DREAM-MPC HYPOTHESIS TEST")
    print("=" * 90)

    # On synthetic: compare latent vs raw cos-sim and CV
    print("\nSynthetic — latent vs raw smoothness:")
    for pt in synth_results:
        if pt["name"] == "baseline":
            continue
        lat_smoother_cos = pt["latent_cos_sim_vs_baseline"] > pt["raw_cos_sim_vs_baseline"]
        lat_smoother_cv = pt[f"latent_perturb_eps{eps_key}_cv"] < pt[f"raw_perturb_eps{eps_key}_cv"]
        verdict = "LATENT SMOOTHER" if (lat_smoother_cos and lat_smoother_cv) else (
            "RAW SMOOTHER" if (not lat_smoother_cos and not lat_smoother_cv) else "MIXED")
        print(f"  {pt['name']:>12}: cos(lat={pt['latent_cos_sim_vs_baseline']:.4f} vs raw={pt['raw_cos_sim_vs_baseline']:.4f}) "
              f"cv(lat={pt[f'latent_perturb_eps{eps_key}_cv']:.4f} vs raw={pt[f'raw_perturb_eps{eps_key}_cv']:.4f}) → {verdict}")

    # On text: compare latent vs LM cos-sim and CV
    print("\nText — latent vs LM smoothness:")
    for pt in text_results:
        if pt["name"] == "baseline":
            continue
        lat_smoother_cos = pt["latent_cos_sim_vs_baseline"] > pt["lm_cos_sim_vs_baseline"]
        lat_smoother_cv = pt[f"latent_perturb_eps{eps_key}_cv"] < pt[f"lm_perturb_eps{eps_key}_cv"]
        verdict = "LATENT SMOOTHER" if (lat_smoother_cos and lat_smoother_cv) else (
            "LM SMOOTHER" if (not lat_smoother_cos and not lat_smoother_cv) else "MIXED")
        print(f"  {pt['name']:>12}: cos(lat={pt['latent_cos_sim_vs_baseline']:.4f} vs lm={pt['lm_cos_sim_vs_baseline']:.4f}) "
              f"cv(lat={pt[f'latent_perturb_eps{eps_key}_cv']:.4f} vs lm={pt[f'lm_perturb_eps{eps_key}_cv']:.4f}) → {verdict}")

    # Overall
    synth_lat_wins_cos = sum(
        1 for pt in synth_results if pt["name"] != "baseline"
        and pt["latent_cos_sim_vs_baseline"] > pt["raw_cos_sim_vs_baseline"])
    synth_lat_wins_cv = sum(
        1 for pt in synth_results if pt["name"] != "baseline"
        and pt[f"latent_perturb_eps{eps_key}_cv"] < pt[f"raw_perturb_eps{eps_key}_cv"])
    text_lat_wins_cos = sum(
        1 for pt in text_results if pt["name"] != "baseline"
        and pt["latent_cos_sim_vs_baseline"] > pt["lm_cos_sim_vs_baseline"])
    text_lat_wins_cv = sum(
        1 for pt in text_results if pt["name"] != "baseline"
        and pt[f"latent_perturb_eps{eps_key}_cv"] < pt[f"lm_perturb_eps{eps_key}_cv"])
    n_synth = len(synth_results) - 1
    n_text = len(text_results) - 1

    print(f"\nSummary:")
    print(f"  Synthetic — latent has higher cos-sim: {synth_lat_wins_cos}/{n_synth} points")
    print(f"  Synthetic — latent has lower CV:       {synth_lat_wins_cv}/{n_synth} points")
    print(f"  Text      — latent has higher cos-sim: {text_lat_wins_cos}/{n_text} points")
    print(f"  Text      — latent has lower CV:       {text_lat_wins_cv}/{n_text} points")

    if text_lat_wins_cos > n_text // 2 and text_lat_wins_cv > n_text // 2:
        print("\n  VERDICT: Dream-MPC's gradient-smoothness claim SUPPORTED on text.")
        print("  Latent prediction landscape is smoother under compression.")
    elif synth_lat_wins_cos <= n_synth // 2 and text_lat_wins_cos > n_text // 2:
        print("\n  VERDICT: Gradient smoothness is DOMAIN-DEPENDENT (same split as H_A/H_D).")
        print("  Latent is smoother on text, rougher on synthetic — matches divergence direction.")
    else:
        print("\n  VERDICT: No clear smoothness advantage for latent prediction.")
        print("  Dream-MPC's gradient-smoothness claim not supported at this scale.")

    print(f"\nTotal elapsed: {time.time()-overall_t0:.1f}s — results: {out}")


if __name__ == "__main__":
    main()
