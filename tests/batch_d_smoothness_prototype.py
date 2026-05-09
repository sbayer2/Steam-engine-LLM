"""
Batch D — Smoothness probe prototype on Phase 2 raw predictions.

Question: does the perturbation-variance metric produce interpretable signal
on the Phase 2 model? If yes, Phase 3's latent-version of the same probe will
also work. If no, Phase 3 needs a different gradient-smoothness measurement.

Procedure:
- Train one model (seed=0)
- For each compression point, take a batch of test sequences
- For each sequence, add ε*N(0,1) noise N times to the first 12 tokens
- Forward each perturbed copy through the compressed model
- Measure variance of pred outputs across the N perturbations
- Compare baseline-uncompressed variance vs cliff-compressed variance

Hypothesis (to validate the probe):
- Smoothness variance should be small at baseline (model is locally consistent)
- Smoothness variance should grow as compression increases (jagged landscape)
- If the variance is dominated by training noise rather than landscape topology,
  baseline variance will already be large — that would be a metric problem.

Per-class smoothness:
- We care about whether smoothness degrades the same way for all classes,
  or whether some pattern types lose smoothness faster
"""

import json
import os
import sys
import time
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from engine import Engine, PATTERN_NAMES, PRED_CTX, PRED_LEN

EPSILONS = [0.01, 0.05, 0.1]
N_PERTURB = 32
N_SAMPLES_PER_CLASS = 20  # 100 total samples
SEED = 0

COMPRESSION_POINTS = [
    ("baseline", 32, 1.0, 1.0),
    ("mid_85x", 4, 48 / 64, 2 / 16),
    ("cliff_171x", 2, 48 / 64, 2 / 16),
    ("max_32k", 1, 1 / 64, 1 / 16),
]


def smoothness_at(engine, bits, latent_ratio, state_ratio, eps, n_perturb, sample_x, sample_y):
    """For each sample, measure variance of pred output under N perturbations."""
    import copy
    from engine import LATENT_DIM, SEQ_LEN, quantize

    m = copy.deepcopy(engine.model)
    if bits < 32:
        quantize(m, bits)
    active = max(1, int(LATENT_DIM * latent_ratio))
    lm = torch.zeros(LATENT_DIM)
    lm[:active] = 1.0
    window = max(1, int(SEQ_LEN * state_ratio)) if state_ratio < 1.0 else SEQ_LEN
    pred_window = min(window, PRED_CTX)
    m.eval()

    rng = np.random.RandomState(123)
    per_sample_variances = []  # one variance per sample, averaged across PRED_LEN dims
    per_class_variances = {c: [] for c in PATTERN_NAMES}

    with torch.no_grad():
        for i in range(len(sample_x)):
            ctx = sample_x[i, :PRED_CTX].unsqueeze(0)  # (1, 12)
            cls = PATTERN_NAMES[int(sample_y[i].item())]

            preds = []
            for _ in range(n_perturb):
                noise = torch.from_numpy(
                    rng.normal(0, eps, (1, PRED_CTX)).astype(np.float32)
                )
                ctx_pert = ctx + noise
                _, p = m(ctx_pert, window=pred_window, latent_mask=lm)
                preds.append(p[0].cpu().numpy())  # (4,)

            preds = np.stack(preds)  # (N, 4)
            # variance for each output dim, then mean across dims
            var_per_dim = preds.var(axis=0)  # (4,)
            sample_var = float(var_per_dim.mean())
            per_sample_variances.append(sample_var)
            per_class_variances[cls].append(sample_var)

    overall = sum(per_sample_variances) / len(per_sample_variances)
    per_class_mean = {c: (sum(v) / len(v) if v else 0.0) for c, v in per_class_variances.items()}
    return {
        "mean_variance": overall,
        "max_variance": max(per_sample_variances),
        "per_class_variance": per_class_mean,
    }


def main():
    print(f"Batch D — smoothness probe prototype (seed={SEED})", flush=True)
    started = datetime.now().isoformat()
    overall_t0 = time.time()

    print("Training base model...", flush=True)
    e = Engine()
    e.train(model_seed=SEED, aug_seed=SEED)

    # Pick stratified sample: N_SAMPLES_PER_CLASS per class
    print(f"Selecting {N_SAMPLES_PER_CLASS} samples per class...", flush=True)
    indices = []
    for c_idx in range(len(PATTERN_NAMES)):
        cls_indices = (e.test_y == c_idx).nonzero(as_tuple=True)[0][:N_SAMPLES_PER_CLASS]
        indices.extend(cls_indices.tolist())
    sample_x = e.test_x[indices]
    sample_y = e.test_y[indices]
    print(f"Sample size: {len(sample_x)}", flush=True)

    results = []
    for eps in EPSILONS:
        for name, bits, latent_ratio, state_ratio in COMPRESSION_POINTS:
            t0 = time.time()
            res = smoothness_at(
                e, bits, latent_ratio, state_ratio, eps, N_PERTURB,
                sample_x, sample_y,
            )
            elapsed = time.time() - t0
            entry = {
                "epsilon": eps,
                "compression": name,
                "bits": bits,
                "latent_ratio": latent_ratio,
                "state_ratio": state_ratio,
                "elapsed_s": round(elapsed, 2),
                **res,
            }
            results.append(entry)
            print(f"  eps={eps} {name}: mean_var={res['mean_variance']:.5f} "
                  f"max={res['max_variance']:.5f} (took {elapsed:.1f}s)",
                  flush=True)

    final = {
        "experiment": "batch_d_smoothness_prototype",
        "started": started,
        "completed": datetime.now().isoformat(),
        "elapsed_s": round(time.time() - overall_t0, 1),
        "config": {
            "seed": SEED,
            "epsilons": EPSILONS,
            "n_perturb": N_PERTURB,
            "n_samples_per_class": N_SAMPLES_PER_CLASS,
            "compression_points": COMPRESSION_POINTS,
        },
        "results": results,
    }
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "batch_d_results.json")
    with open(out, "w") as f:
        json.dump(final, f, indent=2)

    print("\n" + "=" * 60)
    print("SMOOTHNESS SUMMARY")
    print("=" * 60)
    print(f"\n{'eps':>6} {'compression':>12} {'mean_var':>12} {'max_var':>12}")
    for r in results:
        print(f"{r['epsilon']:>6.2f} {r['compression']:>12} "
              f"{r['mean_variance']:>12.6f} {r['max_variance']:>12.6f}")

    # Compute baseline-vs-cliff variance ratio per epsilon
    print(f"\n{'eps':>6} {'baseline_var':>14} {'cliff_var':>12} {'ratio':>10}")
    for eps in EPSILONS:
        b = next(r for r in results if r["epsilon"] == eps and r["compression"] == "baseline")
        c = next(r for r in results if r["epsilon"] == eps and r["compression"] == "cliff_171x")
        ratio = c["mean_variance"] / max(b["mean_variance"], 1e-9)
        print(f"{eps:>6.2f} {b['mean_variance']:>14.6f} {c['mean_variance']:>12.6f} {ratio:>10.2f}x")

    print(f"\nTotal elapsed: {final['elapsed_s']}s — results: {out}")


if __name__ == "__main__":
    main()
