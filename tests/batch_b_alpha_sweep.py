"""
Batch B — α (prediction loss weight) sweep.

Question: does α=1.0 actually balance the two heads, or does one dominate?
This directly informs the β choice for Phase 3's third loss. If raw prediction
saturates at low α and only the third loss (latent prediction) needs tuning,
β=0.5 may be too high or too low.

Procedure:
- Fix model_seed=0 (use whatever seed Batch A says is "median" — TBD)
- Sweep α ∈ {0.0, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0}
- α=0.0: classification-only baseline (recovers Phase 1 behavior)
- For each α: train, then evaluate at the same 5 compression points as Batch A
- Track:
  - baseline accuracy (does prediction loss hurt classification?)
  - baseline pred MSE (does scaling α help prediction or saturate?)
  - cliff position (does α shift where the cliff is?)
  - per-class variance (does α affect class-discrimination retention?)
"""

import json
import os
import sys
import time
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine import Engine, PATTERN_NAMES

ALPHAS = [0.0, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0]
SEED = 0  # TBD: pick from Batch A median

COMPRESSION_POINTS = [
    ("baseline", 32, 1.0, 1.0),
    ("mild_8x", 4, 1.0, 1.0),
    ("mid_85x", 4, 48 / 64, 2 / 16),
    ("cliff_171x", 2, 48 / 64, 2 / 16),
    ("max_32k", 1, 1 / 64, 1 / 16),
]


def per_class_variance(per_class_mse):
    vals = [per_class_mse[c] for c in PATTERN_NAMES]
    mean = sum(vals) / len(vals)
    return sum((v - mean) ** 2 for v in vals) / len(vals)


def run_one_alpha(alpha: float):
    print(f"\n=== alpha {alpha} ===", flush=True)
    t0 = time.time()
    e = Engine()
    e.train(alpha=alpha, model_seed=SEED, aug_seed=SEED)
    train_time = time.time() - t0

    point_results = {}
    for name, bits, latent_ratio, state_ratio in COMPRESSION_POINTS:
        r = e.evaluate(bits=bits, latent_ratio=latent_ratio, state_ratio=state_ratio)
        point_results[name] = {
            "accuracy": r["accuracy"],
            "retained": r["retained"],
            "pred_mse": r["prediction"]["mse"],
            "pred_retained": r["prediction"]["retained"],
            "per_class_mse": r["prediction"]["per_class_mse"],
            "per_class_mse_variance": per_class_variance(r["prediction"]["per_class_mse"]),
        }

    return {
        "alpha": alpha,
        "train_time_s": round(train_time, 2),
        "baseline_acc": e.baseline,
        "baseline_pred_mse": e.baseline_pred_mse,
        "points": point_results,
    }


def main():
    print(f"Batch B — alpha sweep (seed={SEED}, alphas={ALPHAS})", flush=True)
    started = datetime.now().isoformat()
    overall_t0 = time.time()

    runs = []
    for a in ALPHAS:
        runs.append(run_one_alpha(a))
        # snapshot
        snapshot = {
            "experiment": "batch_b_alpha_sweep",
            "started": started,
            "alphas_completed": [r["alpha"] for r in runs],
            "config": {"seed": SEED, "alphas": ALPHAS, "compression_points": COMPRESSION_POINTS},
            "runs": runs,
        }
        out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "batch_b_results.json")
        with open(out, "w") as f:
            json.dump(snapshot, f, indent=2)

    final = {
        "experiment": "batch_b_alpha_sweep",
        "started": started,
        "completed": datetime.now().isoformat(),
        "elapsed_s": round(time.time() - overall_t0, 1),
        "config": {"seed": SEED, "alphas": ALPHAS, "compression_points": COMPRESSION_POINTS},
        "runs": runs,
    }
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "batch_b_results.json")
    with open(out, "w") as f:
        json.dump(final, f, indent=2)

    print("\n" + "=" * 60)
    print("ALPHA SWEEP SUMMARY")
    print("=" * 60)
    print(f"\n{'alpha':>7} {'base_acc':>10} {'base_mse':>10}  {'cliff_acc':>10} {'cliff_mse':>10}")
    for r in runs:
        b = r["points"]["baseline"]
        c = r["points"]["cliff_171x"]
        print(f"{r['alpha']:>7.2f} {b['accuracy']:>10.4f} {b['pred_mse']:>10.4f}  "
              f"{c['accuracy']:>10.4f} {c['pred_mse']:>10.4f}")
    print(f"\nTotal elapsed: {final['elapsed_s']}s — results: {out}")


if __name__ == "__main__":
    main()
