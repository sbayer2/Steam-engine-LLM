"""
Batch C — Fine latent-axis sweep with per-class variance tracking.

Question: where exactly does class-structure collapse along the latent axis?
Phase 2's coarse sweep at {64, 48, 32, 16, 8, 4, 2, 1} dims shows a stair-step,
but the inflection between 16→8→4 is the most interesting region. Sweep finely
to characterize the cliff shape and locate where per-class variance falls.

Procedure:
- Train one model (use median seed from Batch A)
- Fine sweep: latent dims at {64, 56, 48, 40, 32, 28, 24, 20, 16, 14, 12, 10, 8, 7, 6, 5, 4, 3, 2, 1}
- For each, evaluate (no other compression) and record:
  - pred_mse, retained
  - per_class_mse for all 5 classes
  - per_class_variance (the proposed Phase 3 metric)
- Plot pred_mse and per_class_variance vs latent_dim — does variance fall before
  or after MSE rises?

This characterizes the qualitative-vs-quantitative cliff distinction at fine resolution.
"""

import json
import os
import sys
import time
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine import Engine, PATTERN_NAMES, LATENT_DIM

LATENT_DIMS_FINE = [64, 56, 48, 40, 32, 28, 24, 20, 16, 14, 12, 10, 8, 7, 6, 5, 4, 3, 2, 1]
SEED = 0  # TBD from Batch A median


def per_class_variance(per_class_mse):
    vals = [per_class_mse[c] for c in PATTERN_NAMES]
    mean = sum(vals) / len(vals)
    return sum((v - mean) ** 2 for v in vals) / len(vals)


def main():
    print(f"Batch C — fine latent sweep (seed={SEED})", flush=True)
    started = datetime.now().isoformat()
    overall_t0 = time.time()

    print("Training base model...", flush=True)
    e = Engine()
    e.train(model_seed=SEED, aug_seed=SEED)
    print(f"Baseline acc {e.baseline:.4f} pred_mse {e.baseline_pred_mse:.4f}\n", flush=True)

    print(f"{'dims':>6} {'pred_mse':>10} {'retained':>10} {'pcm_var':>10}  per_class_mse")
    results = []
    for d in LATENT_DIMS_FINE:
        r = e.evaluate(latent_ratio=d / LATENT_DIM)
        pc = r["prediction"]["per_class_mse"]
        pcv = per_class_variance(pc)
        entry = {
            "dims": d,
            "latent_ratio": d / LATENT_DIM,
            "pred_mse": r["prediction"]["mse"],
            "pred_retained": r["prediction"]["retained"],
            "accuracy": r["accuracy"],
            "retained": r["retained"],
            "per_class_mse": pc,
            "per_class_variance": pcv,
        }
        results.append(entry)
        pcm_str = " ".join(f"{c[:4]}={pc[c]:.3f}" for c in PATTERN_NAMES)
        print(f"{d:>6} {r['prediction']['mse']:>10.4f} {r['prediction']['retained']:>10.4f} "
              f"{pcv:>10.5f}  {pcm_str}", flush=True)

    final = {
        "experiment": "batch_c_latent_fine_sweep",
        "started": started,
        "completed": datetime.now().isoformat(),
        "elapsed_s": round(time.time() - overall_t0, 1),
        "config": {"seed": SEED, "latent_dims": LATENT_DIMS_FINE},
        "baseline_acc": e.baseline,
        "baseline_pred_mse": e.baseline_pred_mse,
        "results": results,
    }
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "batch_c_results.json")
    with open(out, "w") as f:
        json.dump(final, f, indent=2)

    print(f"\nTotal elapsed: {final['elapsed_s']}s — results: {out}")


if __name__ == "__main__":
    main()
