"""
Batch A — Seed sensitivity.

Question: are Phase 2's results robust across seeds, or seed-lucky?
This directly informs Phase 3 failure mode #2 (does the smoothness probe need
multi-seed averaging?) and the per-class variance metric (is the cliff inversion
robust or noise?).

Procedure:
- 8 model_seed values (model init varies)
- Same aug_seed (training augmentation curriculum identical)
- Same data seeds (train/test sets fixed, so we measure model-init effect, not data effect)
- For each seed: train, then evaluate at baseline + 4 fixed compression points
- Track per-class MSE inversion at the cliff

Compression points to verify:
  - baseline: bits=32, latent=1.0, state=1.0
  - mild     (8.0x): bits=4,  latent=1.0,    state=1.0
  - mid    (85.3x):  bits=4,  latent=48/64,  state=2/16   (Phase 2 "thesis holds" zone)
  - cliff (170.7x):  bits=2,  latent=48/64,  state=2/16   (Phase 2 "cliff reached" zone)
  - max  (32768x):   bits=1,  latent=1/64,   state=1/16   (theoretical maximum)
"""

import json
import os
import sys
import time
from datetime import datetime

# allow importing engine from parent dir
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine import Engine, PATTERN_NAMES

import os as _os
SEEDS = [int(s) for s in _os.environ.get("BATCH_A_SEEDS", "0,1,2,3,4,5,6,7").split(",")]
# v2: 64-token sequences. state_ratio of 8/64 = 0.125 mirrors v1's 2/16 = 0.125
COMPRESSION_POINTS = [
    ("baseline", 32, 1.0, 1.0),
    ("mild_8x", 4, 1.0, 1.0),
    ("mid_85x", 4, 48 / 64, 8 / 64),
    ("cliff_171x", 2, 48 / 64, 8 / 64),
    ("max_131k", 1, 1 / 64, 1 / 64),
]


def per_class_variance(per_class_mse):
    vals = [per_class_mse[c] for c in PATTERN_NAMES]
    mean = sum(vals) / len(vals)
    return sum((v - mean) ** 2 for v in vals) / len(vals)


def run_one_seed(seed: int):
    print(f"\n=== seed {seed} ===", flush=True)
    t0 = time.time()
    e = Engine()
    e.train(model_seed=seed, aug_seed=seed)
    train_time = time.time() - t0

    point_results = {}
    for name, bits, latent_ratio, state_ratio in COMPRESSION_POINTS:
        r = e.evaluate(bits=bits, latent_ratio=latent_ratio, state_ratio=state_ratio)
        point_results[name] = {
            "bits": bits,
            "latent_ratio": latent_ratio,
            "state_ratio": state_ratio,
            "compression_total": r["compression"]["total"],
            "accuracy": r["accuracy"],
            "retained": r["retained"],
            "pred_mse": r["prediction"]["mse"],
            "pred_mae": r["prediction"]["mae"],
            "pred_retained": r["prediction"]["retained"],
            "per_class_acc": r["per_class"],
            "per_class_mse": r["prediction"]["per_class_mse"],
            "per_class_mse_variance": per_class_variance(r["prediction"]["per_class_mse"]),
        }

    return {
        "seed": seed,
        "params": e.n_params,
        "train_time_s": round(train_time, 2),
        "baseline_acc": e.baseline,
        "baseline_pred_mse": e.baseline_pred_mse,
        "points": point_results,
    }


def summarize(runs):
    """Compute mean/std/min/max for key metrics across seeds."""
    n = len(runs)
    keys_per_point = [
        "accuracy", "retained", "pred_mse", "pred_mae", "pred_retained",
        "per_class_mse_variance",
    ]
    summary = {}
    for point in [p[0] for p in COMPRESSION_POINTS]:
        summary[point] = {}
        for k in keys_per_point:
            vals = [r["points"][point][k] for r in runs]
            mean = sum(vals) / n
            var = sum((v - mean) ** 2 for v in vals) / n
            summary[point][k] = {
                "mean": round(mean, 5),
                "std": round(var ** 0.5, 5),
                "min": round(min(vals), 5),
                "max": round(max(vals), 5),
                "values": [round(v, 5) for v in vals],
            }
        # per-class MSE summary at this compression point
        summary[point]["per_class_mse"] = {}
        for cls in PATTERN_NAMES:
            vals = [r["points"][point]["per_class_mse"][cls] for r in runs]
            mean = sum(vals) / n
            var = sum((v - mean) ** 2 for v in vals) / n
            summary[point]["per_class_mse"][cls] = {
                "mean": round(mean, 5),
                "std": round(var ** 0.5, 5),
                "min": round(min(vals), 5),
                "max": round(max(vals), 5),
            }
    return summary


def main():
    print(f"Batch A — seed sensitivity ({len(SEEDS)} seeds)", flush=True)
    started = datetime.now().isoformat()
    overall_t0 = time.time()

    runs = []
    for s in SEEDS:
        runs.append(run_one_seed(s))
        # periodic snapshot to disk so we don't lose progress
        snapshot = {
            "experiment": "batch_a_seed_sensitivity",
            "started": started,
            "seeds_completed": [r["seed"] for r in runs],
            "config": {"seeds": SEEDS, "compression_points": COMPRESSION_POINTS},
            "runs": runs,
        }
        out_name = os.environ.get("BATCH_A_OUT", "batch_a_results.json")
        out = os.path.join(os.path.dirname(os.path.abspath(__file__)), out_name)
        with open(out, "w") as f:
            json.dump(snapshot, f, indent=2)

    summary = summarize(runs)
    final = {
        "experiment": "batch_a_seed_sensitivity",
        "started": started,
        "completed": datetime.now().isoformat(),
        "elapsed_s": round(time.time() - overall_t0, 1),
        "config": {"seeds": SEEDS, "compression_points": COMPRESSION_POINTS},
        "runs": runs,
        "summary": summary,
    }
    out_name = os.environ.get("BATCH_A_OUT", "batch_a_results.json")
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), out_name)
    with open(out, "w") as f:
        json.dump(final, f, indent=2)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for point_name, _, _, _ in COMPRESSION_POINTS:  # noqa: E741
        s = summary[point_name]
        print(f"\n{point_name}:")
        print(f"  acc        mean={s['accuracy']['mean']:.4f}  std={s['accuracy']['std']:.4f}  "
              f"range=[{s['accuracy']['min']:.4f}, {s['accuracy']['max']:.4f}]")
        print(f"  pred_mse   mean={s['pred_mse']['mean']:.4f}  std={s['pred_mse']['std']:.4f}  "
              f"range=[{s['pred_mse']['min']:.4f}, {s['pred_mse']['max']:.4f}]")
        print(f"  pcm_var    mean={s['per_class_mse_variance']['mean']:.4f}  "
              f"std={s['per_class_mse_variance']['std']:.4f}")

    print(f"\nTotal elapsed: {final['elapsed_s']}s — results: {out}")


if __name__ == "__main__":
    main()
