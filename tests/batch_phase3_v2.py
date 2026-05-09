"""
Batch Phase 3 — multi-seed Phase 3 test on Toy v2.

Question: is the negative divergence index (latent worse than raw under
compression) consistent across seeds, or seed noise?

Procedure:
- N seeds (default 3), train Phase 3 model with alpha=1.0, beta=0.5
- Eval at standard 5 compression points
- Track raw_retained, latent_retained, and divergence_index per (seed, point)
- Report sign consistency: are all 3 seeds negative at each compression point?
"""

import json
import os
import sys
import time
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine import Engine, PATTERN_NAMES

SEEDS = [int(s) for s in os.environ.get("PHASE3_SEEDS", "0,1,2").split(",")]
ALPHA = float(os.environ.get("PHASE3_ALPHA", "1.0"))
BETA = float(os.environ.get("PHASE3_BETA", "0.5"))

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


def run_one_seed(seed):
    print(f"\n=== seed {seed} ===", flush=True)
    t0 = time.time()
    e = Engine()
    e.train(model_seed=seed, aug_seed=seed, alpha=ALPHA, beta=BETA)
    train_time = time.time() - t0

    points = {}
    for name, bits, lr, sr in COMPRESSION_POINTS:
        r = e.evaluate(bits=bits, latent_ratio=lr, state_ratio=sr)
        p = r["prediction"]
        points[name] = {
            "compression_total": r["compression"]["total"],
            "accuracy": r["accuracy"],
            "retained": r["retained"],
            "raw_mse": p["mse"],
            "raw_retained": p["retained"],
            "latent_mse": p["latent_mse"],
            "latent_retained": p["latent_retained"],
            "divergence_index": p["divergence_index"],
            "per_class_raw_mse": p["per_class_mse"],
            "per_class_latent_mse": p["per_class_latent_mse"],
            "per_class_raw_pcv": per_class_variance(p["per_class_mse"]),
            "per_class_latent_pcv": per_class_variance(p["per_class_latent_mse"]),
        }
    return {
        "seed": seed,
        "train_time_s": round(train_time, 2),
        "baseline_acc": e.baseline,
        "baseline_raw_mse": e.baseline_pred_mse,
        "baseline_latent_mse": e.baseline_latent_mse,
        "n_params": e.n_params,
        "points": points,
    }


def main():
    print(f"Phase 3 v2 — {len(SEEDS)} seeds, alpha={ALPHA}, beta={BETA}", flush=True)
    started = datetime.now().isoformat()
    t0 = time.time()
    runs = []

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "batch_phase3_v2_results.json")
    for s in SEEDS:
        runs.append(run_one_seed(s))
        snapshot = {
            "experiment": "phase3_v2_multi_seed",
            "started": started,
            "seeds_completed": [r["seed"] for r in runs],
            "config": {"seeds": SEEDS, "alpha": ALPHA, "beta": BETA, "compression_points": COMPRESSION_POINTS},
            "runs": runs,
        }
        with open(out, "w") as f:
            json.dump(snapshot, f, indent=2)

    final = {
        "experiment": "phase3_v2_multi_seed",
        "started": started,
        "completed": datetime.now().isoformat(),
        "elapsed_s": round(time.time() - t0, 1),
        "config": {"seeds": SEEDS, "alpha": ALPHA, "beta": BETA, "compression_points": COMPRESSION_POINTS},
        "runs": runs,
    }
    with open(out, "w") as f:
        json.dump(final, f, indent=2)

    print("\n" + "=" * 72)
    print(f"PHASE 3 v2 SUMMARY ({len(runs)} seeds, alpha={ALPHA}, beta={BETA})")
    print("=" * 72)
    for pt_name, *_ in COMPRESSION_POINTS:
        divs = [r["points"][pt_name]["divergence_index"] for r in runs]
        raw_rets = [r["points"][pt_name]["raw_retained"] for r in runs]
        lat_rets = [r["points"][pt_name]["latent_retained"] for r in runs]
        mean_div = sum(divs) / len(divs)
        std_div = (sum((d - mean_div) ** 2 for d in divs) / len(divs)) ** 0.5
        all_neg = all(d < 0 for d in divs)
        all_pos = all(d > 0 for d in divs)
        if all_neg:
            sign_str = "** ALL NEGATIVE (H_D: latent worse than raw) **"
        elif all_pos:
            sign_str = "** ALL POSITIVE (H_A: JEPA wins) **"
        else:
            sign_str = "mixed signs (H_B / inconclusive)"
        print(f"\n{pt_name}:")
        print(f"  raw_retained    {[round(v,3) for v in raw_rets]}  mean={sum(raw_rets)/len(raw_rets):.3f}")
        print(f"  latent_retained {[round(v,3) for v in lat_rets]}  mean={sum(lat_rets)/len(lat_rets):.3f}")
        print(f"  divergence      {[round(d,3) for d in divs]}  mean={mean_div:+.3f}±{std_div:.3f}")
        print(f"                  {sign_str}")
    print(f"\nTotal elapsed: {time.time()-t0:.1f}s — results: {out}")


if __name__ == "__main__":
    main()
