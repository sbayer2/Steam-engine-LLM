"""
Batch Phase 3 — β (latent loss weight) sweep on Toy v2.

Question: is the negative divergence index (latent worse than raw) a property of
β=0.5 underweighting the latent loss, or genuine? Sweep β across {0.5, 1.0, 2.0,
5.0} with 3 seeds each. If higher β flips the divergence sign, the original
result was design-sensitive. If all β values yield negative divergence, latent
prediction is genuinely more compression-fragile in this toy.

12 runs × ~105s per train + eval = ~22-25 min total.
"""

import json
import os
import sys
import time
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine import Engine, PATTERN_NAMES

BETAS = [0.5, 1.0, 2.0, 5.0]
SEEDS = [0, 1, 2]
ALPHA = 1.0

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


def run_one(beta, seed):
    print(f"\n--- beta={beta} seed={seed} ---", flush=True)
    t0 = time.time()
    e = Engine()
    e.train(model_seed=seed, aug_seed=seed, alpha=ALPHA, beta=beta)
    train_time = time.time() - t0

    points = {}
    for name, bits, lr, sr in COMPRESSION_POINTS:
        r = e.evaluate(bits=bits, latent_ratio=lr, state_ratio=sr)
        p = r["prediction"]
        points[name] = {
            "compression_total": r["compression"]["total"],
            "accuracy": r["accuracy"],
            "raw_mse": p["mse"],
            "raw_retained": p["retained"],
            "latent_mse": p["latent_mse"],
            "latent_retained": p["latent_retained"],
            "divergence_index": p["divergence_index"],
            "per_class_raw_pcv": per_class_variance(p["per_class_mse"]),
            "per_class_latent_pcv": per_class_variance(p["per_class_latent_mse"]),
        }
    return {
        "beta": beta,
        "seed": seed,
        "train_time_s": round(train_time, 2),
        "baseline_acc": e.baseline,
        "baseline_raw_mse": e.baseline_pred_mse,
        "baseline_latent_mse": e.baseline_latent_mse,
        "points": points,
    }


def main():
    print(f"Phase 3 β sweep — betas={BETAS}, seeds={SEEDS}", flush=True)
    started = datetime.now().isoformat()
    overall_t0 = time.time()
    runs = []

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "batch_phase3_beta_results.json")

    for beta in BETAS:
        print(f"\n=== BETA = {beta} ===", flush=True)
        for seed in SEEDS:
            runs.append(run_one(beta, seed))
            snapshot = {
                "experiment": "phase3_beta_sweep",
                "started": started,
                "completed_runs": len(runs),
                "config": {"betas": BETAS, "seeds": SEEDS, "alpha": ALPHA,
                           "compression_points": COMPRESSION_POINTS},
                "runs": runs,
            }
            with open(out, "w") as f:
                json.dump(snapshot, f, indent=2)

    # Summary by beta
    print("\n" + "=" * 80)
    print("β-SWEEP SUMMARY")
    print("=" * 80)
    print(f"\n{'beta':>6} {'point':>14} {'raw_ret':>10} {'lat_ret':>10} {'div_avg':>10} {'div_std':>10} {'all_neg':>9}")
    for beta in BETAS:
        beta_runs = [r for r in runs if r["beta"] == beta]
        for pt_name, *_ in COMPRESSION_POINTS:
            divs = [r["points"][pt_name]["divergence_index"] for r in beta_runs]
            raws = [r["points"][pt_name]["raw_retained"] for r in beta_runs]
            lats = [r["points"][pt_name]["latent_retained"] for r in beta_runs]
            mean_d = sum(divs) / len(divs)
            std_d = (sum((d - mean_d) ** 2 for d in divs) / len(divs)) ** 0.5
            all_neg = all(d < 0 for d in divs)
            all_pos = all(d > 0 for d in divs)
            sign = "ALL NEG" if all_neg else ("ALL POS" if all_pos else "mixed")
            print(f"{beta:>6.1f} {pt_name:>14} {sum(raws)/len(raws):>10.4f} {sum(lats)/len(lats):>10.4f} {mean_d:>+10.4f} {std_d:>10.4f} {sign:>9}")

    print(f"\nTotal elapsed: {time.time()-overall_t0:.1f}s — results: {out}")


if __name__ == "__main__":
    main()
