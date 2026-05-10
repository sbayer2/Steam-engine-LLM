"""
Batch Phase 4 Step 1.5 — multi-seed multi-corpus test.

Question: is the per-corpus divergence pattern from the single-seed run robust
across seeds, or chance? Specifically:
- Does Austen consistently lean H_D on bit-axis compression?
- Does Code consistently lean H_A on bit-axis compression?
- Does the latent-axis H_A finding hold across seeds for all three corpora?

Procedure:
- N seeds (default 3), train Phase 4 Step 2 model in MULTI-CORPUS mode
- For each seed, evaluate at standard compression points
- For each (seed, point), record per-corpus {ppl_retained, lat_retained, divergence}
- Aggregate: mean ± std per (point, corpus) across seeds
- Sign consistency: count seeds where each (point, corpus) cell is positive vs negative
"""

import json
import os
import sys
import time
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from text_engine import TextEngine

SEEDS = [int(s) for s in os.environ.get("PHASE4MC_SEEDS", "0,1,2").split(",")]
BETA = float(os.environ.get("PHASE4MC_BETA", "0.5"))
EPOCHS = int(os.environ.get("PHASE4MC_EPOCHS", "200"))

COMPRESSION_POINTS = [
    ("bits_4", 4, 1.0, 1.0),
    ("bits_2", 2, 1.0, 1.0),
    ("bits_158", 1.58, 1.0, 1.0),
    ("latent_16", 32, 16 / 64, 1.0),
    ("latent_8", 32, 8 / 64, 1.0),
    ("latent_4", 32, 4 / 64, 1.0),
    ("mid_85x", 4, 32 / 64, 32 / 256),
    ("cliff_171x", 2, 32 / 64, 32 / 256),
]


def run_one_seed(seed):
    print(f"\n=== seed {seed} ===", flush=True)
    t0 = time.time()
    e = TextEngine()
    e.train(epochs=EPOCHS, model_seed=seed, aug_seed=seed, beta=BETA, multi_corpus=True)
    train_time = time.time() - t0

    points = {}
    for name, bits, lr, sr in COMPRESSION_POINTS:
        r = e.evaluate(bits=bits, latent_ratio=lr, state_ratio=sr)
        points[name] = {
            "compression_total": r["compression"]["total"],
            "overall": {
                "perplexity": r["perplexity"],
                "perplexity_retained": r["perplexity_retained"],
                "latent_mse": r["latent_mse"],
                "latent_retained": r["latent_retained"],
                "divergence_index": r["divergence_index"],
            },
            "per_corpus": r.get("per_corpus", {}),
        }
    return {
        "seed": seed,
        "train_time_s": round(train_time, 2),
        "baseline_perplexity": e.baseline_perplexity,
        "baseline_latent_mse": e.baseline_latent_mse,
        "baseline_per_corpus": {
            n: {"perplexity": m["perplexity"], "latent_mse": m["latent_mse"]}
            for n, m in e.baseline_per_corpus.items()
        },
        "n_params": e.n_params,
        "points": points,
    }


def main():
    print(f"Phase 4 multi-corpus — {len(SEEDS)} seeds, beta={BETA}, epochs={EPOCHS}", flush=True)
    started = datetime.now().isoformat()
    overall_t0 = time.time()
    runs = []

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "batch_phase4_multi_corpus_results.json")
    for s in SEEDS:
        runs.append(run_one_seed(s))
        snapshot = {
            "experiment": "phase4_multi_corpus",
            "started": started,
            "seeds_completed": [r["seed"] for r in runs],
            "config": {"seeds": SEEDS, "beta": BETA, "epochs": EPOCHS, "compression_points": COMPRESSION_POINTS},
            "runs": runs,
        }
        with open(out, "w") as f:
            json.dump(snapshot, f, indent=2)

    final = {
        "experiment": "phase4_multi_corpus",
        "started": started,
        "completed": datetime.now().isoformat(),
        "elapsed_s": round(time.time() - overall_t0, 1),
        "config": {"seeds": SEEDS, "beta": BETA, "epochs": EPOCHS, "compression_points": COMPRESSION_POINTS},
        "runs": runs,
    }
    with open(out, "w") as f:
        json.dump(final, f, indent=2)

    # Aggregate per-corpus stats across seeds
    print("\n" + "=" * 90)
    print(f"PHASE 4 MULTI-CORPUS SUMMARY ({len(runs)} seeds, beta={BETA}, epochs={EPOCHS})")
    print("=" * 90)

    corpus_names = list(runs[0]["points"][COMPRESSION_POINTS[0][0]]["per_corpus"].keys())
    print(f"\nPer-corpus baseline lat_mse across seeds:")
    for cname in corpus_names:
        vals = [r["baseline_per_corpus"][cname]["latent_mse"] for r in runs]
        m = sum(vals) / len(vals)
        s = (sum((v - m) ** 2 for v in vals) / len(vals)) ** 0.5
        print(f"  {cname:>12}  mean={m:.4f}  std={s:.4f}  values={[round(v,4) for v in vals]}")

    print(f"\n{'point':>14} {'corpus':>12} {'div_avg':>10} {'div_std':>10} {'sign':>15}  per-seed")
    for pt_name, *_ in COMPRESSION_POINTS:
        for cname in corpus_names:
            divs = [r["points"][pt_name]["per_corpus"][cname]["divergence_index"] for r in runs]
            mean_d = sum(divs) / len(divs)
            std_d = (sum((d - mean_d) ** 2 for d in divs) / len(divs)) ** 0.5
            all_pos = all(d > 0 for d in divs)
            all_neg = all(d < 0 for d in divs)
            if all_pos:
                sign = "ALL POS (H_A)"
            elif all_neg:
                sign = "ALL NEG (H_D)"
            else:
                sign = "MIXED"
            print(f"{pt_name:>14} {cname:>12} {mean_d:>+10.4f} {std_d:>10.4f} {sign:>15}  {[round(d,3) for d in divs]}")

    print(f"\nTotal elapsed: {time.time()-overall_t0:.1f}s — results: {out}")


if __name__ == "__main__":
    main()
