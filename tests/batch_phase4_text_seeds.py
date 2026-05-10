"""
Batch Phase 4 — multi-seed dual-loss test on TinyShakespeare.

Question: is the positive divergence index (latent prediction more compression-
robust than raw next-token, H_A direction) consistent across seeds, or seed luck?

Procedure:
- N seeds (default 3), train Phase 4 Step 2 model with alpha=1.0, beta=0.5
- Evaluate at standard compression points
- Track ppl_retained, latent_retained, divergence_index per (seed, point)
- Report sign consistency
"""

import json
import os
import sys
import time
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from text_engine import TextEngine

SEEDS = [int(s) for s in os.environ.get("PHASE4_SEEDS", "0,1,2").split(",")]
BETA = float(os.environ.get("PHASE4_BETA", "0.5"))
EPOCHS = int(os.environ.get("PHASE4_EPOCHS", "200"))

COMPRESSION_POINTS = [
    ("baseline", 32, 1.0, 1.0),
    ("bits_4", 4, 1.0, 1.0),
    ("bits_2", 2, 1.0, 1.0),
    ("bits_158", 1.58, 1.0, 1.0),
    ("latent_32", 32, 32 / 64, 1.0),
    ("latent_16", 32, 16 / 64, 1.0),
    ("latent_8", 32, 8 / 64, 1.0),
    ("latent_4", 32, 4 / 64, 1.0),
    ("window_32", 32, 1.0, 32 / 256),
    ("window_8", 32, 1.0, 8 / 256),
    ("mid_85x", 4, 32 / 64, 32 / 256),
    ("cliff_171x", 2, 32 / 64, 32 / 256),
    ("max", 1, 1 / 64, 1 / 256),
]


def run_one_seed(seed):
    print(f"\n=== seed {seed} ===", flush=True)
    t0 = time.time()
    e = TextEngine()
    e.train(epochs=EPOCHS, model_seed=seed, aug_seed=seed, beta=BETA)
    train_time = time.time() - t0

    points = {}
    for name, bits, lr, sr in COMPRESSION_POINTS:
        r = e.evaluate(bits=bits, latent_ratio=lr, state_ratio=sr)
        points[name] = {
            "compression_total": r["compression"]["total"],
            "perplexity": r["perplexity"],
            "perplexity_retained": r["perplexity_retained"],
            "latent_mse": r["latent_mse"],
            "latent_retained": r["latent_retained"],
            "divergence_index": r["divergence_index"],
        }
    return {
        "seed": seed,
        "train_time_s": round(train_time, 2),
        "baseline_perplexity": e.baseline_perplexity,
        "baseline_latent_mse": e.baseline_latent_mse,
        "n_params": e.n_params,
        "points": points,
    }


def main():
    print(f"Phase 4 text — {len(SEEDS)} seeds, beta={BETA}, epochs={EPOCHS}", flush=True)
    started = datetime.now().isoformat()
    overall_t0 = time.time()
    runs = []

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "batch_phase4_text_results.json")
    for s in SEEDS:
        runs.append(run_one_seed(s))
        snapshot = {
            "experiment": "phase4_text_multi_seed",
            "started": started,
            "seeds_completed": [r["seed"] for r in runs],
            "config": {"seeds": SEEDS, "beta": BETA, "epochs": EPOCHS, "compression_points": COMPRESSION_POINTS},
            "runs": runs,
        }
        with open(out, "w") as f:
            json.dump(snapshot, f, indent=2)

    final = {
        "experiment": "phase4_text_multi_seed",
        "started": started,
        "completed": datetime.now().isoformat(),
        "elapsed_s": round(time.time() - overall_t0, 1),
        "config": {"seeds": SEEDS, "beta": BETA, "epochs": EPOCHS, "compression_points": COMPRESSION_POINTS},
        "runs": runs,
    }
    with open(out, "w") as f:
        json.dump(final, f, indent=2)

    print("\n" + "=" * 80)
    print(f"PHASE 4 TEXT MULTI-SEED SUMMARY ({len(runs)} seeds, beta={BETA}, epochs={EPOCHS})")
    print("=" * 80)
    print(f"\n{'point':>14} {'comp':>10} {'ppl_ret avg':>12} {'lat_ret avg':>12} {'div avg':>10} {'div std':>10}  sign")
    for pt_name, *_ in COMPRESSION_POINTS:
        ppls = [r["points"][pt_name]["perplexity_retained"] for r in runs]
        lats = [r["points"][pt_name]["latent_retained"] for r in runs]
        divs = [r["points"][pt_name]["divergence_index"] for r in runs]
        comp = runs[0]["points"][pt_name]["compression_total"]
        mean_d = sum(divs) / len(divs)
        std_d = (sum((d - mean_d) ** 2 for d in divs) / len(divs)) ** 0.5
        all_pos = all(d > 0 for d in divs)
        all_neg = all(d < 0 for d in divs)
        if all_pos:
            sign = "ALL POS (H_A)"
        elif all_neg:
            sign = "ALL NEG (H_D)"
        else:
            sign = "mixed"
        print(f"{pt_name:>14} {comp:>9.1f}x {sum(ppls)/len(ppls):>12.4f} {sum(lats)/len(lats):>12.4f} {mean_d:>+10.4f} {std_d:>10.4f}  {sign}")

    # Aggregate cell-level sign consistency
    total_neg = sum(1 for r in runs for pt in ["bits_4","bits_2","bits_158","latent_32","latent_16","latent_8","latent_4","mid_85x","cliff_171x","max"]
                    if r["points"][pt]["divergence_index"] < 0)
    total_cells = len(runs) * 10  # 10 non-trivial points (excludes baseline + window-axis where compression is mild)
    print(f"\nNon-trivial cells (excluding baseline & window axis): {total_cells - total_neg}/{total_cells} positive ({100*(total_cells-total_neg)/total_cells:.1f}%)")
    print(f"\nTotal elapsed: {time.time()-overall_t0:.1f}s — results: {out}")


if __name__ == "__main__":
    main()
