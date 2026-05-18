"""
Batch Phase 4 — β (latent loss weight) sweep on TinyShakespeare.

Mechanism falsification test (ADC-020 item 1):

On synthetic Toy v2, higher β made H_D WORSE (divergence more negative) because
the encoder collapsed harder toward easy class-identity targets (baseline_lat_mse
decreased 0.13 → 0.08 as β increased 0.5 → 5.0).

Symmetric prediction on text: higher β should amplify H_A (divergence more
positive) because continuous char targets cannot collapse to class identity —
the encoder is forced to maintain a rich representation under stronger latent
pressure, yielding higher or stable baseline_lat_mse.

If higher β makes H_A WEAKER on text, the mechanism interpretation needs
revision.

12 runs × ~450s per train = ~90 min total on CPU (single-corpus is faster
than multi-corpus).
"""

import json
import os
import sys
import time
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from text_engine import TextEngine

BETAS = [0.5, 1.0, 2.0, 5.0]
SEEDS = [0, 1, 2]
EPOCHS = 200

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
    ("mid_85x", 4, 32 / 64, 32 / 256),
    ("cliff_171x", 2, 32 / 64, 32 / 256),
    ("max", 1, 1 / 64, 1 / 256),
]

NON_TRIVIAL_POINTS = [
    "bits_4", "bits_2", "bits_158",
    "latent_32", "latent_16", "latent_8", "latent_4",
    "mid_85x", "cliff_171x", "max",
]


def run_one(beta: float, seed: int) -> dict:
    print(f"\n--- beta={beta} seed={seed} ---", flush=True)
    t0 = time.time()
    e = TextEngine()
    e.train(epochs=EPOCHS, model_seed=seed, aug_seed=seed, beta=beta, multi_corpus=False)
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
        "beta": beta,
        "seed": seed,
        "train_time_s": round(train_time, 2),
        "baseline_perplexity": round(e.baseline_perplexity, 4),
        "baseline_latent_mse": round(e.baseline_latent_mse, 4),
        "n_params": e.n_params,
        "points": points,
    }


def main():
    print(f"Phase 4 text β sweep — betas={BETAS}, seeds={SEEDS}, epochs={EPOCHS}", flush=True)
    print(f"Mechanism prediction: higher β → H_A amplified (divergence more positive)")
    print(f"Falsification: higher β → H_A weakened → mechanism needs revision\n")
    started = datetime.now().isoformat()
    overall_t0 = time.time()
    runs = []

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "batch_phase4_text_beta_results.json")

    for beta in BETAS:
        print(f"\n{'='*60}\n  BETA = {beta}\n{'='*60}", flush=True)
        for seed in SEEDS:
            runs.append(run_one(beta, seed))
            snapshot = {
                "experiment": "phase4_text_beta_sweep",
                "started": started,
                "completed_runs": len(runs),
                "total_runs": len(BETAS) * len(SEEDS),
                "config": {"betas": BETAS, "seeds": SEEDS, "epochs": EPOCHS,
                           "compression_points": COMPRESSION_POINTS},
                "runs": runs,
            }
            with open(out, "w") as f:
                json.dump(snapshot, f, indent=2)

    elapsed = time.time() - overall_t0

    # Final output
    final = {
        "experiment": "phase4_text_beta_sweep",
        "started": started,
        "completed": datetime.now().isoformat(),
        "elapsed_s": round(elapsed, 1),
        "config": {"betas": BETAS, "seeds": SEEDS, "epochs": EPOCHS,
                   "compression_points": COMPRESSION_POINTS},
        "runs": runs,
    }
    with open(out, "w") as f:
        json.dump(final, f, indent=2)

    # === Summary ===
    print("\n" + "=" * 90)
    print("β-SWEEP ON TEXT — MECHANISM FALSIFICATION")
    print("=" * 90)

    # 1. Baseline latent MSE by β (the diagnostic)
    print(f"\n{'β':>6}  {'base_ppl avg':>12}  {'base_lat_mse avg':>16}  {'lat_mse trend':>14}")
    prev_mse = None
    for beta in BETAS:
        beta_runs = [r for r in runs if r["beta"] == beta]
        ppls = [r["baseline_perplexity"] for r in beta_runs]
        mses = [r["baseline_latent_mse"] for r in beta_runs]
        mean_mse = sum(mses) / len(mses)
        trend = ""
        if prev_mse is not None:
            trend = "↑" if mean_mse > prev_mse else "↓"
        prev_mse = mean_mse
        print(f"{beta:>6.1f}  {sum(ppls)/len(ppls):>12.4f}  {mean_mse:>16.4f}  {trend:>14}")

    print(f"\nOn synthetic: baseline_lat_mse DECREASED with β (collapse). Mechanism predicts")
    print(f"text should NOT decrease (no collapse possible on continuous targets).")

    # 2. Divergence by β × compression point
    print(f"\n{'β':>6} {'point':>14} {'ppl_ret':>9} {'lat_ret':>9} {'div_avg':>9} {'div_std':>9}  sign")
    print("-" * 80)
    for beta in BETAS:
        beta_runs = [r for r in runs if r["beta"] == beta]
        for pt_name in NON_TRIVIAL_POINTS:
            ppls = [r["points"][pt_name]["perplexity_retained"] for r in beta_runs]
            lats = [r["points"][pt_name]["latent_retained"] for r in beta_runs]
            divs = [r["points"][pt_name]["divergence_index"] for r in beta_runs]
            mean_d = sum(divs) / len(divs)
            std_d = (sum((d - mean_d) ** 2 for d in divs) / len(divs)) ** 0.5
            all_pos = all(d > 0 for d in divs)
            all_neg = all(d < 0 for d in divs)
            sign = "ALL POS" if all_pos else ("ALL NEG" if all_neg else "MIXED")
            print(f"{beta:>6.1f} {pt_name:>14} {sum(ppls)/len(ppls):>9.4f} {sum(lats)/len(lats):>9.4f} {mean_d:>+9.4f} {std_d:>9.4f}  {sign}")
        print()

    # 3. Aggregate: does divergence increase monotonically with β?
    print("MECHANISM TEST — mean divergence across non-trivial points by β:")
    print(f"{'β':>6}  {'mean_div':>10}  {'positive cells':>15}  {'direction vs β=0.5':>20}")
    ref_div = None
    for beta in BETAS:
        beta_runs = [r for r in runs if r["beta"] == beta]
        all_divs = [r["points"][pt]["divergence_index"]
                    for r in beta_runs for pt in NON_TRIVIAL_POINTS]
        mean_div = sum(all_divs) / len(all_divs)
        pos_count = sum(1 for d in all_divs if d > 0)
        if ref_div is None:
            ref_div = mean_div
            direction = "(reference)"
        else:
            direction = "AMPLIFIED ✓" if mean_div > ref_div else "WEAKENED ✗"
        print(f"{beta:>6.1f}  {mean_div:>+10.4f}  {pos_count:>8}/{len(all_divs):<6}  {direction:>20}")

    # 4. Verdict
    print("\n" + "=" * 90)
    beta_divs = {}
    for beta in BETAS:
        beta_runs = [r for r in runs if r["beta"] == beta]
        all_divs = [r["points"][pt]["divergence_index"]
                    for r in beta_runs for pt in NON_TRIVIAL_POINTS]
        beta_divs[beta] = sum(all_divs) / len(all_divs)

    monotonic_up = all(beta_divs[BETAS[i]] <= beta_divs[BETAS[i + 1]] for i in range(len(BETAS) - 1))
    monotonic_down = all(beta_divs[BETAS[i]] >= beta_divs[BETAS[i + 1]] for i in range(len(BETAS) - 1))

    if monotonic_up:
        print("VERDICT: β monotonically AMPLIFIES H_A on text. Mechanism CONFIRMED.")
        print("  Higher β → richer latent representation → more compression-robust → H_A stronger.")
    elif monotonic_down:
        print("VERDICT: β monotonically WEAKENS H_A on text. Mechanism FALSIFIED.")
        print("  Same collapse dynamic as synthetic occurs on text. The continuous-target")
        print("  explanation for the H_A/H_D split needs revision.")
    else:
        print("VERDICT: NON-MONOTONIC. Mechanism partially supported.")
        peak_beta = max(beta_divs, key=beta_divs.get)
        print(f"  Peak H_A at β={peak_beta}. Collapse pressure exists on text but saturates.")
        print(f"  Interpretation: continuous targets resist collapse up to a point, then")
        print(f"  encoder finds text-specific shortcuts (smooth-averaging, bigram identity).")

    print(f"\nTotal elapsed: {elapsed:.1f}s ({elapsed/60:.1f} min) — results: {out}")


if __name__ == "__main__":
    main()
