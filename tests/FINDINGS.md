# Steam Engine Test Findings (Pre-Phase-3 + Phase 4 Step 1)

## Phase 4 Step 1 — Text model first results (2026-05-09)

Built `text_engine.py`, a minimal char-level GPT (2 layers, 64 embd, 4 heads, 256 seq_len, 65 vocab) trained on TinyShakespeare with the same compression-aware curriculum as synthetic Toy v2. 200 epochs in 284s on CPU. 120K params. Final val perplexity 9.16.

**Three plateau-cliff replications across domains (synthetic → text):**

### Bit-width axis — ternary rescue replicates

| bits | loss | perplexity | retained |
|---|---|---|---|
| 32 / 16 / 8 | 2.22 | 9.16 | 1.00 (plateau) |
| 4 | 2.71 | 14.95 | 0.61 |
| 2 | 3.66 | 39.01 | 0.24 |
| **1.58** | **2.89** | **18.03** | **0.51 (recovery)** |
| 1 | 3.28 | 26.47 | 0.35 |

The 1.58-bit > 2-bit > 1-bit ordering is preserved on text. Group-wise ternary scaling carries the recovery from synthetic to natural language. This is non-trivial — it confirms the group-wise quantization optimization is a genuine compression-favorable structure, not a synthetic-toy artifact.

### Latent-dim axis — cliff at 8d, same as Toy v2

| dims | perplexity |
|---|---|
| 64 / 48 / 32 | 9-10 (plateau) |
| 16 | 12.4 (inflection) |
| 8 | 34.3 (cliff) |
| 4 / 2 / 1 | 111 / 380 / 603 (catastrophic) |

Cliff position at 8d on text exactly matches Toy v2's latent cliff at 8d. Suggests the cliff is bottleneck-capacity-driven, not task-specific.

### State-window axis — even flatter than synthetic

| window | perplexity |
|---|---|
| 256 → 32 | ~9.2 (zero degradation) |
| 16 | 9.20 |
| 8 | 9.46 |
| 4 | 10.66 (still small) |

Char-level prediction is heavily local; 2-layer model has limited long-range capacity. **The state window can shrink 64× with no measurable loss.** Implication for Mamba-style small-state architectures: there's room to compress context drastically when the task is local.

### Compound compression

- 128× (4-bit, 32d, w=32): perplexity 12.4, retained 0.74 (plateau)
- 256× (2-bit, 32d, w=32): perplexity 47, retained 0.19 (cliff)
- 524288× (1-bit, 1d, 1-token): perplexity 245, retained 0.04 (degenerate)

**Step 1 verdict:** the framework transfers. Compression-aware curriculum doesn't destabilize text training. Ternary rescue, latent-cliff-at-8d, and state-axis-flatness all replicate. Phase 4 Step 2 (latent prediction head for the H_D test on text) is unblocked.

---

# Steam Engine Pre-Phase-3 Test Findings

Tests run before Phase 3 implementation, to validate Phase 2 results and inform Phase 3 design choices. Each batch addresses a specific question about whether Phase 3's planned metrics will work.

## Batch A — Seed sensitivity (in progress)

**Question:** Are Phase 2's findings — especially the per-class MSE inversion at the cliff — robust across model initialization seeds, or seed-lucky?

**Why it matters for Phase 3:**
- If cliff position varies by ±20% across seeds, the smoothness probe needs multi-seed averaging (breaks the "60-second retrain" property)
- If per-class inversion pattern is seed-dependent (Fibonacci-easiest in some seeds, Periodic-easiest in others), the per-class variance metric still works as a *quantity* but the *direction* of inversion is noise — we report |variance| not signed
- If baseline pred_mse is tightly clustered, single-seed Phase 3 results are defensible

**Smoke test (seed=0):** baseline 89.2% acc / 0.358 pred_mse. At 170.7x cliff, Fibonacci easiest (0.68); Periodic and Random hardest (1.12). Phase 2 screenshot (seed=42) showed Periodic easiest at the same compression. **The inversion target appears seed-dependent.**

**Findings (8/8 seeds complete, 495s):**

| seed | base_acc | base_mse | cliff_acc | cliff_mse | cliff_pcv | cliff_easiest |
|---|---|---|---|---|---|---|
| 0 | 0.892 | 0.358 | 0.202 | 0.978 | 0.027 | Fibonacci |
| 1 | 0.866 | 0.357 | 0.242 | 1.077 | 0.008 | Fibonacci |
| 2 | 0.880 | 0.358 | 0.594 | 0.665 | 0.092 | Fibonacci |
| 3 | 0.864 | 0.358 | 0.510 | 0.664 | 0.155 | Fibonacci |
| 4 | 0.890 | 0.353 | 0.522 | 0.781 | 0.060 | Geometric |
| 5 | 0.878 | 0.364 | 0.290 | 0.997 | 0.151 | Fibonacci |
| 6 | 0.890 | 0.362 | 0.470 | 0.795 | 0.081 | Fibonacci |
| 7 | 0.886 | 0.357 | 0.222 | 0.961 | 0.040 | Fibonacci |

**Coefficient of variation (CV = std/mean) by compression and metric:**

| compression | acc CV | pred_mse CV | per-class-variance CV |
|---|---|---|---|
| baseline | 1.2% | 0.8% | 1.2% |
| mild_8x | 2.6% | 1.2% | 2.6% |
| mid_85x | **11.5%** | 2.8% | 6.4% |
| cliff_171x | **29.3%** | **17.3%** | **66.4%** |
| max_32k | 10.5% | 22.9% | 30.2% |

**Headline findings:**

1. **GAME-CHANGER: cliff zone is not a point.** Cliff accuracy varies 20–59% across seeds at identical compression. The PCV metric's CV is 66% at the cliff — almost an order of magnitude noisier than at the plateau (6%). Single-seed Phase 3 measurements at the cliff cannot distinguish raw-vs-latent target effects below this noise floor.
2. **PCV is reliable in the plateau but breaks at the cliff** — exactly where Phase 3 wants to use it. Need multi-seed averaging for cliff-zone metrics.
3. **Cliff inversion is real but seed-dependent.** 7/8 seeds → Fibonacci easiest at cliff_171x. Seed=42 (user's screenshot default) → Periodic easiest. Two basins of attraction exist; Fibonacci is the dominant one.
4. **Two-regime cliff confirmed at scale**: 171x is uniform-noise cliff (low PCV, high mean MSE). 32k is degenerate-output cliff (high PCV, model collapsed to fixed prediction that fits Fibonacci-shaped sequences).

**Implications for Phase 3 design (proposed amendments to plan):**

- Web UI stays single-seed (interactive demo property preserved).
- Add a CLI offline-analysis tool that runs multi-seed (≥4) experiments and reports mean ± std for cliff-zone metrics.
- Reframe "test at cliff_171x" as "test across cliff zone {2-bit, latent ∈ [4,16], w=2}" with seed averaging.
- Phase 3 published findings need explicit mean ± std reporting, not single-shot screenshots.

---

---

## Batch D — Smoothness probe prototype (complete, single seed)

**Question:** does the perturbation-variance metric distinguish plateau-regime from cliff-regime, validating Phase 3's planned smoothness probe?

**Result: NO. Major finding.**

Variance under input ε-perturbation (mean across 4 prediction dims, 32 perturbations, 100 stratified samples):

| ε | baseline | mid_85x | cliff_171x | max_32k | mid/base | cliff/base | max/base |
|---|---|---|---|---|---|---|---|
| 0.01 | 0.000081 | 0.000058 | 0.000079 | 0.000000 | 0.71 | 0.97 | 0.00 |
| 0.05 | 0.001972 | 0.001430 | 0.001935 | 0.000003 | 0.72 | 0.98 | 0.00 |
| 0.10 | 0.007471 | 0.005640 | 0.007218 | 0.000013 | 0.75 | 0.97 | 0.00 |

The probe shows three clear regimes:
1. **Plateau (baseline → mid_85x):** smooth, slightly tighter under compression (regularization)
2. **Cliff (171x):** ~97% of baseline variance — *indistinguishable* from healthy plateau by this metric, even though prediction MSE has tripled
3. **Degenerate (max_32k):** zero variance — model collapsed to a constant function

**Why the probe fails:**
- The training curriculum injects random window masks, latent dropout, weight noise during epochs ≥80. The model *has learned to be invariant* to these perturbations. The probe tests exactly that invariance.
- At the cliff, low variance isn't "smooth landscape" — it's "the bottleneck has destroyed the input→output signal path, so the model is producing near-mean predictions regardless of input." That's a different failure mode than the one the probe was designed to detect.
- Dream-MPC's actual claim is about **weight-perturbation smoothness** (does quantization noise destroy the gradient landscape), not input-perturbation smoothness. Our probe tests the wrong axis.

**Game-changer for Phase 3 plan:**
- The current smoothness probe definition cannot test Dream-MPC's gradient-smoothness claim
- What it does measure is "input-output sensitivity" — useful but different
- Phase 3 needs to either drop the probe (Option α), replace with weight-perturbation (β), replace with gradient-norm (γ), or reframe as input-sensitivity index (δ)

**Recommendation pending user decision: δ + α** — keep what we found and reframe; drop the Dream-MPC test from Phase 3 (defer to Phase 4 with a proper weight-perturbation probe).

---

## Batch B — α (prediction loss weight) sweep (queued)

**Question:** Does α=1.0 balance the heads, or is one dominating? Where does prediction quality saturate vs. classification quality degrade?

**Why it matters for Phase 3:** picks β for the latent prediction loss (the third head)

**Findings:** TBD.

---

## Toy v2 viability gate (3 seeds, 269s)

**Built Toy v2:** SEQ_LEN=64, N_CLASSES=10, PRED_CTX=48, PRED_LEN=16, EPOCHS=120. Same 46K params. New patterns: Quadratic, Exponential, AR(1), Damped oscillator, RandomWalk added; total 10-class predictability spectrum.

**Per-class baseline MSE (single seed) — predictability spectrum:**
- Trivial (~0.01): Arithmetic, Fibonacci, Exponential, Geometric
- Medium: Quadratic (0.027), Damped (0.107)
- Hard-to-predict-but-classifiable: Periodic (0.806 MSE / 98% acc) — recognizes pattern but can't lock phase
- Hard both: AR(1) (0.589 / 49% acc), RandomWalk (0.320 / 53% acc)
- Noise: Random (0.952 / 58% acc)

**v1 vs v2 CV comparison:**

| compression | metric | v1 CV (8 seeds) | v2 CV (3 seeds) | factor |
|---|---|---|---|---|
| baseline | accuracy | 1.2% | 0.8% | 1.5x |
| baseline | pred_mse | 0.8% | 0.1% | 8x |
| baseline | per_class_mse_variance | 1.2% | 0.6% | 2x |
| mid_85x | accuracy | 11.5% | 2.0% | 6x |
| mid_85x | pred_mse | 2.8% | 2.5% | similar |
| mid_85x | per_class_mse_variance | 6.4% | 1.5% | 4x |
| cliff_171x | accuracy | 29.3% | 33.1% | similar |
| cliff_171x | pred_mse | 17.3% | 5.5% | 3x |
| cliff_171x | per_class_mse_variance | **66.4%** | **23.5%** | **3x** |

**Cliff-inversion stability:** All 3 v2 seeds have Fibonacci easiest at cliff (MSE 0.180, 0.188, 0.184). v1 had seed=4 → Geometric and seed=42 → Periodic outliers. v2 is consistent.

**Initial verdict (3-seed sample, REVISED below):**
- Plateau zone (≤ mid_85x): 1-2% CV. Single-seed publishable.
- Cliff zone: 23.5% PCV CV (3 seeds; this turned out to be lucky-low — see correction).
- Cliff accuracy CV (33%) is the new floor — the cliff is genuinely stochastic across initializations.

## CORRECTION — 8-seed sample (2026-05-08, ran "wider" Batch A)

Wider sweep (8 seeds, 702s) gives the honest picture. The 3-seed result was lucky-low: seeds 0/1/2 all landed in the Fibonacci cliff basin; seeds 3-7 spread across Damped, Exponential, Geometric basins. Seed 6 in particular has cliff PCV 0.207 — a high outlier.

| compression | metric | v1 CV (8s) | v2 CV (3s) | v2 CV (8s) | corrected factor |
|---|---|---|---|---|---|
| baseline | accuracy | 1.2% | 0.8% | **0.9%** | 1.3x ✓ |
| baseline | pred_mse | 0.8% | 0.1% | **0.6%** | 1.4x ✓ |
| baseline | per_class_mse_variance | 1.2% | 0.6% | **0.9%** | 1.3x ✓ |
| mid_85x | accuracy | 11.5% | 2.0% | **11.3%** | ~equal |
| mid_85x | pred_mse | 2.8% | 2.5% | **3.3%** | slightly worse |
| mid_85x | per_class_mse_variance | 6.4% | 1.5% | **6.6%** | ~equal |
| cliff_171x | accuracy | 29.3% | 33.1% | **30.7%** | ~equal |
| cliff_171x | pred_mse | 17.3% | 5.5% | **15.1%** | 1.1x ✓ |
| cliff_171x | per_class_mse_variance | **66.4%** | **23.5%** | **44.6%** | **1.5x ✓** |

**Cliff inversion is MORE diverse on v2 (8 seeds), not less:**
- v1: Fibonacci 7, Geometric 1
- v2: Fibonacci 3, Damped 3, Exponential 1, Geometric 1

With 10 classes, more degenerate post-cliff basins exist.

**Reframed verdict:**
- Toy v2 is a better instrument than v1 — but ~1.5x tighter on cliff PCV, not 3x. Plateau metrics ~1.3x tighter. Mid-zone metrics essentially equal.
- **The across-seed CV was the wrong yardstick for Phase 3.** Phase 3's H_A test is paired (raw vs latent at fixed seed). Within-seed correlation between the two heads matters, not across-seed CV of either alone.
- Phase 3 implementation is itself the actual viability test — we cannot pre-validate the paired-difference signal until both heads are trained.
- Multi-seed reporting still mandatory for published cliff results (8-12 seeds, not 4).
- Text-model pivot still deferred to Phase 4+.

## Phase 3 — first multi-seed run on Toy v2 (3 seeds, alpha=1.0, beta=0.5, 317s)

**Question:** does latent-space prediction outperform raw-space prediction under compression (H_A) as JEPA/Dreamer/Dream-MPC predict?

**Result: H_A is REJECTED.** Divergence index is negative across all 3 seeds and all 4 compression points. The standard deviation is small (~10-20% of mean) — the result is robust.

**Divergence index (= latent_retained − raw_retained), 3 seeds:**

| compression | raw_retained avg | latent_retained avg | divergence | std | direction |
|---|---|---|---|---|---|
| baseline | 1.000 | 1.000 | 0.000 | 0.000 | n/a |
| mild_8x | 0.823 | 0.478 | -0.345 | 0.052 | H_D (latent worse) |
| mid_85x | 0.729 | 0.382 | -0.347 | 0.076 | H_D |
| cliff_171x | 0.239 | 0.042 | -0.197 | 0.013 | H_D |
| max_131k | 0.204 | 0.065 | -0.138 | 0.030 | H_D |

**Mechanism — per-class breakdown at cliff_171x (seed 0):**

| class | base_raw | base_lat | cliff_raw | cliff_lat | raw_ret | lat_ret |
|---|---|---|---|---|---|---|
| Arithmetic | 0.010 | 0.068 | 1.109 | 1.682 | 0.009 | 0.040 |
| Geometric | 0.011 | 0.058 | 1.457 | 1.910 | 0.007 | 0.030 |
| Fibonacci | 0.010 | 0.076 | 0.153 | 2.413 | 0.067 | 0.031 |
| AR1 | 0.587 | 0.300 | 1.134 | 1.356 | 0.518 | 0.221 |
| **Periodic** | **0.817** | **0.063** | 1.501 | 1.467 | **0.544** | **0.043** |
| Random | 0.959 | 0.146 | 1.407 | 1.430 | 0.682 | 0.102 |

The mechanism is clean: classes where raw prediction was already saturating at noise floor (Periodic 0.82, Random 0.96, AR1 0.59 — these are the unpredictable-by-design classes) have nothing further to fall under compression — raw_retained stays high (0.5-0.7). Their latent targets at baseline were *very easy* (0.063, 0.146, 0.300) because the encoder partially collapsed toward class-identity representation. Under compression the compressed-online's latent diverges from the uncompressed-teacher's latent and relative degradation explodes.

**Connection to [[wiki/representation-collapse.md]]:** the parallel raw-head was supposed to prevent latent collapse. It worked enough to keep training stable but not enough to keep the latent rich. Latent partially collapsed toward "what class is this" — an easy target at baseline, brittle under compression.

**Three implications:**

1. **JEPA's compression-friendliness claim does not hold in this toy.** Phase 3 measured the opposite of what Dream-MPC and JEPA predicted.
2. **The result is publishable as a contrarian finding** — robust at 3 seeds, signs all consistent, with a clean mechanistic explanation tied to representation-collapse literature.
3. **Whether the result generalizes is the next question.** Possible explanations to rule out: β=0.5 too low (Option B in plan), asymmetric target setup hurting latent (Option C), latent target dim 64 vs raw 16 too unfair, or the toy's synthetic patterns being fundamentally too clean.

## Phase 3 β-sweep — Option B answer (12 runs, 1289s)

**Question:** is the negative divergence index a β=0.5 underweighting artifact?

**Result: NO. β tuning cannot rescue H_A.** Tested β ∈ {0.5, 1.0, 2.0, 5.0} × 3 seeds. **48 of 48 cells (100%) are negative**, and the trend is monotonic: higher β makes divergence MORE negative through β=2.0, then a small U-turn at β=5.0 (still negative everywhere).

| β | mild_div | mid_div | cliff_div | max_div |
|---|---|---|---|---|
| 0.5 | -0.345 | -0.347 | -0.197 | -0.138 |
| 1.0 | -0.433 | -0.485 | -0.322 | -0.153 |
| **2.0** | **-0.507** | **-0.599** | **-0.444** | **-0.174** |
| 5.0 | -0.394 | -0.534 | -0.400 | -0.138 |

**Mechanism confirmed:** higher β trains the latent head harder, which pushes the encoder to produce easier-to-predict latent targets at baseline (baseline_lat 0.128 at β=0.5 → 0.082 at β=5.0). That "easier" target is precisely what diverges most when the compressed online encoder cannot match the uncompressed teacher's representation. This is a clean empirical signature of the [[representation-collapse]] failure mode the JEPA literature warned about.

**Phase 3 finding is robust:**
- 3 independent runs (single-seed prototype, 3-seed multi-seed, 4-β×3-seed sweep) all show negative divergence
- 100% of measured cells (60+ data points) are negative
- The result is not design-sensitive within the β knob

**Remaining untested knobs (Option B sub-options):**
- Train longer (120 → 250 epochs) — convergence test
- Shrink target latent dim (64 → 16, matching raw target) — dimensionality-asymmetry test
- Use compressed teacher (symmetric setup) — asymmetric-target test

**Recommendation:** the β sweep is a strong negative result for "design-sensitive." Declaring the finding (Option A) is justified. If user wants one more sanity check, target-dim shrink is the cleanest remaining test (different *structure* not just a knob).

---

## Open questions to potentially address

- **Batch C — Fine latent-axis sweep with per-class variance**: where exactly does class-structure collapse? Phase 2 sweep is at {64, 48, 32, 16, 8, 4, 2, 1} dims; we may want {64, 56, 48, 40, 32, 24, 16, 12, 8, 6, 4, 3, 2, 1} to characterize the inflection.
- **Batch D — Smoothness probe prototype on Phase 2 raw predictions**: validate the perturbation-variance metric works on existing model before committing it to Phase 3. If raw prediction is already smooth/jagged in interpretable ways, the latent version will be too.
- **Batch E — Compression-aware curriculum sensitivity**: warmup epoch (currently 80) controls when compression augmentation kicks in. Does shifting it move the cliff?
- **Batch F — Training duration**: 250 epochs current. Where's the diminishing-returns point? Does Phase 3's three-loss objective need more epochs?
