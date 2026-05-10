# Steam Engine — Small-State Thesis Demo

**An interactive science instrument that asks: how much can you crush an AI model before it stops working — and which parts crush gracefully versus which parts cliff?**

Drag three sliders. Watch a small neural network get squeezed in three different ways at once. See exactly where it survives and where it breaks. Two engines run side-by-side — a 51K-parameter synthetic toy and a 127K-parameter char-level GPT trained on Shakespeare + Austen + code. Both trained from scratch on CPU in minutes.

## The 30-second pitch (no AI background required)

Modern AI models are huge. Researchers want to shrink them to run on phones, robots, and embedded devices. There are three ways to shrink:

1. **Use less precision per number.** 32-bit floats → 4-bit integers → 1-bit (just +1 or −1).
2. **Use fewer "thought channels" inside.** Middle layers can have 64 dimensions or 8.
3. **Remember less of the input.** A small attention window over recent inputs instead of everything.

The naive expectation: shrink any of these and quality drops proportionally. **The actual finding** (across multiple research labs, including this demo): you can shrink a *well-designed* model 100× smaller and still keep most of its accuracy. The quality-vs-compression curve looks like a hockey stick — a long flat plateau, then a sudden cliff. This is the **small-state thesis**: structure (what the model knows) survives compression in ways precision (the floating-point weights) does not.

This demo lets you watch the hockey-stick in real time. It also produces a **new mechanistic finding**: whether predicting in abstract "latent space" is *more* or *less* compression-friendly than predicting in raw observation space depends on a property of the training setup you can measure before deploying any compression. This refines a contested literature claim (JEPA-style training is compression-friendly) into a falsifiable mechanism.

## What you see when you run it

Two engine tabs at the top of the page:

**Synthetic Toy v2** (default tab) — a 51K-param 2-layer encoder trained on 10 synthetic time-series patterns. Three sliders compress along bit-width / latent-dim / attention-window. Per-class survival bars show which patterns crash first. Sweep charts pre-compute axis-by-axis cliffs. Live forecasts compare model predictions to actual sequence values.

**Text (Phase 4)** — a 127K-param char-level GPT, trained jointly on three corpora (Shakespeare, Austen, Python source). Click "Train Text Engine" first time (~8 min on CPU). Then the same compression sliders work, but now you get perplexity instead of accuracy, per-corpus retention bars, and a live text-generation panel — type a prompt, generate under whatever compression you've dialed in. Watch coherence degrade in real time.

## Run locally

```bash
git clone https://github.com/sbayer2/Steam-engine-LLM.git
cd Steam-engine-LLM

python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python main.py
```

Open **http://localhost:8000**. Synthetic trains on startup (~90s). Text trains on demand from its tab (~8 min). Drag sliders, watch metrics update.

## The three compression axes

Each axis cuts a different dimension of the model's representation cost. Orthogonal — apply independently or stack.

| Axis | What it cuts | Inspired by | What it tests |
|------|-------------|-------------|---------------|
| **Bit-Width** | Precision per parameter | BitNet b1.58, Bonsai-8B | Does the model rely on exact weight values, or just on direction (sign)? |
| **Latent Dim** | What is predicted | JEPA, V-JEPA 2 | How much "thinking space" does the model need internally? |
| **State Size** | What is remembered | Mamba, DreamerV3 RSSM | How much of the input does the model actually need to attend to? |

Shared bet behind all three: **the structure of what's being represented is much lower-dimensional than the surface form, and finding that structure is the whole game.** Cuts that respect structure preserve capability. Cuts that don't, don't.

## The trick: compression-aware training

A model trained at full precision and then compressed afterward usually collapses. The honest version of the thesis is: **structure survives brutal cuts when the system is designed for it.**

Each batch randomly applies during training:
- **Window masking** (40% probability): attention restricted to a random small window
- **Latent dropout** (40% probability): random latent dimensions zeroed
- **Weight noise injection** (every 5 epochs): Gaussian noise on weights simulating post-training quantization

The model learns to expect compression. Crush it at inference and it's already invariant to that crushing. Same recipe is used (under different names) in BitNet's training, JEPA's stop-gradient + EMA, Dreamer's RSSM design.

## Two engines, same testing framework

```
Synthetic Toy v2 (engine.py)           Text Engine (text_engine.py)
~51K params                            ~127K params
2-layer attention encoder              2-layer causal GPT (nanochat-inspired)
10 pattern classes / 64 tokens         3 corpora / 256 tokens / 114-char vocab
                                       (Shakespeare + Austen + Python code)
        │                                       │
        └─── triple-head architecture ───────────┘
              │
              ├── Classification head (synthetic only)
              ├── Raw prediction head    (next 16 values / next-token CE)
              └── Latent prediction head (JEPA-style)

Joint loss: L = (CE | classification) + α·MSE_raw + β·MSE_latent
Target encoder: same model, stop_grad, uncompressed (asymmetric setup)
```

Both engines use the same compression-aware curriculum, the same three orthogonal compression axes, the same multi-seed measurement framework, and the same divergence-index metric (`latent_retained − raw_retained`) to compare the two prediction heads' compression-robustness.

## Headline findings — the cross-domain reversal and the mechanism

The project's primary empirical contribution: **a single mechanism explains opposite outcomes on different domains, and that mechanism is measurable before deploying compression.**

### The reversal

| Setting | Target distribution | Cells measured | Direction |
|---|---|---|---|
| Synthetic Toy v2 | 10 sharp pattern classes | 48/48 (4 β × 3 seeds × 4 compression points) | **ALL NEG (H_D)** — latent worse than raw under compression |
| Text TinyShakespeare alone | continuous chars | 30/30 non-trivial cells | **ALL POS (H_A)** — latent better than raw |
| Text multi-corpus (within one model) | 3 corpora differ | per-corpus 3/3 sign consistency | **direction depends on corpus** |

Same architecture pattern. Same training recipe. Same compression-axis testing framework. **Opposite outcomes — driven entirely by the structure of the target distribution.**

### The mechanism

**Compression-robustness direction tracks baseline latent MSE.** Lower baseline lat_mse → encoder collapsed onto an "easy summary" → fragile under heavy compression (H_D). Higher baseline lat_mse → encoder forced to maintain rich representation → robust (H_A).

Concrete per-corpus evidence from one jointly-trained text model:

| corpus | baseline lat_mse | rank | bits_2 divergence | cliff_171x divergence |
|---|---|---|---|---|
| Austen | 0.11 | lowest (most collapsed) | **−0.12** (H_D) | **−0.12** (H_D) |
| Shakespeare | 0.16 | middle | mild | mild |
| Code | 0.24 | highest (most diverse) | **+0.14** (H_A) | **+0.11** (H_A) |

All three measurements 3/3 sign-consistent across seeds. The baseline-lat-MSE rank perfectly orders the divergence direction at heavy compression.

### Why JEPA's compression-friendliness claim is conditional

The JEPA literature (LeCun 2022, V-JEPA 2, Dream-MPC) argues latent-space prediction is compression-friendly because targets are smooth and low-dimensional. **This is true — but only when the encoder fails to collapse.** When the training target rewards a low-dim shortcut (categorical labels, easy summarizable patterns), the encoder takes the shortcut. The collapsed latent is highly predictable at baseline (low MSE) but fragile under compression because precise weight values are what hold the collapse together.

When the training target structurally prevents collapse (continuous high-dim diverse targets like masked-region prediction on real video, contrastive losses with negative pairs, char-level next-position prediction with rich vocabulary), the encoder can't shortcut. Baseline MSE stays moderate. Compression preserves direction (the "Hamming bit" of the latent vector) and the prediction stays in the right neighborhood — MSE in continuous space scales gracefully with small shifts.

### Most striking single result

On the text model at `latent_4` (4 latent dimensions out of 64): perplexity drops to 7.2% retained — text generation is broken — but latent prediction stays at **73.4% retained**. Divergence **+0.66 across 3 seeds**. JEPA's "predict the meaning, not the surface" empirically validated under aggressive compression on natural text, with mechanistic explanation tying compression-friendliness to non-collapse target distributions.

### The cliff is stochastic

Eight seeds at identical hyperparameters on synthetic: cliff-region accuracy ranges 20–59%. Different model initializations land in different post-cliff basins. CV ~30% even on Toy v2. **Production deployment decisions based on a single seed are unreliable.** Steam Engine reports mean ± std over ≥3 seeds for all published numbers.

## Practical applications — refined predictions

The cross-domain mechanism produces concrete, falsifiable predictions for real architectures:

### 1. Robot VLAs — predict per-architecture compression survival in advance
- **OpenVLA (discrete action tokens)** → next-token-like categorical target → expect H_D under int4 quantization. Action distributions degrade faster than auxiliary heads would.
- **V-JEPA 2-AC (continuous embedding actions)** → continuous high-dim target → expect H_A territory. Latent action heads should survive aggressive quantization.
- **π0 (flow matching)** → continuous but action-anchored → behavior is in between; depends on flow predictor's effective latent dimensionality.

The cleanest test: int4 quantization of both V-JEPA 2-AC and π0, identical OpenX manipulation benchmarks, compare retained task success. **~$200 of GPU time. Steam Engine predicts V-JEPA 2-AC survives better.**

### 2. SSL-pretrained encoders for edge deployment
DINO, SimCLR, JEPA-family pretraining uses continuous high-dim diverse targets with explicit anti-collapse mechanisms. **Predict H_A under quantization** — these encoders should quantize 2-5× better than supervised-classification-trained equivalents of the same size, even before any quantization-aware retraining.

### 3. LLMs with mixed objectives — per-head quantization budgets
An LLM trained with both next-token cross-entropy and an auxiliary latent SSL head will see different compression fragility per head. Steam Engine's mechanism predicts:
- LM head (discrete vocab CE) → H_D-prone under quantization
- Latent SSL head (continuous embedding MSE) → H_A-territory

**Implication:** keep the LM head at higher precision while aggressively quantizing the auxiliary head. Current uniform-quantization recipes (per-tensor or per-layer quantization configs) miss this structure.

### 4. Embedding-based retrieval (RAG, semantic search)
Per-class collapse pattern translates: under aggressive embedding quantization, queries with shared semantic structure get merged into one cluster. The failure mode is *cluster collapse*, not uniform precision loss. Measure pairwise discrimination retention before and after compression — standard "retrieval accuracy" metrics miss this.

### 5. Edge LLMs (BitNet, Bonsai, Ternary Bonsai)
Compression-aware training curriculum is the actual recipe. Confirmed empirically across both synthetic and text: naive PTQ on a normally-trained model fails immediately; the same model trained with random window/latent/weight perturbation survives 8-16× compression. **Add the perturbation schedule to your QAT recipe.** Group-wise ternary (1.58-bit) recovery is real and reproduces across domains.

### 6. The pre-deployment diagnostic
**Before quantizing a model, measure baseline latent MSE on its target objective.** Rank the model's heads by baseline MSE. Low MSE → expect H_D under aggressive quantization on that head's task. High MSE → expect H_A. This is the most actionable single output of Steam Engine: a measurable structural property that predicts compression-robustness direction across architectures and tasks, before any compression is applied.

## Research directions

What this demo has done and what it hasn't:

**Completed and shipped:**
- ✓ Phase 2 dual-head model (classification + raw value prediction)
- ✓ Phase 3 triple-head model with JEPA-style latent prediction; H_D measured on synthetic
- ✓ Toy v2 architectural pivot (10 classes / 64 tokens) to reduce seed noise
- ✓ Phase 4 Steps 1–3: char-level GPT framework, latent head, multi-corpus joint training, UI integration; H_A confirmed on text; per-corpus mechanism validated

**Open, queued for next sessions:**
1. **β sweep on text** — symmetric counterpart to the synthetic β sweep that already confirmed H_D was tuning-robust. Hypothesis: higher β should amplify H_A on text. Mechanism falsification.
2. **Pretrained nanochat / production checkpoint validation** — apply Steam Engine's quantization engine to a real pretrained model with both LM and auxiliary heads, measure per-head compression survival. Tests whether mechanism transfers to production scale.
3. **Adversarial synthetic patterns** — design pattern classes that should produce H_D versus H_A according to the mechanism (collapse-prone vs collapse-resistant by construction), measure ahead of time. Predictiveness test.
4. **Gradient-norm probe** — replaces the deprecated input-perturbation probe. Tests Dream-MPC's gradient-smoothness claim directly via weight-perturbation or Jacobian-norm measurements.
5. **Model capacity scale-up** — depth=4 / n_embd=128 text model would produce real Shakespeare words at baseline, making the live-generation demo more visceral without changing any empirical findings.
6. **Sub-1-bit representations** — grouping / learned sparsification to push effective bit-widths below 1.0.
7. **Three-axis composition architectures** — 1.58-bit Mamba-backbone JEPA encoder for edge robot policies. Composability matrix has empty slots waiting.

## Pattern set (synthetic Toy v2)

Ten classes spanning a predictability gradient:

| Class | Description | Baseline accuracy |
|---|---|---|
| Arithmetic | a + b·t | 98% |
| Geometric | a · r^t | 82% |
| Quadratic | a + b·t + c·t² | 74% |
| Exponential | a · exp(−k·t) + offset | 72% |
| Fibonacci | s[n] = 0.5·(s[n−1]+s[n−2]) | 100% |
| AR(1) | s[n] = ρ·s[n−1] + ε | 49% |
| Periodic | sin(ω·t + φ) | 98% |
| Damped | exp(−k·t)·sin(ω·t) | 94% |
| RandomWalk | cumulative noise | 53% |
| Random | pure noise | 58% |

AR(1) / RandomWalk / Random have low classification accuracy by design — they're informative about *prediction* (MSE saturates near 1.0 in standardized units) rather than classification.

## Code structure

```
steam-engine/
├── main.py                                  # FastAPI server, both engines, /api/text/* endpoints
├── engine.py                                # synthetic Toy v2 engine, triple-head
├── text_engine.py                           # char-level GPT engine, dual-head, multi-corpus
├── requirements.txt
├── data/                                    # downloaded on first run (gitignored)
│   ├── tinyshakespeare.txt
│   ├── pride_and_prejudice.txt
│   └── code_corpus.txt
├── frontend/
│   ├── index.html                           # Single-page UI with engine-tab switcher
│   ├── app.js                               # Sliders + Chart.js + live updates for both engines
│   └── style.css                            # Dark research aesthetic
└── tests/
    ├── batch_a_seed_sensitivity.py          # Multi-seed cliff CV measurement (synthetic)
    ├── batch_phase3_v2.py                   # Phase 3 raw vs latent comparison (synthetic)
    ├── batch_phase3_beta_sweep.py           # β robustness validation (synthetic)
    ├── batch_d_smoothness_prototype.py      # Deprecated input-perturbation probe
    ├── batch_phase4_text_seeds.py           # Phase 4 multi-seed (single-corpus text)
    ├── batch_phase4_multi_corpus.py         # Phase 4 multi-corpus per-corpus mechanism
    └── FINDINGS.md                          # Empirical record across all batches
```

For full decision history including the abandoned input-perturbation probe (ADC-007), the mechanism-prediction reversal (ADC-016), and the capacity-vs-polish tradeoff (ADC-019), see the project page in the companion vault at `wiki/steam-engine.md` (Architecture Decision Cycle log, ADC-001 through ADC-020).

## Literature

- Ma, S. et al. (2024). *The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits.* arXiv:2402.17764.
- Bardes, A. et al. (2025). *V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction and Planning.* arXiv:2506.09985.
- Assran, M. et al. (2023). *Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture.* CVPR 2023 (I-JEPA).
- Spieler, J. & Behnke, S. (2026). *Dream-MPC: Gradient-Based Model Predictive Control with Latent Imagination.* ICML 2026.
- Gu, A. & Dao, T. (2023). *Mamba: Linear-Time Sequence Modeling with Selective State Spaces.* arXiv:2312.00752.
- Hafner, D. et al. (2024). *Mastering Diverse Domains through World Models.* Nature.
- LeCun, Y. (2022). *A Path Towards Autonomous Machine Intelligence.* The JEPA manifesto.
- Brohan, A. et al. (2024). *OpenVLA: An Open-Source Vision-Language-Action Model.*
- Black, K. et al. (2024). *π0: A Vision-Language-Action Flow Model for General Robot Control.*
- Karpathy, A. (2025). *nanochat: minimal end-to-end ChatGPT-like training pipeline.* github.com/karpathy/nanochat — architectural reference for the Phase 4 text engine.

## License

MIT
