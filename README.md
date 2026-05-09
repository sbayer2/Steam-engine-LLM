# Steam Engine — Small-State Thesis Demo

**An interactive science instrument that asks: how much can you crush an AI model before it stops working — and which parts crush gracefully versus which parts cliff?**

Drag three sliders. Watch a small neural network get squeezed in three different ways at once. See exactly where it survives and where it breaks. ~46K parameters, trains in ~90 seconds, demonstrates a real research finding the paper authors haven't published yet.

![preview placeholder — local at http://localhost:8000 once running]

## The 30-second pitch (no AI background required)

Modern AI models are huge — billions of weights, gigabytes of memory. Researchers want to shrink them to run on phones, robots, and embedded devices. There are three ways to shrink:

1. **Use less precision per number.** Instead of 32-bit floats, use 4-bit integers, or even 1-bit (just +1 or −1).
2. **Use fewer "thought channels" inside.** The middle layers of a model can have 64 dimensions or 8.
3. **Remember less of the input.** Look at only a small window of recent inputs instead of everything.

The naive expectation: shrink any of these and quality drops proportionally. Make the model 10× smaller, lose 10× the quality.

**The actual finding** (across multiple research labs, including this demo): you can shrink a *well-designed* model **128 times smaller** and still keep **78% of its accuracy**. The relationship looks like a hockey-stick — a long flat plateau, then a sudden cliff. Quality survives brutal cuts, then collapses suddenly. This is the **small-state thesis**: structure (what the model fundamentally knows) survives compression in ways precision (the floating-point weights) does not.

This demo lets you watch that hockey-stick in real time. It also surfaces a *new* finding — that one popular optimization technique (predicting in "abstract latent space" instead of "raw value space") is actually *more* compression-fragile than the simpler alternative, contradicting recent papers.

## What you see when you run it

- **Three sliders** for the three compression axes — Bit-Width, Latent Dim, State Size
- **A live accuracy gauge** plus a per-class breakdown showing which patterns the model still recognizes
- **Sweep charts** that pre-compute what happens along each axis in isolation
- **A "Divergence" card** that reveals the new Phase 3 finding (described below)
- **Live forecasts** with sparklines comparing the model's predictions to actual sequence values

The model is trained from scratch on synthetic time-series patterns: linear trends, exponential decay, oscillations, recurrent rules, random noise — 10 distinct pattern types in 64-token sequences. Simple enough to understand, rich enough to show real signal.

## Run locally

```bash
git clone https://github.com/sbayer2/Steam-engine-LLM.git
cd Steam-engine-LLM

# Set up environment (Python 3.10+)
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Run
python main.py
```

Open **http://localhost:8000**. The model trains on startup (~90s on CPU), then the UI is live. Drag sliders, watch metrics update.

## The three compression axes

Each axis cuts a different dimension of the model's representation cost. They are orthogonal — you can apply them independently or stack them.

| Axis | What it cuts | Inspired by | Slider range | What it tests |
|------|-------------|-------------|--------------|---------------|
| **Bit-Width** | Precision per parameter | BitNet b1.58 (Ma et al., 2024), Bonsai-8B (PrismML, 2026) | FP32 → 1-bit binary | Does the model rely on the *exact value* of each weight, or just on the *direction* (positive vs negative)? |
| **Latent Dim** | What is predicted | JEPA (LeCun, 2022), V-JEPA 2 (Bardes et al., 2025) | 64 → 1 dimension | How much "thinking space" does the model need internally? Can the bottleneck be tiny? |
| **State Size** | What is remembered | Mamba (Gu & Dao, 2023), DreamerV3 (Hafner et al., 2024) | 64 → 1 token window | How much of the input does the model actually need to attend to? |

The shared bet behind all three: **the structure of what's being represented is much lower-dimensional than the surface form, and finding that structure is the whole game.** Cuts that respect structure preserve capability. Cuts that don't, don't.

## The trick: compression-aware training

A model trained at full precision and then compressed afterward usually collapses. The honest version of the thesis is: **structure survives brutal cuts when the system is designed for it.**

This demo's training has a curriculum:

1. **Epochs 0–40**: Standard training. The model learns the task at full fidelity.
2. **Epochs 40–120**: Compression-aware training. Each batch randomly applies:
   - **Window masking** (40% of batches): attention is restricted to a random small window, forcing locally-identifiable features
   - **Latent dropout** (40% of batches): random latent dimensions are zeroed, forcing distributed representations
   - **Weight noise injection** (every 5 epochs): Gaussian noise simulates post-training quantization

The model learns to expect compression. By the time you crush it at inference, it's been trained to be invariant to that crushing. Same idea is used (under different names) in BitNet's training recipe, JEPA's stop-gradient + EMA, and Dreamer's RSSM design.

## The architecture: triple-head model

This started as a classification toy and grew. Current state:

```
        ┌─────────────────────────────────┐
        │  Shared encoder (~46K params)   │
        │  embed → attn → bottleneck      │
        └──────────┬──────────────────────┘
                   │ pooled latent
                   ├────────────────┬────────────────┐
                   ▼                ▼                ▼
        ┌────────────────┐ ┌────────────────┐ ┌─────────────────┐
        │ Class head     │ │ Raw pred head  │ │ Latent pred head│
        │ → 10 classes   │ │ → 16 values    │ │ → 64-dim latent │
        └────────────────┘ └────────────────┘ └─────────────────┘

        Joint loss: L = CE(class) + α·MSE(raw) + β·MSE(latent)
```

- **Class head**: pattern-type recognition (10 classes). Phase 1.
- **Raw pred head**: forecast next 16 values from first 48. Phase 2.
- **Latent pred head**: predict the *encoder's own representation* of the next 16 values. Phase 3 — this is the JEPA-style approach the field has been excited about.

The third head is trained against an "uncompressed teacher" — the same model running with no compression on the full sequence, with stop-grad. The compressed online encoder learns to predict what the uncompressed teacher's representation would be. This setup mirrors how JEPA, Dreamer, and Dream-MPC actually work in production world-model systems.

## Headline findings

### Phase 2 — Compression hurts prediction differently than classification

Latent-axis compression is the sharpest cliff. Bit-width has a sharper drop but recovers slightly at ternary (1.58-bit) due to group-wise scaling. State-size compression barely matters — most signal is in the first 12-48 tokens.

Per-class behavior reveals the structure of failure: under heavy compression, the easy deterministic patterns (Arithmetic, Geometric, Quadratic, Exponential) **collapse together** into one indistinguishable cluster, while distinctive patterns (Fibonacci, Periodic, AR(1), Random) survive in their own basins. The 8d-latent partial-collapse regime keeps 4 of 10 classes near baseline accuracy and crashes the other 6 to near zero.

### Phase 3 — JEPA-style latent prediction is *more* compression-fragile than raw prediction

This is the surprise. The wiki literature ([[dream-mpc]], [[v-jepa-2]], [[leworldmodel]]) suggests latent-space prediction should be compression-friendly because targets are smooth and low-dimensional. **Steam Engine measured the opposite.**

Across **48 measured cells** (4 β values × 3 random seeds × 4 compression points), 100% are negative — meaning latent prediction always retained less performance than raw prediction under compression. The result is robust to loss weighting (β tuning makes it monotonically worse, not better).

Mechanism (empirically confirmed): the encoder partially collapses toward easy class-identity targets at baseline, making latent prediction easier than raw prediction at full precision. Under compression, the compressed online encoder can't match the uncompressed teacher's representation, and the easier-baseline target diverges further. This is the [representation-collapse failure mode](https://en.wikipedia.org/wiki/Joint_embedding_predictive_architecture) the JEPA literature warned about — empirically detected by the divergence index even when training loss is stable.

### The cliff is stochastic, not a fixed point

Eight seeds at identical hyperparameters: cliff-region accuracy ranges 20–59%. Different model initializations land in different post-cliff basins. Cliff position has CV ~30% across seeds even on the larger toy. This is a real property of compression-edge behavior, not a measurement artifact. Production deployment decisions based on a single seed are therefore unreliable.

## Practical applications — what this means for real models

The findings transfer in concrete ways. Five examples ranked by directness:

### 1. Edge LLMs (BitNet, Bonsai-class)
Compression-aware training curriculum is the actual recipe. Confirmed empirically that naive PTQ on a normal-trained model fails immediately while the same model trained with random window/latent/weight perturbation survives 8-16× compression. **Optimization:** add the perturbation schedule to your QAT recipe. For deployment guidance: profile per-task accuracy at the target compression level (not averaged perplexity). Tasks with shared structural fingerprints collapse together; tasks with distinctive fingerprints survive.

### 2. Robot VLAs — π0 vs V-JEPA 2-AC under quantization
Phase 3's H_D finding directly conflicts with the field's intuition. The cleanest single test: quantize both V-JEPA 2-AC (latent target) and π0 (direct action target) to int4, run identical manipulation benchmarks, compare retained task success. Steam Engine predicts V-JEPA 2-AC degrades more. **Cost: ~$200 of GPU time, 2 days.** Result would directly contradict or confirm Dream-MPC's compression-friendliness framing on production architectures.

### 3. Embedding-based retrieval (RAG, semantic search)
Per-class collapse pattern translates: under aggressive embedding quantization (1-bit, binary), queries with shared semantic structure produce identical retrieval results — distinct queries get merged into one cluster. The failure mode isn't uniform precision loss; it's *cluster collapse*. **Optimization:** measure pairwise discrimination retention before and after compression. Standard "retrieval accuracy" metrics miss this structure of failure.

### 4. DreamerV3 deployment on edge devices
The compression-aware curriculum maps directly to RSSM hidden-state augmentation. Training with random RSSM-state masking + imagination horizon truncation + weight noise should produce a Dreamer that survives 4-bit weight quantization. **Risk:** Phase 3's H_D finding suggests the imagination rollout itself (latent-space prediction) may be especially fragile. The deployed policy could fail in ways the offline checkpoint wouldn't show.

### 5. Time-series forecasting models (TimesFM, Chronos)
Add a regime-classification head alongside the value-forecast head, train jointly. Steam Engine's data shows the joint training gives the encoder richer representations than single-task training, and under compression the regime head provides interpretability ("model thinks we're in a seasonal regime; here's the forecast") that black-box quantized forecasts can't.

## Research directions

What this demo's data licenses, but doesn't yet test:

1. **Validate H_D on production VLAs.** The V-JEPA 2-AC vs π0 quantization comparison is the cleanest experiment in the field that nobody has run.
2. **Phase 4 — text-model scaling.** Tiny TinyStories character-level transformer with the same dual/triple-head architecture. Tests whether the synthetic-pattern findings generalize to real distributions. ~1-2 days of work.
3. **Phase 4 — gradient-norm probe.** Replace the deprecated input-perturbation probe (which measures input-output sensitivity, not landscape jaggedness) with a proper weight-perturbation or Jacobian-norm probe. This actually tests Dream-MPC's gradient-smoothness concern.
4. **Sub-1-bit representations.** Grouping or learned sparsification could push effective bit-widths below 1.0. Anticipated by the intelligence-density framing; the artifact does not yet exist.
5. **Three-axis composition architectures.** A 1.58-bit Mamba-backbone JEPA encoder for edge-deployed robot policies. Each axis demonstrated independently; no one has published the triple composition. The composability matrix has empty slots waiting.
6. **Why is the cliff stochastic?** Different seeds land in different post-cliff basins. Is there a principled characterization of basin geometry, or is it irreducibly chaotic? Connects to the wider question of when neural networks have well-defined "task-specific" modes.
7. **Where does small-state lose?** DIAMOND tops Atari100k at 145.9% HNS, beating DreamerV3's 112.4%. Full attention wins specific benchmarks. The thesis is a tendency, not a law. Characterizing the regime boundary (which task properties favor full attention?) would make the framework predictive.

## Pattern set (Toy v2)

Ten classes, designed to span a predictability gradient:

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

Note: AR(1), RandomWalk, and Random have low classification accuracy by design — they share noise-floor characteristics. They're informative about *prediction* (where their MSE saturates near 1.0 in standardized units) more than classification.

## Code structure

```
steam-engine/
├── main.py               # FastAPI server
├── engine.py             # PyTorch model, data generation, compression engine
├── requirements.txt
├── frontend/
│   ├── index.html        # Single-page UI
│   ├── app.js            # Sliders + Chart.js + live updates
│   └── style.css         # Dark research aesthetic
└── tests/
    ├── batch_a_seed_sensitivity.py        # Multi-seed cliff CV measurement
    ├── batch_phase3_v2.py                 # Phase 3 raw vs latent comparison
    ├── batch_phase3_beta_sweep.py         # β robustness validation
    ├── batch_d_smoothness_prototype.py    # Deprecated input-perturbation probe
    └── FINDINGS.md                        # Empirical record across all batches
```

## Literature

- Ma, S. et al. (2024). *The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits.* arXiv:2402.17764.
- Bardes, A. et al. (2025). *V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction and Planning.* arXiv:2506.09985.
- Spieler, J. & Behnke, S. (2026). *Dream-MPC: Gradient-Based Model Predictive Control with Latent Imagination.* ICML 2026. — The paper that motivated Phase 3's smoothness probe (which Steam Engine then deprecated; see ADC-007).
- Gu, A. & Dao, T. (2023). *Mamba: Linear-Time Sequence Modeling with Selective State Spaces.* arXiv:2312.00752.
- Hafner, D. et al. (2024). *Mastering Diverse Domains through World Models.* Nature.
- LeCun, Y. (2022). *A Path Towards Autonomous Machine Intelligence.* — The JEPA manifesto.
- Brohan, A. et al. (2024). *OpenVLA: An Open-Source Vision-Language-Action Model.*
- Black, K. et al. (2024). *π0: A Vision-Language-Action Flow Model for General Robot Control.*
- Wang, Z. et al. (2024). *Is Sora a World Simulator? A Comprehensive Survey on General World Models and Beyond.* arXiv:2405.03520.

## License

MIT
