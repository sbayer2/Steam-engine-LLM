# Steam Engine — Small-State Thesis Demo

An interactive demo that proves a simple claim: **high-dimensional abstract structure survives brutal information cuts**. Like a steam engine proved thermodynamics by running, this proves the small-state thesis by running a neural network under extreme compression and watching capability survive.

A 44K-parameter sequence classifier is trained with compression-awareness, then squeezed along three orthogonal axes simultaneously. At **128× total compression**, it still retains **~78% of baseline accuracy**. The structure of the task — not the precision of the weights — is what carries capability.

## The three axes

| Axis | What it cuts | Inspiration | Slider range |
|------|-------------|-------------|--------------|
| **Bit-width** | Precision per parameter | BitNet b1.58 (Ma et al., 2024), Bonsai-8B (PrismML, 2026) | FP32 → 1-bit binary |
| **Latent dimensionality** | What is predicted | JEPA (LeCun, 2022), V-JEPA 2 (Bardes et al., 2025) | 64 → 1 dimension |
| **Sequence state** | What is remembered | Mamba (Gu & Dao, 2023), DreamerV3 RSSM (Hafner et al., 2024) | 16 → 1 token window |

Each axis operates on a different dimension of representation cost. The shared assumption is identical: **the structure of what's being represented is much lower-dimensional than the surface form, and finding that structure is the whole game**. Cuts that respect structure preserve capability. Cuts that don't, don't.

## Run locally

```bash
# Clone
git clone https://github.com/sbayer2/Steam-engine-LLM.git
cd Steam-engine-LLM

# Set up environment (requires Python 3.10+)
uv venv .venv && source .venv/bin/activate
uv pip install torch fastapi uvicorn numpy

# Or with pip
python -m venv .venv && source .venv/bin/activate
pip install torch fastapi uvicorn numpy

# Run
python main.py
```

Open **http://localhost:8000** in your browser. The model trains on startup (~30s on CPU), then the UI is live. Drag the sliders and watch accuracy, memory, weight distributions, and per-class breakdown update in real time.

## What you see

- **Three sliders** control the three compression axes independently
- **Accuracy / Retained / Compression / Memory** metrics update live
- **Per-class accuracy bars** show which pattern types survive compression (arithmetic and fibonacci survive best — they're identifiable from 2-3 local points)
- **Sweep charts** show pre-computed accuracy curves along each axis in isolation
- **Weight distribution histogram** collapses from a smooth Gaussian to three spikes at ternary — the visual signature of quantization
- **Live predictions** with sparkline visualizations of each test sequence

## How the training works

The model uses a curriculum that builds compression robustness:

1. **Epochs 0–80**: Standard training (AdamW, cosine schedule). The model learns the task at full precision, full latent, full attention.
2. **Epochs 80–250**: Compression-aware training. Each batch randomly applies:
   - **Window masking** (40% of batches): attention restricted to a random window [3–12], forcing the model to learn locally-identifiable features
   - **Latent dropout** (40% of batches): random latent dimensions zeroed, forcing the model to distribute information across dimensions rather than depending on all of them
   - **Weight noise injection** (every 5 epochs): Gaussian noise proportional to weight magnitude, simulating the perturbation of post-training quantization

This curriculum is the honest version of the thesis: structure survives brutal cuts **when the system is designed for it**. A model trained at full precision and then naively compressed collapses. A model trained to expect compression learns representations that are naturally robust.

## The task

Classify 16-step sequences into 5 pattern types:

- **Arithmetic**: constant slope (identifiable from any 2 points)
- **Geometric**: constant ratio (identifiable from 2–3 points)
- **Periodic**: sinusoidal (needs ~half a period)
- **Fibonacci-like**: each value is the mean of the previous two (identifiable from 3 consecutive points)
- **Random**: no structure (unclassifiable by design — serves as a control)

The patterns are normalized and noised. Baseline accuracy is ~90% at full precision.

## Research directions

**1. Ternary-quantized VLA action accuracy.** The experiment this thesis most wants to see. Text quality survives 1.58-bit quantization (BitNet b1.58, Bonsai-8B). Does action accuracy survive when applied to vision-language-action models like OpenVLA (Brohan et al., 2024) or π0 (Black et al., 2024)? This is the first real composability test at the quantization × embodiment intersection.

**2. Explicit three-axis composition architectures.** A 1.58-bit Mamba-backbone JEPA encoder for edge-deployed robot policies. Each axis has been demonstrated independently; no one has published the triple composition. The composability matrix has three empty slots (ternary+SSM, ternary+JEPA, all three). When one of these is published, it should not be a surprise.

**3. Where does small-state lose?** DIAMOND (a Transformer-based, generative world model) tops the Atari100k leaderboard at 145.9% average human-normalized score, beating DreamerV3's RSSM at 112.4%. Full attention wins specific benchmarks. A clean characterization of the regime boundary — which task properties favor full attention over small state? — would make the thesis predictive rather than descriptive.

**4. Unified compression metric.** Intelligence density (capability per GB) captures the bit-width axis but not latent dimensionality or sequence state. A score that ranks architectures across all three cuts simultaneously does not exist. This demo's "total compression × retained accuracy" is a rough first attempt.

**5. Training-time composition.** This demo uses a curriculum (normal → compression-aware). Is there a single training objective that naturally produces representations robust to all three axes without staged curriculum? Quantization-aware training (QAT) handles axis 1. Dropout handles axis 2. What handles axis 3, and can they be unified?

**6. Sub-1-bit representations.** Grouping, mixture-of-experts at the bit-width level, or learned sparsification could push effective bit-widths below 1.0 without quality collapse. The intelligence-density metric anticipates the regime; the artifact does not yet exist.

## Literature

- Ma, S. et al. (2024). *The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits.* arXiv:2402.17764. — The training-aware ternary recipe that makes axis 1 work.
- Bardes, A. et al. (2025). *V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction and Planning.* arXiv:2506.09985. — JEPA at scale; the latent-dimensionality axis in production.
- Gu, A. & Dao, T. (2023). *Mamba: Linear-Time Sequence Modeling with Selective State Spaces.* arXiv:2312.00752. — The selective SSM that makes axis 3 work.
- Hafner, D. et al. (2024). *Mastering Diverse Domains through World Models.* Nature. — DreamerV3; RSSM as the bridging architecture between latent prediction and recurrent state.
- LeCun, Y. (2022). *A Path Towards Autonomous Machine Intelligence.* — The JEPA manifesto; the anti-generative argument that motivates axis 2.
- Brohan, A. et al. (2024). *OpenVLA: An Open-Source Vision-Language-Action Model.* — 7B VLA; the first target for ternary-quantized action accuracy experiments.
- Black, K. et al. (2024). *π0: A Vision-Language-Action Flow Model for General Robot Control.* — 3.3B VLA with flow matching; the other camp in the VLA split.
- Wang, Z. et al. (2024). *Is Sora a World Simulator? A Comprehensive Survey on General World Models and Beyond.* arXiv:2405.03520. — The survey that distinguishes video generation from world modeling.

## Architecture

```
steam-engine/
├── main.py            # FastAPI server, serves API + frontend
├── engine.py          # PyTorch model, data generation, compression engine
├── requirements.txt   # Python dependencies
└── frontend/
    ├── index.html     # Single-page app
    ├── style.css      # Dark research aesthetic
    └── app.js         # Sliders, Chart.js visualizations, live updates
```

## License

MIT
