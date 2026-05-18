"""
Steam Engine — Small-State Thesis Demo
FastAPI server: trains synthetic engine on startup, lazy-trains text engine
on first request. Serves compression API + frontend.
"""

import os
import threading
from contextlib import asynccontextmanager
from typing import List, Literal, Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from engine import Engine
from text_engine import TextEngine

engine = Engine()
text_engine = TextEngine()
text_engine_lock = threading.Lock()
text_engine_training = False
text_engine_started_at: Optional[float] = None

FRONTEND_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "frontend")


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Steam Engine starting...")
    thread = threading.Thread(target=engine.train, daemon=True)
    thread.start()
    yield


app = FastAPI(title="Steam Engine", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


class EvalRequest(BaseModel):
    bits: float = 32
    latent_ratio: float = 1.0
    state_ratio: float = 1.0
    mode: Literal["classify", "predict", "both"] = "both"


class PredictRequest(BaseModel):
    sequence: List[float]
    bits: float = 32
    latent_ratio: float = 1.0
    state_ratio: float = 1.0


class TextEvalRequest(BaseModel):
    bits: float = 32
    latent_ratio: float = 1.0
    state_ratio: float = 1.0


class TextGenerateRequest(BaseModel):
    prompt: str = "ROMEO:\n"
    max_new: int = 200
    temperature: float = 0.8
    bits: float = 32
    latent_ratio: float = 1.0
    state_ratio: float = 1.0


@app.get("/api/status")
def status():
    return {
        "ready": engine.ready,
        "baseline": engine.baseline,
        "baseline_pred_mse": engine.baseline_pred_mse,
        "baseline_latent_mse": engine.baseline_latent_mse,
        "params": engine.n_params,
    }


@app.post("/api/evaluate")
def evaluate(req: EvalRequest):
    if not engine.ready:
        return {"error": "Model still training"}
    return engine.evaluate(req.bits, req.latent_ratio, req.state_ratio, mode=req.mode)


@app.post("/api/predict")
def predict(req: PredictRequest):
    if not engine.ready:
        return {"error": "Model still training"}
    return engine.predict(req.sequence, req.bits, req.latent_ratio, req.state_ratio)


@app.get("/api/sweeps")
def sweeps():
    if not engine.ready:
        return {"error": "Model still training"}
    return {
        "classify": engine.sweeps,
        "predict": engine.pred_sweeps,
        "latent": engine.latent_sweeps,
    }


# -----------------------------------------------------------------------------
# Text engine — lazy-trained on first request to /api/text/start_training.

def _train_text_engine_thread():
    global text_engine_training, text_engine_started_at
    import time
    text_engine_started_at = time.time()
    try:
        text_engine.train(epochs=200, model_seed=0, aug_seed=0, beta=0.5, multi_corpus=False, corpus="code")
    finally:
        with text_engine_lock:
            text_engine_training = False


@app.post("/api/text/start_training")
def text_start_training():
    """Idempotent. Kicks off text engine training in a background thread if
    not already running and not already trained. Returns immediately."""
    global text_engine_training
    with text_engine_lock:
        if text_engine.ready:
            return {"status": "ready", "message": "Already trained"}
        if text_engine_training:
            return {"status": "training", "message": "Training already in progress"}
        text_engine_training = True
    thread = threading.Thread(target=_train_text_engine_thread, daemon=True)
    thread.start()
    return {"status": "started", "message": "Training started; ETA ~8 min"}


@app.get("/api/text/status")
def text_status():
    import time
    elapsed = (time.time() - text_engine_started_at) if text_engine_started_at else None
    base = {
        "ready": text_engine.ready,
        "training": text_engine_training,
        "elapsed_s": round(elapsed, 1) if elapsed else None,
        "params": text_engine.n_params,
    }
    if text_engine.ready:
        base.update({
            "baseline_perplexity": round(text_engine.baseline_perplexity, 4),
            "baseline_latent_mse": round(text_engine.baseline_latent_mse, 4),
            "vocab_size": text_engine.tokenizer.vocab_size if text_engine.tokenizer else None,
            "multi_corpus": text_engine.multi_corpus,
            "corpora": list(text_engine.baseline_per_corpus.keys()) if text_engine.baseline_per_corpus else [],
            "baseline_per_corpus": {
                k: {"perplexity": round(v["perplexity"], 4), "latent_mse": round(v["latent_mse"], 4)}
                for k, v in (text_engine.baseline_per_corpus or {}).items()
            },
        })
    return base


@app.post("/api/text/evaluate")
def text_evaluate(req: TextEvalRequest):
    if not text_engine.ready:
        return {"error": "Text model not trained. POST /api/text/start_training to begin."}
    return text_engine.evaluate(req.bits, req.latent_ratio, req.state_ratio)


@app.get("/api/text/sweeps")
def text_sweeps():
    if not text_engine.ready:
        return {"error": "Text model not trained"}
    return {
        "raw": text_engine.sweeps,
        "latent": text_engine.latent_sweeps,
    }


@app.post("/api/text/generate")
def text_generate(req: TextGenerateRequest):
    if not text_engine.ready:
        return {"error": "Text model not trained"}
    text = text_engine.generate(
        req.prompt, max_new=req.max_new, temperature=req.temperature,
        bits=req.bits, latent_ratio=req.latent_ratio, state_ratio=req.state_ratio,
    )
    return {"text": text, "prompt": req.prompt}


@app.get("/")
def index():
    return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))


app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
