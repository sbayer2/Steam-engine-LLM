"""
Steam Engine — Small-State Thesis Demo
FastAPI server: trains model on startup, serves compression API + frontend.
"""

import os
import threading
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from engine import Engine

engine = Engine()

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


@app.get("/api/status")
def status():
    return {"ready": engine.ready, "baseline": engine.baseline, "params": engine.n_params}


@app.post("/api/evaluate")
def evaluate(req: EvalRequest):
    if not engine.ready:
        return {"error": "Model still training"}
    return engine.evaluate(req.bits, req.latent_ratio, req.state_ratio)


@app.get("/api/sweeps")
def sweeps():
    if not engine.ready:
        return {"error": "Model still training"}
    return engine.sweeps


@app.get("/")
def index():
    return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))


app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
