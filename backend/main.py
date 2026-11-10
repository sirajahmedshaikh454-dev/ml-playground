"""Optional heavy-model backend for ML Playground.

Run:  .venv/Scripts/python -m uvicorn main:app --host 127.0.0.1 --port 8000
Binds to localhost only; data is sent from your own browser to your own machine.
"""
import asyncio
import json
import math
import threading
import uuid
from collections import OrderedDict
from queue import Empty, Queue

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

try:  # lazy-safe: some Windows Application Control policies block sklearn's DLLs
    from sklearn.ensemble import (GradientBoostingClassifier, GradientBoostingRegressor,
                                  RandomForestClassifier, RandomForestRegressor)
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.metrics import accuracy_score, log_loss, mean_squared_error, r2_score
    from sklearn.neural_network import MLPClassifier, MLPRegressor
    from sklearn.svm import SVC, SVR
    SKLEARN_ERROR = None
except ImportError as e:
    SKLEARN_ERROR = f"scikit-learn could not load: {e}"

MAX_CELLS = 5_000_000  # rows * features cap per request
MODELS = ["svm", "gboost", "forest", "mlp"]

app = FastAPI(title="ML Playground backend")
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"https?://(localhost|127\.0\.0\.1)(:\d+)?",  # local front-end only
    allow_methods=["GET", "POST"],
    allow_headers=["content-type"],
)

STORE: "OrderedDict[str, object]" = OrderedDict()  # trained models, in memory only, LRU


class TrainReq(BaseModel):
    task: str = Field(pattern="^(classification|regression)$")
    model: str
    params: dict[str, float] = {}
    Xtr: list[list[float]]
    ytr: list[float]
    Xva: list[list[float]]
    yva: list[float]
    k: int = 0


class PredictReq(BaseModel):
    id: str
    X: list[list[float]]


def _check(X, y=None):
    X = np.asarray(X, dtype=float)
    if X.ndim != 2 or X.size == 0 or X.size > MAX_CELLS or not np.isfinite(X).all():
        raise HTTPException(422, "X must be a finite, non-empty 2D array within size limits")
    if y is not None and len(y) != len(X):
        raise HTTPException(422, "X and y lengths differ")
    return X


def _metrics(task, m, Xtr, ytr, Xva, yva, k):
    if task == "regression":
        pt, pv = m.predict(Xtr), m.predict(Xva)
        return {"loss": math.sqrt(mean_squared_error(ytr, pt)), "val_loss": math.sqrt(mean_squared_error(yva, pv)),
                "r2": r2_score(ytr, pt), "val_r2": r2_score(yva, pv)}
    out = {"acc": accuracy_score(ytr, m.predict(Xtr)), "val_acc": accuracy_score(yva, m.predict(Xva))}
    if hasattr(m, "predict_proba"):
        labels = list(range(k))
        out["loss"] = log_loss(ytr, m.predict_proba(Xtr), labels=labels)
        out["val_loss"] = log_loss(yva, m.predict_proba(Xva), labels=labels)
    return out


def _train(r: TrainReq, Xtr, ytr, Xva, yva, emit, stop: threading.Event):
    cls, p, k = r.task == "classification", r.params, r.k
    if r.model == "svm":
        C, gamma = p.get("C", 1.0), p.get("gamma", 0.1)
        m = CalibratedClassifierCV(SVC(C=C, gamma=gamma), ensemble=False) if cls else SVR(C=C, gamma=gamma)
        m.fit(Xtr, ytr)
        emit(1, 1, m)
    elif r.model == "gboost":
        n = int(p.get("trees", 100))
        kw = dict(n_estimators=n, learning_rate=p.get("lr", 0.1), max_depth=int(p.get("depth", 3)), random_state=0)
        m = (GradientBoostingClassifier if cls else GradientBoostingRegressor)(**kw).fit(Xtr, ytr)
        # replay the boosting stages so the curve shows learning over trees
        every = max(1, n // 60)
        stages_t, stages_v = list(m.staged_predict(Xtr)), list(m.staged_predict(Xva))
        for i in range(every - 1, n, every):
            if stop.is_set():
                return m
            met = ({"acc": accuracy_score(ytr, stages_t[i]), "val_acc": accuracy_score(yva, stages_v[i])} if cls else
                   {"loss": math.sqrt(mean_squared_error(ytr, stages_t[i])), "val_loss": math.sqrt(mean_squared_error(yva, stages_v[i])),
                    "r2": r2_score(ytr, stages_t[i]), "val_r2": r2_score(yva, stages_v[i])})
            emit(i + 1, n, m if i + every >= n else None, met)
    elif r.model == "forest":
        n = int(p.get("trees", 200))
        m = (RandomForestClassifier if cls else RandomForestRegressor)(n_estimators=0, warm_start=True, max_depth=int(p.get("depth", 0)) or None, n_jobs=-1, random_state=0)
        step = max(1, n // 25)
        for t in range(step, n + step, step):
            if stop.is_set():
                return m
            m.n_estimators = min(t, n)
            m.fit(Xtr, ytr)
            emit(m.n_estimators, n, m)
    elif r.model == "mlp":
        epochs = int(p.get("epochs", 80))
        hidden = (int(p.get("hidden", 64)),) * int(p.get("layers", 2))
        kw = dict(hidden_layer_sizes=hidden, learning_rate_init=p.get("lr", 0.001), alpha=p.get("l2", 1e-4), random_state=0)
        m = MLPClassifier(**kw) if cls else MLPRegressor(**kw)
        for e in range(1, epochs + 1):
            if stop.is_set():
                return m
            m.partial_fit(Xtr, ytr, classes=list(range(k))) if cls else m.partial_fit(Xtr, ytr)
            emit(e, epochs, m)
    else:
        raise ValueError(f"unknown model {r.model}")
    return m


@app.get("/health")
def health():
    return {"ok": True, "models": [] if SKLEARN_ERROR else MODELS, "error": SKLEARN_ERROR}


@app.post("/train")
async def train(r: TrainReq, request: Request):
    if SKLEARN_ERROR:
        raise HTTPException(503, SKLEARN_ERROR)
    if r.model not in MODELS:
        raise HTTPException(422, f"model must be one of {MODELS}")
    Xtr, Xva = _check(r.Xtr, r.ytr), _check(r.Xva, r.yva)
    ytr, yva = np.asarray(r.ytr), np.asarray(r.yva)
    if r.task == "classification":
        ytr, yva = ytr.astype(int), yva.astype(int)
    rid, q, stop = uuid.uuid4().hex, Queue(), threading.Event()

    def emit(step, total, model, met=None):
        q.put({"type": "progress", "id": rid, "step": step, "total": total,
               "metrics": met if met is not None else _metrics(r.task, model, Xtr, ytr, Xva, yva, r.k)})

    def work():  # sklearn is CPU-bound: keep it off the event loop
        try:
            m = _train(r, Xtr, ytr, Xva, yva, emit, stop)
            STORE[rid] = m
            while len(STORE) > 20:
                STORE.popitem(last=False)
            q.put({"type": "done", "id": rid})
        except Exception as e:  # report instead of dying silently
            q.put({"type": "error", "message": str(e)})

    threading.Thread(target=work, daemon=True).start()

    async def stream():
        while True:
            if await request.is_disconnected():
                stop.set()
                return
            try:
                ev = await asyncio.to_thread(q.get, True, 0.5)
            except Empty:
                continue
            yield f"data: {json.dumps(ev)}\n\n"
            if ev["type"] in ("done", "error"):
                return

    return StreamingResponse(stream(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})


@app.post("/predict")
def predict(r: PredictReq):
    m = STORE.get(r.id)
    if m is None:
        raise HTTPException(404, "model not found (backend restarted or evicted)")
    X = _check(r.X)
    return {"pred": m.predict(X).tolist(), "proba": m.predict_proba(X).tolist() if hasattr(m, "predict_proba") else None}
