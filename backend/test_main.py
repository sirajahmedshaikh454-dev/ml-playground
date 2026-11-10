"""Run: .venv/Scripts/python -m pytest -q  (or just python test_main.py)"""
import json

import numpy as np
from fastapi.testclient import TestClient

import main
from main import app

c = TestClient(app)
rng = np.random.default_rng(0)
X = rng.normal(size=(200, 3))
yc = (X[:, 0] + X[:, 1] > 0).astype(int)
yr = X @ np.array([2.0, -1.0, 0.5]) + rng.normal(scale=0.1, size=200)


def run(task, model, y, params=None):
    body = {"task": task, "model": model, "params": params or {}, "k": 2,
            "Xtr": X[:150].tolist(), "ytr": y[:150].tolist(), "Xva": X[150:].tolist(), "yva": y[150:].tolist()}
    with c.stream("POST", "/train", json=body) as r:
        assert r.status_code == 200, r.read()
        evs = [json.loads(line[6:]) for line in r.iter_lines() if line.startswith("data: ")]
    assert evs[-1]["type"] == "done", evs[-1]
    return evs


def test_plumbing_with_stub_model():
    """SSE stream, model store, /predict and cancellation-free happy path, no sklearn needed."""
    class Stub:
        def predict(self, X): return np.zeros(len(X))
        def predict_proba(self, X): return np.full((len(X), 2), 0.5)

    def fake_train(r, Xtr, ytr, Xva, yva, emit, stop):
        for i in range(1, 4):
            emit(i, 3, None, {"acc": 0.5 + i / 10, "val_acc": 0.5})
        return Stub()

    saved = main.SKLEARN_ERROR, main._train
    main.SKLEARN_ERROR, main._train = None, fake_train
    try:
        evs = run("classification", "svm", yc)
        assert [e["step"] for e in evs[:-1]] == [1, 2, 3] and evs[-1]["type"] == "done"
        p = c.post("/predict", json={"id": evs[-1]["id"], "X": X[:4].tolist()}).json()
        assert p["pred"] == [0, 0, 0, 0] and len(p["proba"]) == 4
    finally:
        main.SKLEARN_ERROR, main._train = saved


def test_health_reports_availability():
    h = c.get("/health").json()
    assert h["ok"] and (h["models"] == main.MODELS) == (main.SKLEARN_ERROR is None)


def test_all_models():
    if main.SKLEARN_ERROR:
        print("skip real models:", main.SKLEARN_ERROR[:80])
        return
    for model in ["svm", "gboost", "forest", "mlp"]:
        evs = run("classification", model, yc, {"epochs": 20, "trees": 30})
        assert evs[-2]["metrics"]["val_acc"] > 0.8, (model, evs[-2])
        p = c.post("/predict", json={"id": evs[-1]["id"], "X": X[:5].tolist()}).json()
        assert len(p["pred"]) == 5
        evs = run("regression", model, yr, {"epochs": 50, "trees": 30})
        assert evs[-2]["metrics"]["val_r2"] > 0.7, (model, evs[-2])


def test_rejects_bad_input():
    bad = {"task": "classification", "model": "svm", "Xtr": [[1, None]], "ytr": [0], "Xva": [[1, 2]], "yva": [0]}
    saved = main.SKLEARN_ERROR
    main.SKLEARN_ERROR = None  # exercise validation even where sklearn cannot load
    try:
        assert c.post("/train", json=bad).status_code == 422
        assert c.post("/train", json={**bad, "model": "rm -rf"}).status_code == 422
    finally:
        main.SKLEARN_ERROR = saved
    assert c.post("/predict", json={"id": "nope", "X": [[1]]}).status_code == 404


if __name__ == "__main__":
    test_plumbing_with_stub_model()
    test_health_reports_availability()
    test_all_models()
    test_rejects_bad_input()
    print("ok")
