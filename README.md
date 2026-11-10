# ML Playground

Upload any dataset, clean it, and **watch machine-learning models learn, live, in your browser**: supervised, unsupervised and reinforcement learning, with an optional Python backend for heavier models.

Your data never leaves your machine. Everything runs client-side unless you start the local backend yourself.

## What's inside

| Area | What you get |
|---|---|
| **Data Lab** `/lab` | Six-step pipeline: **Data** (CSV, TSV, JSON, Excel, plain text, image folders, 3 samples) → **Clean** (missing values, outliers, duplicates, type fixes, one-click auto-clean, undoable recipe) → **Explore** (distributions, correlation heatmap, 3D PCA cloud) → **Features** (target, one-hot/label encoding, scaling, text bag-of-words, seeded and stratified split) → **Train** → **Evaluate** |
| **Models (browser)** | Linear & Logistic Regression, Neural Network (TensorFlow.js), Decision Tree (grown by depth), Random Forest (grown by tree), kNN, Gaussian Naive Bayes · k-Means, DBSCAN, PCA, Anomaly Detection |
| **Models (Python, optional)** | SVM (RBF), Gradient Boosting, large Random Forest, scikit-learn MLP. They appear automatically when the backend is running. |
| **Live training** | Play / pause / step / stop at adjustable speed; train vs validation curves; decision boundary or regression fit; clusters in 2D and 3D; overfitting warning |
| **Evaluation** | Test-set accuracy, macro F1, majority baseline, confusion matrix, ROC/AUC (one-vs-rest), R², RMSE, MAE, predicted-vs-actual, residuals |
| **RL Arena** `/rl` | Paintable gridworld (walls, traps, goal, start) with Q-learning vs SARSA, tunable reward, penalty, step cost and slippery floor, live Q-value heatmap and animated policy replay. CartPole with learning curve, learned vs random policy. |
| **Experiments** `/compare` | Every run auto-saved (IndexedDB): leaderboard, learning-curve overlay, split-mismatch warning, JSON export/import |

**Honest numbers by design:** the split happens before any fitting, scalers, encoders and vocabularies are fitted on training rows only, live curves use a validation slice of the training set, and the test set is touched once, when a run finishes.

## Quick start

Requires Node 20+.

```bash
cd app
npm install
npm run dev
```

Open http://localhost:3000.

### Optional: Python backend (heavier models)

Requires Python 3.11+.

```bash
cd backend
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt      # Windows
# .venv/bin/python -m pip install -r requirements.txt        # macOS / Linux
.venv/Scripts/python -m uvicorn main:app --host 127.0.0.1 --port 8000
```

Reload the Train step and the Python models show up with a green badge. To use another address, set `NEXT_PUBLIC_ML_BACKEND` (default `http://127.0.0.1:8000`).

The backend binds to localhost, accepts requests only from localhost origins, validates every array (finite values, size cap), keeps models in memory only, and never unpickles anything.

> **Windows note:** if `/health` returns `models: []` with "blocked by an Application Control policy", Windows Smart App Control has blocked scikit-learn's native DLLs. It may clear on a second start once the files have been checked. If not, run the backend under WSL or on another machine.

## Checks

```bash
cd app
npm run smoke      # every engine on every sample + RL agents must learn (CartPole 500/500)
npx tsc --noEmit
npx eslint src
npm run build
```

```bash
cd backend
.venv/Scripts/python test_main.py   # SSE plumbing, all 4 sklearn models x 2 tasks, input validation
```

## How it fits together

```
app/src/lib/
  ingest.ts     file -> typed Dataset (CSV/JSON/Excel/text/images)
  clean.ts      serializable CleanOp list, profile, histograms, correlation
  features.ts   leak-free prepare(): split -> fit on train -> encode/scale
  ml.ts         Engine = async generator of Progress events (11 models)
  remote.ts     same Engine interface over SSE to the Python backend
  rl.ts         gridworld + CartPole environments and agents
  metrics.ts    classification / regression reports, ROC
  useRunner.ts  play/pause/step/stop any generator with a frame budget
  store.ts      zustand lab state + Dexie (IndexedDB) experiment store
backend/main.py FastAPI: /health, /train (SSE), /predict
```

Every model, in the browser or in Python, streams the same `Progress` events, so the UI controls, charts and run history work identically for all of them.

## Known limits

- In-browser data is capped at 100k rows and 60 MB per file. Images are downsampled to 16×16 grayscale.
- DBSCAN, anomaly detection and silhouette are O(n²): clustering samples up to 4,000 rows, and kNN uses up to 3,000 training rows.
- Training runs on the main thread in time slices, not yet in a Web Worker.
- CartPole uses tabular Q-learning on a discretized state. There is no deep Q-network yet.
- No accounts and no cloud sync, by design: runs live in this browser, so export them to move them.
