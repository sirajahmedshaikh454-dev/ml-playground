# ML Playground

## What This Is

A full Next.js website where anyone uploads their own data (CSV, JSON, Excel, images, text), walks it through every data-science step (cleaning, EDA, feature engineering, train/test split), then runs any kind of model on it and watches the training and testing happen live. It covers supervised, unsupervised and reinforcement learning (reward/punishment) in one place, with rich 2D/3D visualization and model comparison.

## Core Value

A user can upload any dataset and see, end to end, how a model learns from it: clean data in, live training visualization, honest test results out.

## Requirements

### Validated

(None yet — ship to validate)

### Active

- [ ] Upload data in any form: CSV, JSON, Excel, images, plain text; preview and schema detection
- [ ] Guided data cleaning: missing values, outliers, dtype fixes, dedupe, with before/after view
- [ ] Automatic EDA: distributions, correlations, class balance, summary stats
- [ ] Feature engineering: encoding, scaling, selection, text/image featurization; train/test split controls
- [ ] Supervised models: regression and classification (linear/logistic, trees, forests, kNN, SVM, neural net)
- [ ] Unsupervised models: clustering (k-means, DBSCAN, hierarchical), PCA/t-SNE/UMAP, anomaly detection
- [ ] Reinforcement learning: Q-learning/SARSA/DQN on gridworld and CartPole with reward/penalty visualization
- [ ] Live training visualization: loss/accuracy curves, decision boundaries, cluster plots, RL reward curves, agent animation
- [ ] Testing/evaluation: confusion matrix, ROC/PR, residuals, metrics per task type
- [ ] Model comparison across runs and algorithms
- [ ] 3D visualizations (Three.js / react-three-fiber) and Framer Motion animation
- [ ] Browser-first training (TensorFlow.js/ONNX) with optional Python FastAPI backend for heavy models and large files
- [ ] Experiments saved locally in the browser and exportable (no accounts)
- [ ] Modern, production-quality UI built from a ui-ux-pro-max design system; responsive and accessible

### Out of Scope

- User accounts and cloud storage — v1 is local-only; keeps data on the user's machine
- Hosted GPU training / distributed training — heavy models run on a user-run FastAPI backend only
- Model serving / deployment APIs — this is a learning and experimentation tool, not MLOps
- Mobile app — web-first, responsive only

## Context

- Greenfield project in `ML-Playground` on the user's Desktop.
- Audience is learners, students and analysts who want to see ML working, not just get numbers.
- Heavy models (large data, deep nets, big ensembles) are impractical in a browser, hence the optional FastAPI backend (scikit-learn, PyTorch).
- Reinforcement learning needs environments (gridworld, CartPole) simulated in the client for live animation.

## Constraints

- **Tech stack**: Next.js (App Router) + TypeScript, Tailwind — user-specified
- **Visualization**: Three.js / react-three-fiber, Framer Motion, Recharts/D3 — user-specified
- **ML runtime**: TensorFlow.js/ONNX in browser plus optional Python FastAPI backend — browser-first for zero-setup, backend for heavy work
- **Privacy**: Data stays local by default; backend is opt-in and run by the user
- **Performance**: Training must not freeze the UI — use Web Workers and streamed progress
- **Design**: ui-ux-pro-max design system for all UI

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Browser-first + optional FastAPI | Zero setup for common cases, power when needed | — Pending |
| Full scope in v1, delivered in phases | User wants all paradigms; phasing manages risk | — Pending |
| No accounts, local save (IndexedDB + export) | Keeps scope and privacy simple | — Pending |

## Evolution

This document evolves at phase transitions and milestone boundaries.

**After each phase transition** (via `/gsd-transition`):
1. Requirements invalidated? → Move to Out of Scope with reason
2. Requirements validated? → Move to Validated with phase reference
3. New requirements emerged? → Add to Active
4. Decisions to log? → Add to Key Decisions
5. "What This Is" still accurate? → Update if drifted

**After each milestone** (via `/gsd-complete-milestone`):
1. Full review of all sections
2. Core Value check — still the right priority?
3. Audit Out of Scope — reasons still valid?
4. Update Context with current state

---
*Last updated: 2026-11-09 after initialization*
