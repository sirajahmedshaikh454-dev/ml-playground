# Architecture Research

**Domain:** Interactive, local-first ML playground (Next.js + in-browser ML in Web Workers + optional FastAPI backend)
**Researched:** 2026-11-09
**Confidence:** MEDIUM-HIGH (structure/patterns HIGH, drawn from established browser-ML playground practice and platform behavior; three version-sensitive items flagged MEDIUM/LOW in "Open Verification Items")

## Standard Architecture

ML playgrounds (TensorFlow Playground, Teachable Machine, sklearn-in-Pyodide demos, Weights-and-Biases-style local dashboards) converge on the same shape: a **thin UI over a framework-free domain core, with all compute isolated behind a message boundary (worker or network) that emits a stream of typed events**. The UI never calls model code directly; it subscribes to events. That one rule is what makes browser training and FastAPI training interchangeable.

Two consequences specific to this project:

1. **The "upload -> clean -> features -> train" flow is not universal.** RL has no dataset. So the top-level concept is a **Session** of one of four kinds (`tabular`, `image`, `text`, `rl`), where RL sessions enter at "pick environment" and skip the data pipeline entirely. Do not force RL through the dataset store.
2. **Next.js is only a shell here.** Everything interesting is client-side. Use client components, no server actions or API routes for data, and prefer `output: 'export'` (static) so the "your data never leaves your machine" promise is structurally true, not just a policy.

### System Overview

```
┌──────────────────────────────────────────────────────────────────────────┐
│                       PRESENTATION  (main thread, React)                  │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌───────────┐ ┌──────────────┐   │
│  │ Upload & │ │ Clean/EDA│ │ Features │ │ Train Lab │ │ Compare /    │   │
│  │ Preview  │ │ Wizard   │ │ & Split  │ │ (live viz)│ │ Experiments  │   │
│  └────┬─────┘ └────┬─────┘ └────┬─────┘ └─────┬─────┘ └──────┬───────┘   │
│       │            │            │             │              │           │
│  ┌────┴────────────┴────────────┴─────────────┴──────────────┴───────┐   │
│  │ Viz layer: Recharts (low-freq) | Canvas/D3 (high-freq) | r3f (3D)  │   │
│  │ consumes ViewModels from stores, never raw models/tensors          │   │
│  └────────────────────────────────┬───────────────────────────────────┘   │
├───────────────────────────────────┼──────────────────────────────────────┤
│                  APPLICATION  (main thread, TS, no React)                 │
│  ┌──────────────┐  ┌───────────────────┐  ┌───────────────────────────┐   │
│  │ Session store│  │ Run Orchestrator  │  │ Live Run Store            │   │
│  │ (Zustand)    │  │ start/pause/stop, │  │ ring buffers + rAF-batched│   │
│  │              │  │ picks Executor    │  │ subscriptions for charts  │   │
│  └──────┬───────┘  └─────────┬─────────┘  └───────────▲───────────────┘   │
│         │                    │  TrainEvent stream     │                   │
├─────────┼────────────────────┼────────────────────────┼──────────────────┤
│                 EXECUTION BOUNDARY (message-passing only)                 │
│  ┌──────┴──────────┐  ┌──────┴──────────────────┐  ┌──┴────────────────┐ │
│  │ Data Worker     │  │ Compute Worker (per run)│  │ RemoteExecutor    │ │
│  │ parse, profile, │  │ Engine (supervised /    │  │ fetch + SSE client│ │
│  │ pipeline fit/   │  │ unsupervised / RL) +    │  │ -> same TrainEvent│ │
│  │ transform, EDA  │  │ Environment (RL)        │  │ stream            │ │
│  └──────┬──────────┘  └──────┬──────────────────┘  └──┬────────────────┘ │
├─────────┼────────────────────┼────────────────────────┼──────────────────┤
│                       DOMAIN CORE (pure TS, shared by workers+UI)         │
│  Dataset (columnar) | PipelineSpec | Engine/Environment interfaces |      │
│  Metrics (single implementation) | Model Registry | Event schema (zod)    │
├─────────┼────────────────────┼────────────────────────┼──────────────────┤
│                           PERSISTENCE (local-only)                        │
│  ┌──────────────┐  ┌─────────────────┐  ┌──────────────┐ ┌────────────┐  │
│  │ IndexedDB    │  │ OPFS / IDB blobs│  │ Export/Import│ │ localStorage│  │
│  │ (Dexie): meta│  │ raw files, col  │  │ zip + manifest│ │ prefs only │  │
│  │ runs, metrics│  │ buffers, weights│  │ (schema ver.) │ │            │  │
│  └──────────────┘  └─────────────────┘  └──────────────┘ └────────────┘  │
└──────────────────────────────────────────────────────────────────────────┘
                                   │ (opt-in, user-run)
                         ┌─────────▼──────────┐
                         │ FastAPI backend    │  sklearn / PyTorch
                         │ POST /runs         │  emits identical TrainEvents
                         │ GET /runs/{id}/events (SSE)                    │
                         │ POST /runs/{id}/cancel                         │
                         └────────────────────┘
```

### Component Responsibilities

| Component | Responsibility | Typical Implementation |
|-----------|----------------|------------------------|
| Ingest / Parsers | Turn File -> `RawTable` or `ImageSet` or `TextCorpus`; never block UI | PapaParse (worker mode, chunked) for CSV; SheetJS for xlsx; `JSON.parse` (stream for large) ; images via `createImageBitmap` + `OffscreenCanvas` |
| Schema Inference | Detect dtype per column (num/int/cat/bool/datetime/text), missing markers, cardinality, target candidates | Pure function over a sample + full-pass stats; user-overridable, result stored as `Schema` |
| Dataset Store | Own the canonical in-memory + persisted representation | Columnar typed arrays (`Float64Array` per numeric col, dictionary-encoded `Uint32Array` + `string[]` per categorical) held in Data Worker; main thread holds only a preview slice + stats |
| Preprocessing Pipeline | Reproducible, ordered, serializable steps (impute, dedupe, outlier, dtype cast, encode, scale, select, featurize, split) | `PipelineSpec` = JSON list of `{id, op, params, columns}`; each op has `fit(train)` -> `state` and `transform(data, state)`; interpreter lives in Data Worker |
| EDA Engine | Distributions, correlations, class balance, summary stats | Pure functions over columnar data run in Data Worker; returns small chart-ready summaries (histogram bins, corr matrix), never raw columns |
| Model Registry | Declarative catalog of algorithms: paradigm, task types, param schema, capabilities, lazy engine factory, where it can run | Static TS map; param forms auto-generated from zod schema; `import()` of heavy libs happens inside the worker |
| Engine abstraction | Uniform "train and tell me what happens" contract across supervised / unsupervised / RL | `Engine` interface returning `AsyncIterable<TrainEvent>` + `predict` / `snapshot` (see Pattern 1) |
| Environment (RL) | Simulate gridworld / CartPole deterministically with seed; expose render state | `Environment` interface (`reset/step/renderState`) in domain core; runs inside the compute worker next to the agent |
| Run Orchestrator | Lifecycle of a run: validate config, choose Executor (worker vs remote), start/pause/stop, persist on completion | Plain TS class; owns the Worker handle and `terminate()` for hard cancel |
| Executors | Transport adapters that yield `TrainEvent`s | `WorkerExecutor` (Comlink + `MessageChannel`/callback), `RemoteExecutor` (fetch + EventSource/`fetch`-stream) |
| Live Run Store | Absorb high-frequency events without re-rendering React per step | External store (not React state): append to ring buffers, notify subscribers at most once per animation frame |
| Evaluation module | Metrics and curves from `(yTrue, yPred, proba)` per task type | **One** pure TS implementation, used for browser *and* backend runs (backend returns test-set predictions) |
| Viz layer | Render ViewModels: curves, boundaries, clusters, confusion matrix, RL reward + agent, 3D scatter | Recharts for dashboards (<= few thousand points, low update rate); raw Canvas2D/D3 for high-frequency (boundary heatmap, 100k-point scatter); r3f for 3D |
| Experiment Store | Persist datasets meta, pipelines, runs, metric histories, model artifacts; list/compare/delete; export/import | Dexie (IndexedDB) for structured records; OPFS (fallback IDB blobs) for large binaries; `fflate` zip + `manifest.json` with `schemaVersion` |
| Backend (optional) | Run heavy sklearn/PyTorch jobs, stream progress | FastAPI + Pydantic; background task per run; SSE for events; REST for control |

## Recommended Project Structure

```
src/
├── app/                         # Next.js App Router (thin: routes + layouts only)
│   ├── (workspace)/
│   │   ├── data/                # upload, preview, clean, EDA, features, split
│   │   ├── train/               # model picker + live training lab
│   │   ├── rl/                  # environment picker + RL lab (no dataset)
│   │   └── experiments/         # saved runs, comparison, export/import
│   └── layout.tsx               # design-system shell, providers
├── core/                        # PURE TS. No React, no DOM, no Next. Importable by workers.
│   ├── dataset/                 # columnar types, schema inference, hashing
│   ├── pipeline/                # PipelineSpec, ops/ (impute, scale, encode...), interpreter
│   ├── engine/                  # Engine + Environment interfaces, TrainEvent schema, registry
│   ├── metrics/                 # classification, regression, clustering, RL summaries
│   └── contracts/               # zod schemas shared with backend (generated types)
├── engines/                     # concrete implementations, each lazily imported
│   ├── supervised/              # linear, logistic, tree, forest, knn, svm, mlp (tfjs)
│   ├── unsupervised/            # kmeans, dbscan, hierarchical, pca, tsne, umap, anomaly
│   ├── rl/                      # agents (q-learning, sarsa, dqn) + envs (gridworld, cartpole)
│   └── remote/                  # RemoteEngine client for FastAPI
├── workers/                     # *.worker.ts entrypoints + Comlink wiring only
│   ├── data.worker.ts
│   └── compute.worker.ts
├── runtime/                     # main-thread orchestration (no React)
│   ├── orchestrator.ts
│   ├── executors/               # worker.ts, remote.ts
│   └── live-run-store.ts
├── storage/                     # Dexie schema, OPFS helpers, export/import, migrations
├── viz/                         # chart components + ViewModel adapters
│   ├── charts2d/  canvas/  three/
├── ui/                          # design-system components (ui-ux-pro-max tokens)
└── state/                       # Zustand slices (session, pipeline draft, UI prefs)
backend/                         # optional FastAPI project (separate deploy, same event schema)
└── app/{main.py, runs.py, engines/, schemas.py}
```

### Structure Rationale

- **`core/` has zero framework imports.** It is imported by both workers and UI, unit-testable in Node, and is the only place metrics and pipeline semantics are defined. This enforces the "single source of truth" rule.
- **`engines/` is separate from `core/engine/`** so interfaces stay tiny and stable while implementations (and their heavy dependencies like tfjs) are lazy-loaded only inside workers.
- **`workers/` contains only entrypoints.** Worker files are bundler-sensitive; keeping them trivial means bundler quirks never touch domain logic.
- **`runtime/` sits between UI and workers** so React components never see a `Worker` object. Swapping `WorkerExecutor` for `RemoteExecutor` is invisible above this line.
- **`app/` stays thin.** Route files compose feature components; no logic. RL gets its own route because its entry flow differs.

## Architectural Patterns

### Pattern 1: Engine as an event stream (the central abstraction)

**What:** Every training mechanism, regardless of paradigm or where it runs, is `Engine.run(config, input, signal) -> AsyncIterable<TrainEvent>`. Paradigm differences live in the event payloads, not the control flow.
**When to use:** Always; it is the seam that makes supervised/unsupervised/RL and browser/backend uniform.
**Trade-offs:** (+) one orchestrator, one live store, one persistence format, remote engines are "just another engine". (-) Event schema must be designed carefully up front; changing it later touches every engine and the backend.

**Example:**
```typescript
// core/engine/events.ts  (discriminated union, validated with zod at the backend boundary)
type TrainEvent =
  | { t: 'status'; phase: 'init' | 'training' | 'evaluating' | 'done' | 'cancelled' }
  | { t: 'progress'; step: number; total?: number; metrics: Record<string, number> } // loss, acc, reward...
  | { t: 'viz'; kind: 'boundary' | 'clusters' | 'projection' | 'rl-frame' | 'weights';
      payload: Float32Array | RlFrame | object }       // transferable where possible
  | { t: 'artifact'; kind: 'model' | 'predictions'; ref: string }
  | { t: 'error'; message: string; recoverable: boolean };

// core/engine/engine.ts
interface Engine<Cfg, In> {
  run(cfg: Cfg, input: In, ctx: { seed: number; signal: AbortSignal }): AsyncIterable<TrainEvent>;
  predict?(x: Float32Array, shape: number[]): Promise<Float32Array>; // for boundary grids, evaluation
  dispose(): void;                                                    // free tf.js memory etc.
}
// In = TabularSplit | ImageSplit | TextSplit for supervised/unsupervised; EnvSpec for RL.
```

### Pattern 2: Fit-on-train, transform-everywhere pipeline (reproducibility + leakage safety)

**What:** The preprocessing pipeline is a serializable `PipelineSpec` interpreted by one engine. The train/test **split is a pipeline step placed before any stateful step** (imputer means, scaler stats, encoder vocab, selected features are `fit` on train only and stored as `state`). Reproducibility = `datasetHash + PipelineSpec + seed`.
**When to use:** All tabular/text/image preprocessing.
**Trade-offs:** (+) the same spec re-applies to test data and to new predictions, can be exported, diffed, replayed, and shown before/after per step. (-) Every op needs both `fit` and `transform`; ops that look "stateless" (dedupe, outlier removal) must be explicitly classified as train-only or both.

**Example:**
```typescript
interface PipelineOp<P, S> {
  id: string;
  fit(train: Table, p: P): S;            // pure, deterministic given seed
  transform(t: Table, p: P, s: S): Table;
  appliesTo: 'train-only' | 'train-and-test'; // e.g. outlier removal is train-only
}
// Interpreter: split -> fit ops sequentially on train -> transform train & test -> emit step snapshots
```

### Pattern 3: Worker per run, hard-terminate to cancel

**What:** Spawn a fresh compute worker per training run; Stop = `worker.terminate()` (plus cooperative `AbortSignal` for clean state). Long-lived Data Worker holds the dataset; compute worker receives a **transferred** copy of the training buffers (`postMessage(buf, [buf])`) or re-reads from OPFS.
**When to use:** Any tight numeric loop (k-means, SVM, tfjs `fit`) that cannot yield reliably.
**Trade-offs:** (+) guaranteed cancel, no leaked tf.js memory across runs, a crash cannot corrupt the UI or the dataset. (-) startup cost per run (lazy `import()` of tfjs ~ hundreds of ms after first cached load), data must be copied/transferred.

**Example:**
```typescript
// runtime/executors/worker.ts
const w = new Worker(new URL('../../workers/compute.worker.ts', import.meta.url), { type: 'module' });
const api = Comlink.wrap<ComputeApi>(w);
await api.start(cfg, Comlink.transfer(input, input.buffers), Comlink.proxy(onEvents)); // batched events
signal.addEventListener('abort', () => w.terminate());
```

### Pattern 4: Batched event bridge, external live store

**What:** Worker coalesces events (e.g. every 50 ms or every N steps) into arrays and posts once; main-thread `LiveRunStore` appends into fixed-size ring buffers and notifies subscribers **at most once per `requestAnimationFrame`**. Charts read from the buffer (with decimation, e.g. LTTB, beyond ~2k points). React state is *not* touched per step.
**When to use:** All live curves, RL reward plots, boundary updates.
**Trade-offs:** (+) UI stays at 60 fps while 1000s of steps/s are produced; pause/speed control is trivial (just stop draining). (-) Slightly more plumbing than `useState`; must use `useSyncExternalStore`.

### Pattern 5: Declarative model registry driving UI and capability gating

**What:** Each algorithm is a registry entry: `{ id, paradigm, tasks, paramSchema (zod), defaults, vizCapabilities, runsOn: ('browser'|'backend')[], load: () => import(...) }`. Model picker, param forms, "this model needs the backend for >50k rows" hints, and which live-viz panels to show are all derived from it.
**When to use:** From the first engine onward; adding the 15th algorithm should be one file + one registry line.
**Trade-offs:** (+) tiny marginal cost per algorithm; browser/backend parity is explicit (same `id` on both sides). (-) Upfront schema work.

### Pattern 6: Contract-first backend (schema shared, transport thin)

**What:** The FastAPI Pydantic models are the source of truth for `RunRequest` and `TrainEvent`; export OpenAPI and generate TS types (or hand-mirror the zod schema and contract-test both). Transport: `POST /runs` -> `{runId}`; `GET /runs/{id}/events` as **SSE** (`Last-Event-ID` resume); `POST /runs/{id}/cancel`; `GET /health` + `GET /capabilities` (which model ids and size limits the backend supports). Dataset goes up as multipart Parquet/CSV (already split + preprocessed client-side), predictions come back as an `artifact` event so the client computes metrics with the shared module.
**When to use:** When the backend phase starts; but define the event schema in Phase 1 so it exists as a contract.
**Trade-offs:** SSE over WebSocket: progress is one-directional, control is rare (cancel), SSE auto-reconnects and passes through proxies simply. Use WebSocket only if interactive RL control (live reward-shaping, step-by-step) needs low-latency bidirectional input.

## Data Flow

### Request Flow (tabular happy path)

```
File drop
   ↓
[Upload UI] → [Data Worker: parse (chunked) → infer Schema → columnar Dataset]
   ↓                         ↓ persists blobs (OPFS) + meta (Dexie)
[Preview slice + Schema + stats → main thread]
   ↓  user edits cleaning/feature steps (PipelineSpec draft, Zustand)
[Data Worker: interpret PipelineSpec → step snapshots (before/after summaries) + EDA summaries]
   ↓  user chooses target, split %, seed
[Data Worker: split → fit on train → transform train/test → typed-array buffers]
   ↓  user picks model + params (from Model Registry)
[Run Orchestrator] ──choose Executor──┬→ [Compute Worker: Engine.run → TrainEvents]
                                      └→ [RemoteExecutor → FastAPI SSE → TrainEvents]
   ↓ (batched events)
[Live Run Store ring buffers] → rAF → [Viz layer: curves / boundary / clusters / 3D]
   ↓ on 'done'
[Evaluation module: predictions → metrics/curves] → [Experiment Store: run record]
   ↓
[Compare view reads runs from Dexie → overlay curves, metrics table]
```

### State Management

```
Dexie (IndexedDB)  ──liveQuery──▶  Experiments/Compare views     (durable truth)
Zustand session slices ◀──actions── Wizard UI                    (ephemeral drafts: pipeline, config)
Live Run Store (external)  ──subscribe──▶ Charts (rAF-batched)   (hot path, never in React state)
Data Worker memory (columnar)  ◀──Comlink calls──  Runtime       (large data never lives in React)
```

Rule of thumb: **big or fast data lives in workers/external stores; React holds only small summaries and the user's intent.**

### Key Data Flows

1. **Dataset ingest:** `File` -> (transfer to Data Worker) -> chunked parse -> columnar typed arrays -> persisted -> only preview rows (<= ~200) + per-column stats sent to the main thread. Raw rows are never materialized as `Array<Record<string, any>>` (5-10x memory blow-up).
2. **Preprocessing:** `PipelineSpec` JSON goes *to* the worker; step snapshots (histograms, null counts, before/after samples) come *back*. The spec is saved with each run, making any run replayable.
3. **Training stream:** compute worker -> batched `TrainEvent[]` -> Live Run Store -> charts. `viz` events carry `Float32Array`s sent as transferables (decision-boundary grid = N x N class/probability map computed worker-side via `predict`, drawn on a Canvas).
4. **RL loop:** environment + agent step inside the compute worker at full speed; worker emits (a) per-episode reward metrics (cheap, every episode), (b) `rl-frame` snapshots (agent position / cart state) sampled at <= 30 Hz for animation; main thread animation (Framer Motion/Canvas) interpolates between frames. Speed slider changes the worker's sampling/throttle, not the sim.
5. **Backend run:** client sends already-split typed buffers + `RunRequest`; SSE returns the same `TrainEvent`s; final predictions returned as an `artifact`; client's metrics module produces the evaluation so browser and backend runs are directly comparable.
6. **Save / export:** Run record = `{configHash, pipelineSpec, datasetRef+hash, seed, executor, metricHistory (decimated), finalMetrics, modelArtifactRef, createdAt}`. Export = zip of JSON records + binaries + `manifest.json {schemaVersion}`; import runs migrations.

## Suggested Build Order (dependencies drive phases)

```
0 Foundation/contracts ──▶ 1 Ingest+DatasetStore ──▶ 2 Clean+EDA ──▶ 3 Features+Split
        │                                                                   │
        └──▶ (contracts: TrainEvent, Engine, PipelineSpec fixed here)       ▼
                                          4 VERTICAL SLICE: Engine+Worker+Live viz (1 supervised model)
                                                       │
                          ┌────────────────────────────┼────────────────────────┐
                          ▼                            ▼                        ▼
                5 Eval+Experiment store      6 More supervised +         7 RL (env + agent +
                  + Compare                    unsupervised + projection    animation; no dataset dep.)
                          └──────────────┬─────────────┘                        │
                                         ▼                                      │
                                 8 3D viz + motion polish  ◀────────────────────┘
                                         ▼
                              9 FastAPI backend + RemoteExecutor
                                         ▼
                              10 Export/import, a11y, perf hardening
```

| # | Build | Why here | Depends on |
|---|-------|----------|------------|
| 0 | **Foundation:** Next scaffold (client-only/static), design tokens, **prod-build worker + Comlink spike**, core types: `Dataset`, `Schema`, `PipelineSpec`, `TrainEvent`, `Engine`, `Environment` | Worker bundling and the event contract are the two things that are expensive to change later; prove them first | none |
| 1 | **Ingest + Dataset Store:** parsers (CSV/JSON/XLSX first; images/text can follow), schema inference, columnar store, preview, Dexie + OPFS | Everything tabular needs a dataset; the columnar decision constrains every later phase | 0 |
| 2 | **Cleaning + EDA** (pipeline interpreter v1: impute, dedupe, outliers, dtype; before/after snapshots) | Establishes the `fit/transform` op model; EDA is cheap once data is columnar | 1 |
| 3 | **Feature engineering + split** (encode, scale, select, text/image featurizers, leakage-safe split) | Produces the exact typed-array input every engine consumes | 2 |
| 4 | **Vertical slice: Engine abstraction + Run Orchestrator + compute worker + Live Run Store + loss curve + decision boundary** with *one* simple supervised model (logistic/linear) | De-risks the core value ("watch it learn") end-to-end before breadth; validates event batching and cancel | 0, 3 |
| 5 | **Evaluation + Experiment Store + Compare** | Metrics need predictions from (4); comparison needs persisted runs | 4 |
| 6 | **Algorithm breadth:** remaining supervised, unsupervised, projections (PCA/t-SNE/UMAP), anomaly | Mostly "add a registry entry + engine"; cheap once the seam is proven | 4, 5 |
| 7 | **RL:** `Environment` impls, Q-learning/SARSA/DQN, reward viz, agent animation | Only needs the Engine seam + Live store, **not** phases 1-3, so it can run in parallel with 5-6 | 4 |
| 8 | **3D + motion:** r3f projections/cluster views, loss-surface or RL scenes, Framer Motion polish | Pure consumers of ViewModels; safe to defer | 6, 7 |
| 9 | **FastAPI backend:** contract (OpenAPI), SSE endpoint, capability discovery, `RemoteExecutor`, size-based "use backend" hints | Interface already abstract; this is an additional Executor, not a rewrite | 4 (contract), 6 (parity) |
| 10 | **Export/import, migrations, a11y, perf budgets** | Needs stable record shapes (`schemaVersion`) | 5+ |

**Phase-structure implications for the roadmap:**
- Insist on the **vertical slice (4) before breadth (6, 7)**. A playground that has 12 half-wired algorithms and no streaming is worse than one algorithm that visibly learns.
- Define the **event schema and engine interface in Phase 0** even though the backend ships in Phase 9; retrofitting is the costliest mistake.
- RL is a **parallel track**, not "after unsupervised"; it is the paradigm that most stresses the streaming/animation path, so consider doing a thin RL slice (gridworld + Q-learning) right after Phase 4 to validate the abstraction against its hardest consumer.
- Images and text: ingestion (blob -> tensor via OffscreenCanvas / TF-IDF-hashing) is a **separate sub-track of phases 1 and 3**; ship tabular first, add them without touching the engine seam.

## Scaling Considerations

Scale here is **dataset size and model cost**, not user count (no server, no accounts).

| Scale | Architecture Adjustments |
|-------|--------------------------|
| <= ~10k rows, small models | Everything in-browser, JS engines; default path. Live viz at full fidelity. |
| ~10k-500k rows | Chunked parse, columnar typed arrays, subsample for EDA and boundary plots, decimated metric history, tfjs on WASM/WebGPU backend; warn at memory estimate thresholds; offer backend. |
| 500k+ rows / deep nets / big ensembles / large files | Route to FastAPI (sklearn/PyTorch). Client still does preview/schema/EDA on a sample; the file goes to backend directly, never fully parsed in the browser. |

### Scaling Priorities

1. **First bottleneck: main-thread jank and memory from data representation.** Fix: columnar typed arrays, parsing and pipelines in workers, never row-object arrays, send only summaries to React.
2. **Second bottleneck: chart update rate and point count.** Fix: ring buffers + rAF batching + LTTB decimation; switch Recharts -> Canvas for > a few thousand points; sample scatter for 3D.
3. **Third: IndexedDB quota / export size.** Fix: store decimated metric histories, weights in OPFS, `navigator.storage.estimate()` UI, `navigator.storage.persist()`, per-run delete.

## Anti-Patterns

### Anti-Pattern 1: Training or parsing on the main thread "just for small data"

**What people do:** Run `model.fit` or `Papa.parse` inline, add `await tf.nextFrame()` calls to keep UI alive.
**Why it's wrong:** UI hitches on any non-trivial model; cancel is impossible; the architecture can't later move work off-thread without rewriting every call site.
**Do this instead:** Workers from day one behind the `Engine` seam; the main thread never imports tfjs or engine code.

### Anti-Pattern 2: `setState` (or Zustand set) per training step

**What people do:** `onEpochEnd -> setHistory([...h, m])`, tensors or whole models in React state.
**Why it's wrong:** Re-render storms, O(n^2) array copies, memory leaks from retained tensors, dropped frames.
**Do this instead:** External Live Run Store with ring buffers and rAF-coalesced notifications; keep tensors strictly inside the worker; main thread sees plain numbers and typed arrays.

### Anti-Pattern 3: Fitting scalers/encoders/imputers before the split (data leakage)

**What people do:** Clean + scale the whole dataset, then split, so test statistics leak into training and "honest test results" are inflated.
**Why it's wrong:** Directly violates the project's core value ("honest test results").
**Do this instead:** Split is an early pipeline step; all stateful ops `fit` on train only; ops declare `train-only` vs `train-and-test`; show the leakage-safe order in the UI.

### Anti-Pattern 4: Two metric implementations (TS in browser, Python on backend)

**What people do:** Compute accuracy/ROC in sklearn on the server and in JS locally.
**Why it's wrong:** Silent definitional differences (averaging, tie handling, thresholds) make cross-run comparison meaningless.
**Do this instead:** Backend returns predictions/probabilities; one TS metrics module computes everything. Contract-test it against sklearn fixtures once.

### Anti-Pattern 5: Datasets as `Array<Record<string, unknown>>`

**What people do:** Keep parsed rows as objects and pass them around.
**Why it's wrong:** 5-10x memory, slow structured clone to workers, no zero-copy transfer, slow numeric loops.
**Do this instead:** Columnar typed arrays with dictionary-encoded categoricals; transfer `ArrayBuffer`s.

### Anti-Pattern 6: One monolithic worker / reusing the compute worker across runs

**What people do:** One worker holds data, runs training, runs EDA.
**Why it's wrong:** A long run blocks EDA/preview responses; leaked tfjs memory accumulates; cancel is cooperative-only.
**Do this instead:** Long-lived Data Worker + disposable per-run Compute Worker (terminate = cancel + cleanup).

### Anti-Pattern 7: Coupling visualizations to engines

**What people do:** Decision-boundary component imports the tfjs model and calls `predict`.
**Why it's wrong:** Breaks when the model runs remotely or in a worker; makes viz untestable.
**Do this instead:** Engine emits `viz` events (ViewModels, e.g. a grid of class IDs); components render ViewModels only. For remote engines, the backend/`predict` endpoint supplies the same grid.

### Anti-Pattern 8: Reaching for SharedArrayBuffer by default

**What people do:** Use SAB for zero-copy sharing.
**Why it's wrong:** Requires COOP/COEP cross-origin isolation headers, which constrain/break third-party embeds and complicate static hosting and Next config.
**Do this instead:** Transferable `ArrayBuffer`s and per-run copies; revisit SAB only if profiling proves a need.

### Anti-Pattern 9: Forcing RL through the dataset pipeline

**What people do:** Model RL as "dataset = environment" to reuse the wizard.
**Why it's wrong:** Misleading UX and awkward abstractions; RL's inputs are an `EnvSpec` + hyperparameters, its outputs are episodes and policies.
**Do this instead:** Separate session kind and route; shared Engine/Live store/Experiment store beneath.

## Integration Points

### External Services

| Service | Integration Pattern | Notes |
|---------|---------------------|-------|
| FastAPI backend (user-run) | `RemoteExecutor`: `POST /runs`, SSE `GET /runs/{id}/events`, `POST /runs/{id}/cancel`, `GET /capabilities` | Configurable base URL (default `http://localhost:8000`); needs CORS config. If the site is served over HTTPS and calls a local HTTP backend, browsers may apply mixed-content / local-network-access permission prompts (verify per browser). Run it as an explicit, opt-in "Connect backend" action with a health check; never silent. |
| TensorFlow.js (in compute worker) | Dynamic `import()` inside worker; pick backend `webgpu` -> `webgl` -> `wasm` -> `cpu` with feature detection | Maintenance trajectory of tfjs is uncertain (see Open Verification Items); isolate behind `Engine` so replacing it with ONNX Runtime Web / LiteRT.js or hand-written JS does not leak. Always `tf.tidy`/`dispose`. |
| ONNX Runtime Web (optional) | Inference of imported/pre-trained models; browser training is CPU-WASM only and limited | Do not plan v1 on ORT training; use for inference-style features at most. |
| Classic ML in JS | Hand-rolled or `ml-*` packages for k-means, trees, kNN, PCA, DBSCAN | Prefer implementations that can expose per-iteration state (centroids per step, tree splits) over black-box `fit`, since the product is *watching* learning. |
| Pyodide + scikit-learn (alternative) | Run in a dedicated worker for sklearn parity without a backend | Large download (tens of MB), partial coverage, slow cold start; keep as a possible later `Engine` implementation, not a v1 foundation. |

### Internal Boundaries

| Boundary | Communication | Notes |
|----------|---------------|-------|
| UI <-> Runtime | Direct TS calls + store subscriptions | UI never sees `Worker`/`EventSource`. |
| Runtime <-> Data Worker | Comlink RPC; large buffers via transfer | Data Worker owns the canonical dataset; main holds summaries. |
| Runtime <-> Compute Worker | Comlink + batched event callback (proxy) or raw `MessagePort` for hot path | Terminate = cancel. Batch >= 1 frame worth. |
| Runtime <-> Backend | HTTP + SSE | Event schema identical to worker events. |
| Engines <-> Environment (RL) | Direct in-worker calls | Never crosses the thread boundary per step. |
| Core <-> everything | Imports only (pure functions/types) | Enforce with lint rule: `core/` may not import from `react`, `next`, DOM globals, or `engines/`. |
| Storage <-> rest | Dexie repo functions; versioned schema + migrations | Only `storage/` touches IndexedDB/OPFS. |
| Viz <-> Stores | Read ViewModels via hooks | Viz never imports `engines/`. |

## Open Verification Items (flag for phase-level research)

| Item | Confidence | Why it matters | Resolve in |
|------|-----------|----------------|-----------|
| Worker bundling under current Next.js (`new Worker(new URL(..., import.meta.url))` with Turbopack and webpack, dev and prod). Search results show Turbopack reworked worker handling recently and report at least one regression in a Node-worker case in 16.2.x | MEDIUM | Whole execution model depends on it | Phase 0 spike on the pinned Next version; test `next build` output, not just dev |
| TensorFlow.js long-term status and WebGPU/WebGL backends inside Web Workers (OffscreenCanvas); secondary sources suggest Google is steering toward LiteRT.js, no official tfjs maintenance statement found | LOW-MEDIUM | Choice of NN engine; mitigated by `Engine` abstraction | Phase 0/4 spike; check tfjs GitHub releases |
| Browser behavior for HTTPS-origin -> `http://localhost` backend calls (mixed content, Local/Private Network Access prompts) | LOW-MEDIUM | Backend UX for hosted deploys | Phase 9 research |
| ONNX Runtime Web training capability (reported CPU/WASM only, WebGPU backend experimental) | MEDIUM | Decides whether ORT is inference-only | Only if ONNX import is scoped |
| Storage quotas / OPFS availability across Safari/Firefox | MEDIUM | Persistence layer fallbacks | Phase 1 |

## Sources

- TensorFlow Playground, Teachable Machine, and sklearn-in-Pyodide playground repos (established pattern: UI + isolated compute + streamed state); e.g. sklearn-classifiers-playground (Pyodide, worker usage listed as TODO) - https://github.com/stefano-meschiari/sklearn-classifiers-playground (MEDIUM, via web search)
- ONNX Runtime Web README (WebGPU experimental, WebGL maintenance mode) - https://cdn.jsdelivr.net/npm/onnxruntime-web@1.30.0/README.md (MEDIUM)
- Microsoft Open Source blog on ORT Web training (CPU/WASM) - https://opensource.microsoft.com/blog/wp-json/wp/v2/posts/95050 (MEDIUM, possibly outdated)
- LiteRT.js / tfjs direction (secondary coverage, unofficial) - https://xenospectrum.com/en/google-litert-js-browser-ai-webgpu-wasm-webnn/ (LOW)
- Next.js Turbopack worker entrypoint change (#88602) and regression report (#93432 / issue 86495) - https://github.com/vercel/next.js/issues/86495 (MEDIUM)
- Comlink + Next.js examples - https://www.github.com/koheitakumi/nextjs-typescript-comlink , https://blog.logrocket.com/comlink-web-workers-match-made-in-heaven (MEDIUM)
- Pyodide/scikit-learn in-browser (partial support, large size) - https://www.kdnuggets.com/2020/07/why-put-scikit-learn-browser.html (LOW-MEDIUM, dated)
- Web platform behavior assumed from standard knowledge (Transferables, OPFS, IndexedDB, SSE `Last-Event-ID`, COOP/COEP requirement for SharedArrayBuffer) (HIGH, not re-fetched this session)
- Provider confidence tiers were not run through the `classify-confidence` seam (gsd-tools not located; only built-in WebSearch used); tiers above are manual judgments.

---
*Architecture research for: interactive ML playground (Next.js + browser ML workers + optional FastAPI)*
*Researched: 2026-11-09*
