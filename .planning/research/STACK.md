# Technology Stack

**Project:** ML Playground (browser-first interactive ML app + optional FastAPI backend)
**Researched:** 2026-11-09
**Method:** Versions verified directly against the npm and PyPI registries on 2026-11-09 (Context7/MCP tools were not available in this session). Next.js 16 behavior verified against the official release post. SheetJS install guidance verified against docs.sheetjs.com. Claims not verifiable from a registry are marked MEDIUM/LOW.

Confidence legend: HIGH = registry/official docs verified this session; MEDIUM = registry verified for version but behavior/fit based on ecosystem knowledge; LOW = unverified, validate in a spike.

---

## Recommended Stack

### Core Framework

| Technology | Version | Purpose | Why | Conf. |
|------------|---------|---------|-----|-------|
| Node.js | 24 LTS (>=20.9 minimum) | Toolchain runtime | Next 16 requires >=20.9; Node 20 is at/near EOL, so start on 24 LTS | MEDIUM |
| Next.js (App Router) | 16.4.5 | App shell, routing, bundling | Turbopack is now the stable default bundler; React Compiler support is stable; Node 20.9+/TS 5.1+/Chrome 111+/Safari 16.4+ baseline is fine for a WebGPU/WASM app. Use it as a mostly-static client app (no server routes needed for v1) | HIGH |
| React / React DOM | 19.3.1 | UI | Required by Next 16 and R3F v9. NOTE: `@react-three/fiber` 9.8.2 peers `react >=19 <19.4`; do not bump React to 19.4 until R3F lifts the cap | HIGH |
| TypeScript | **~6.0.3 (pin; do NOT take 7.x)** | Types | npm `latest` is 7.0.2 (native compiler), but `typescript-eslint` 8.72.1 (used by eslint-config-next) peers `typescript >=4.8.4 <6.1.0`. Pin `~6.0.3` until typescript-eslint supports 7 | HIGH |
| Tailwind CSS | 4.3.5 + `@tailwindcss/postcss` 4.3.5 | Styling | v4 is CSS-first (`@import "tailwindcss"` + `@theme`), no `tailwind.config.js`. Design tokens from the ui-ux-pro-max design system map cleanly to `@theme` variables | HIGH |
| `babel-plugin-react-compiler` | 1.0.0 | Auto-memoization | Enable via `reactCompiler: true` in next.config.ts; cuts re-render cost for chart/table-heavy screens. Off by default in Next; costs build time (Babel). Enable after first phase if build time is acceptable | MEDIUM |
| pnpm | 12.x | Package manager | Fast, strict node_modules; good for the large ML dep tree. (npm works too.) | MEDIUM |

### In-Browser ML Runtime (the load-bearing decision)

| Technology | Version | Purpose | Why | Conf. |
|------------|---------|---------|-----|-------|
| `@tensorflow/tfjs` (+ `-backend-wasm`, `-backend-webgpu`) | 4.22.0 | Neural nets, DQN, gradient-descent models, in-browser **training** | Still the only mature JS library with autograd + `model.fit` callbacks (per-epoch/batch hooks give live loss curves). **Caveat: last stable release Oct 2024, a 4.23.0-rc.0 on the `next` tag from Jan 2025; effectively maintenance-mode.** Acceptable because it is stable and the API we need is tiny. Isolate behind a `ModelRuntime` interface so it can be swapped. Backend order: webgpu -> webgl -> wasm -> cpu with feature detection | HIGH (version) / MEDIUM (long-term health) |
| `onnxruntime-web` | 1.31.1 | **Inference** of ONNX models (backend-trained models exported with skl2onnx/torch.onnx, imported user models) | Active (Nov 2026 release), WASM/WebGPU/WebNN EPs. Use for inference only, not training. Package is ~150 MB unpacked: ship only the wasm/webgpu bundle you use and self-host the `.wasm` files (copy to `public/ort/`, set `ort.env.wasm.wasmPaths`) | HIGH (version) / MEDIUM (Turbopack wiring) |
| `@huggingface/transformers` | 4.3.2 | Text/image featurization (sentence embeddings, CLIP/ViT features) | The maintained successor to `@xenova/transformers` (stuck at 2.17.2). Runs on onnxruntime-web, in a worker, with model files cached by the browser. Lazy-load only when user picks "embed text/images" | HIGH (version) / MEDIUM (fit) |
| `ml-matrix` | 6.15.1 | Linear algebra for hand-written/ml.js algorithms | Active, dependency of the ml.js family | HIGH |
| ml.js family: `ml-random-forest` 2.1.0, `ml-cart` 2.1.1, `ml-knn` 3.0.0, `ml-pca` 4.1.1, `ml-kmeans` 7.0.1, `ml-hclust` 4.0.0, `ml-regression` 6.3.0, `ml-naivebayes` 4.0.0 | as listed | Black-box classical models (forest, kNN, PCA, hierarchical clustering, Naive Bayes) | Zero-config, pure JS, work in a worker. **Most were last published 2022** (kmeans/hclust/regression are the exceptions): stable but frozen; they have **no per-iteration callbacks**, so they cannot drive live-training animation | MEDIUM |
| `umap-js` | 1.4.0 | UMAP in browser | `fit`/`fitAsync` with epoch callback gives animated embedding. Last release 2024; only viable JS UMAP | MEDIUM |
| Hand-written TS (no dependency) | n/a | Stepwise algorithms: linear/logistic regression via GD, k-means, DBSCAN, t-SNE, tabular Q-learning/SARSA, CartPole/gridworld environments | Anything that must **animate each step** (centroids moving, boundary refining, Q-table filling) is easier to write as a generator/`step()` than to wrap a library without hooks. Gridworld and CartPole are ~50-100 lines each; no gym-style JS library is worth adding | HIGH (as a decision) |

### Data Ingestion and Wrangling (all in a Web Worker)

| Technology | Version | Purpose | Why | Conf. |
|------------|---------|---------|-----|-------|
| `papaparse` | 5.7.0 | CSV/TSV parsing | De facto standard; streaming (`step`/`chunk`) and `worker: true`; handles delimiter/header detection and malformed rows. Use `dynamicTyping: false` and do our own schema inference (avoids silent coercion of IDs/zip codes) | HIGH |
| **SheetJS `xlsx` from the SheetJS CDN tarball** | 0.20.3 | Excel (.xlsx/.xls/.ods) parsing | **Do NOT `npm i xlsx`**: npm is frozen at 0.18.5, which carries CVE-2023-30533 (prototype pollution) and a ReDoS fixed in 0.20.2. Install `npm i https://cdn.sheetjs.com/xlsx-0.20.3/xlsx-0.20.3.tgz` (docs.sheetjs.com calls the CDN the authoritative source) and pin it in the lockfile. Parse in a worker (user-supplied files are untrusted input) | HIGH |
| `arquero` | 8.0.3 | In-memory columnar table wrangling (filter, derive, group/rollup, joins, pivot, sample) | Small, dependency-light, columnar, Apache Arrow-compatible, immutable verbs map 1:1 to "cleaning steps" so each step can be recorded/replayed/undone for before/after view. Maintained | HIGH (version) / MEDIUM (fit) |
| `simple-statistics` | 8.0.0 | Descriptive stats, correlation, quantiles/IQR for outliers | Tiny, tree-shakable, typed; covers EDA summary stats without pulling mathjs | MEDIUM |
| `d3-array` (via `d3` 7.9.2 or standalone) | 3.x | Binning/histograms, quantiles | Needed anyway for charts | HIGH |
| `fflate` | 0.8.3 | ZIP in/out (image folders uploaded as zip; experiment export bundle) | Fast, small, worker-friendly; `jszip` is heavier and slower | HIGH |
| Native browser APIs | n/a | Images: `createImageBitmap` + `OffscreenCanvas` -> tensors; text: `File.text()`/`TextDecoder`; JSON: `JSON.parse` in worker | No library needed. Keep image pipeline off the main thread | HIGH |
| `zod` | 4.6.7 | Validate imported experiment files, backend message payloads, dataset schema objects | Trust-boundary validation for anything user-imported or network-received | HIGH |

Optional/deferred: `@duckdb/duckdb-wasm` (1.33.1-dev; SQL over large CSV/Parquet) and `hyparquet` (1.33.0, Parquet) only if "large file" support proves necessary in-browser; otherwise route big files to the backend.

### Concurrency and State

| Technology | Version | Purpose | Why | Conf. |
|------------|---------|---------|-----|-------|
| Web Workers (module workers, `new Worker(new URL('./x.worker.ts', import.meta.url), {type:'module'})`) | n/a | Parsing, wrangling, training, RL simulation off the main thread | Turbopack and webpack both support the `new URL` worker pattern. Hard requirement from PROJECT.md (training must not freeze UI) | HIGH |
| `comlink` | 4.4.2 | RPC over postMessage | Tiny; stale since late 2024 but stable. Use for request/response (`await worker.fit(cfg)`); use `Comlink.proxy` callbacks or a dedicated `MessagePort` for progress streaming; transfer `Float32Array`/`ArrayBuffer` (never clone big tensors) | HIGH |
| `zustand` | 5.1.0 | Client state (current dataset, pipeline steps, run status, live metric buffers) | Minimal, works with React Compiler and external-store updates from worker callbacks. Throttle metric pushes (batch per animation frame) | MEDIUM |
| `dexie` + `dexie-react-hooks` | 4.4.7 / 4.4.0 | IndexedDB persistence (datasets, pipelines, runs, model weights) with `useLiveQuery` | Best DX over raw IndexedDB; schema versioning; Blob storage works. Request `navigator.storage.persist()` to reduce eviction risk. Prefer over `idb` 8.0.4 (lower level) | HIGH |

### Visualization and UI

| Technology | Version | Purpose | Why | Conf. |
|------------|---------|---------|-----|-------|
| `recharts` | 3.10.1 | Loss/accuracy curves, bar charts, ROC/PR, model-comparison charts | User-specified; v3 is current and React 19 compatible. SVG-based: set `isAnimationActive={false}` for live streams and cap/decimate to a few thousand points | HIGH (version) / MEDIUM (perf) |
| `d3` (import per-module: `d3-scale`, `d3-shape`, `d3-contour`, `d3-delaunay`, `d3-hierarchy`, `d3-array`, `d3-interpolate`) | 7.9.2 | Math/scales/geometry for custom charts; contour for decision boundaries; dendrograms | Use D3 for **math, not DOM**: let React render, D3 compute. Render dense plots (scatter >5k points, decision-boundary grids, heatmaps) to **`<canvas>`** instead of SVG | HIGH |
| `three` | 0.187.0 | 3D engine | User-specified. Pin exact (0.x makes breaking changes between minors); `@types/three` 0.187.0 must match | HIGH |
| `@react-three/fiber` | 9.8.2 | React renderer for three | v9 pairs with React 19. Use `frameloop="demand"` for static scenes | HIGH |
| `@react-three/drei` | 10.7.9 | Helpers (OrbitControls, Text/Html labels, Line, Instances, Points) | v10 pairs with R3F v9/React 19 | HIGH |
| `motion` (import from `motion/react`) | 14.11.1 | UI animation (Framer Motion) | "Framer Motion" is now published as `motion` (same version as `framer-motion` 14.11.1; both exist). Use `motion`; wrap app in `<MotionConfig reducedMotion="user">`. Do not drive per-frame simulation (RL agent) with it; use canvas/rAF | HIGH (version) / MEDIUM (naming guidance) |
| shadcn CLI (`shadcn`) + `radix-ui` + `lucide-react` + `clsx` + `tailwind-merge` + `class-variance-authority` | 4.23.0 / 1.8.1 / 1.70.0 / 2.1.1 / 3.7.0 / 0.7.1 | Accessible component primitives copied into the repo, styled with Tailwind tokens | Radix gives accessible dialogs/tabs/select/slider for free (meets the accessibility requirement); components are owned code, so the ui-ux-pro-max design system can restyle them. `tailwind-merge` must be v3 for Tailwind 4 | MEDIUM |
| `next-themes` | 0.4.6 | Light/dark | Standard | HIGH |
| `@tanstack/react-table` + `@tanstack/react-virtual` | 9.3.1 / 3.14.26 | Dataset preview table with virtualized rows/columns | Headless; virtualization is mandatory for 100k-row previews. react-table is on a new major (v9): skim its migration notes first, or start with `react-virtual` + a plain table | MEDIUM |
| `react-dropzone` | 20.2.0 | Drag-and-drop upload with a11y | Handles accept-type filtering, folder drops, keyboard operation | MEDIUM |

### Optional Backend (user-run, opt-in)

| Technology | Version | Purpose | Why | Conf. |
|------------|---------|---------|-----|-------|
| Python | 3.12 (3.13 OK; floor is 3.11) | Runtime | torch 2.15 and scikit-learn 1.9.1 require >=3.11; numpy 2.5.4 and xgboost 3.4.1 require >=3.12 | HIGH |
| FastAPI | 0.162.2 | REST + streaming API | Standard; Pydantic v2 models double as the contract mirrored by zod on the frontend | HIGH |
| uvicorn | 0.54.0 | ASGI server | Standard | HIGH |
| pydantic | 2.14.3 | Request/response models | Bundled with FastAPI | HIGH |
| `sse-starlette` | 3.7.0 | `EventSourceResponse` for training-progress streaming | **SSE over WebSocket** for one-way progress (auto-reconnect, plain HTTP, easier proxying). Start training with `POST /runs`, stream `GET /runs/{id}/events`, stop with `DELETE`. Use WebSocket only if truly bidirectional control (interactive RL step/pause) is needed | MEDIUM |
| scikit-learn | 1.9.1 | Classical ML (SVM, forests, GBMs, clustering, metrics) | Reference implementation; also the source of truth to cross-check the JS models | HIGH |
| PyTorch | 2.15.0 (CPU wheels by default) | Deep nets, DQN | Per-epoch callbacks are trivial in a train loop; offer CUDA if the user has it | HIGH |
| numpy / pandas (or polars) | 2.5.4 / 3.1.1 (polars 2.1.0) | Data handling | pandas 3.x is current; polars optional for large files | HIGH |
| `umap-learn` | 0.5.12 | UMAP/t-SNE-class reductions for big data | Where browser UMAP is too slow | HIGH |
| `skl2onnx` + `onnx` | 1.20.0 / 1.23.2 | Export sklearn/torch models to ONNX for **browser inference** | Enables "train heavy on backend, explore in browser" with onnxruntime-web | MEDIUM |
| `openpyxl` | 3.1.5 | Excel reading server-side | Standard | HIGH |
| `python-multipart` | 0.0.33 | File upload parsing | Required by FastAPI for `UploadFile` | HIGH |
| `uv` | 0.13.x | Python env/dependency management | One-command setup for end users running the backend | MEDIUM |
| `pytest` + `httpx` | latest | Backend tests | Standard | MEDIUM |

Optional later: `xgboost` 3.4.1, `lightgbm` 4.7.0, `gymnasium` 1.4.0 (RL parity checks), `pyarrow` 26.0.1.

Training execution rule: run fits in a worker thread/process (`ProcessPoolExecutor` or `anyio.to_thread`), push progress through an `asyncio.Queue` the SSE generator drains. Never train inside the event loop. sklearn has no universal epoch callback: use `partial_fit`/`warm_start`/`max_iter` loops for SGD/MLP/boosting, and report coarse "stage" progress for models that cannot be stepped.

### Frontend Streaming Client

| Library | Version | Purpose | When to Use | Conf. |
|---------|---------|---------|-------------|-------|
| `eventsource-parser` | 4.1.3 | Parse SSE from `fetch()` streams | Native `EventSource` is GET-only with no body/headers; use `fetch` + this parser for POST-initiated streams and `AbortController` cancellation | MEDIUM |

### Testing and Tooling

| Library | Version | Purpose | When to Use | Conf. |
|---------|---------|---------|-------------|-------|
| `vitest` | 5.1.0 | Unit tests for algorithms, wrangling steps, metrics | Pure-TS ML logic (k-means, Q-learning, env physics, metric math) must have unit tests with fixed seeds | HIGH |
| `@vitest/browser` + `playwright` | 5.1.0 / 1.65.0 | Worker/WebGPU/IndexedDB tests in a real browser | Anything touching workers, OffscreenCanvas, Dexie | MEDIUM |
| `@playwright/test` | 1.65.0 | E2E (upload -> clean -> train -> evaluate) | Smoke test per phase | HIGH |
| `@testing-library/react` + `jsdom` | 16.3.3 / 31.0.0 | Component tests | UI components only; do not test ML in jsdom | MEDIUM |
| `eslint` + `eslint-config-next` + `typescript-eslint` | 10.14.0 (fall back to 9.x if a plugin lags) / 16.4.5 / 8.72.1 | Lint | Next 16 removed `next lint`; run ESLint directly (flat config). Biome 2.6.6 is a valid single-tool alternative but lacks Next/React-hooks rule coverage | MEDIUM |
| `@webgpu/types` | 0.1.74 | WebGPU typings | dev dep for backend detection code | MEDIUM |
| `@types/node`, `@types/react`, `@types/react-dom`, `@types/three` | 22-24 matching Node / 19.3.0 / 19.x / 0.187.0 | Types | Match runtime majors (the registry `@types/node` is 26.x; pick the major matching your Node) | MEDIUM |

---

## Alternatives Considered

| Category | Recommended | Alternative | Why Not |
|----------|-------------|-------------|---------|
| Dataframe | Arquero 8.0.3 | Danfo.js (`danfojs` 1.2.0) | Last release Apr 2025; drags in `@tensorflow/tfjs ^3.13` (old, duplicates our 4.22), plotly, mathjs 9, and a CDN xlsx; heavy and stale. Pandas-like API is nice but not worth it |
| Dataframe | Arquero | DuckDB-WASM | Excellent for SQL on large files but ~MBs of WASM and a separate memory space; defer, use backend for big data |
| Excel | SheetJS CDN 0.20.3 | `xlsx` from npm | Frozen at 0.18.5 with known CVEs |
| Excel | SheetJS CDN 0.20.3 | `read-excel-file` 9.3.14 / `exceljs` 4.4.0 | `read-excel-file` is a safe npm-native fallback for .xlsx only (no .xls); `exceljs` is stale (Dec 2024) and heavy |
| Browser DL training | TF.js 4.22.0 | ONNX Runtime Web training | ORT-web is inference-focused; no practical in-browser training path for this use case |
| Browser DL training | TF.js | LiteRT.js | Inference runtime for `.tflite`; no training. Revisit only if TF.js becomes unusable |
| Browser DL training | TF.js | brain.js / synaptic / ConvNetJS | Abandoned or toy-grade; no GPU backends |
| Browser sklearn | ml.js + hand-written | Pyodide 314.0.7 (scikit-learn in WASM) | Real sklearn parity with zero backend, but 10+ MB initial download, slow start, no per-step hooks. Keep as a possible later "full sklearn in browser" mode; not v1 |
| Charts | Recharts + D3/canvas | visx 4.0.0 | Another abstraction layer over D3 that we do not need; Recharts for standard charts, D3+canvas for custom |
| Charts | Recharts + D3/canvas | Plotly (`plotly.js-dist-min` 4.2.0) | ~MB-scale bundle, own styling system, fights the design system |
| Charts | Recharts + D3/canvas | ECharts 6.1.0 / uPlot 1.6.32 | Both are legitimate (ECharts canvas performance; uPlot for 100k-point live lines). Not user-specified; keep uPlot as the **escape hatch** if Recharts cannot keep up with live curves |
| Charts (TF) | Recharts | `@tensorflow/tfjs-vis` 1.5.1 | Last release Nov 2023; depends on preact 8, vega 5, glamor; will not fit design system |
| Animation | `motion` | GSAP / react-spring | User chose Framer Motion |
| State | Zustand | Redux Toolkit / Jotai | Redux is overkill for local-first single-user; Zustand is enough |
| Server state | none (local-first) | TanStack Query 5.107.1 | No remote CRUD in v1; backend interaction is run/stream/cancel, simpler with plain `fetch`. Add only if backend job-list polling grows |
| Streaming | SSE | WebSocket / Socket.IO | One-way progress needs SSE only. Socket.IO adds a protocol and bundle for no benefit |
| Persistence | Dexie | raw IndexedDB / `idb` / localStorage | localStorage cannot hold datasets; raw IDB is boilerplate |
| Lint/format | ESLint + eslint-config-next | Biome 2.6.6 | Viable, faster; chosen ESLint for Next + React Compiler/hooks rule coverage |
| Language | TypeScript 6.0.x | TypeScript 7.0.x | Tooling (typescript-eslint) does not yet accept 7 |
| Sentence/image embeddings | `@huggingface/transformers` 4.3.2 | `@xenova/transformers` 2.17.2 | Superseded |
| Backend framework | FastAPI | Flask / Django | User-specified; async streaming is the reason |

## What NOT to Use

- `npm i xlsx` (0.18.5): known CVEs; use the CDN tarball.
- `danfojs`, `@tensorflow/tfjs-vis`, `@xenova/transformers`, brain.js/synaptic/ConvNetJS, `ml-svm` for serious SVM (use backend sklearn, or libsvm-js 0.2.1 WASM if browser SVM is mandatory).
- `middleware.ts` (deprecated in Next 16; use `proxy.ts` if ever needed), Pages Router, `next lint`, sync `params`/`cookies()` access.
- Server Actions or Next API routes for training: training lives in workers (browser) or FastAPI (backend), never in the Node runtime.
- SSR of tfjs/three/onnx: import via `next/dynamic` with `ssr: false` or inside `"use client"` leaf components that load lazily, otherwise builds and hydration fail on `window`/WebGL access.
- Training or CSV/Excel parsing on the main thread.
- Moving large arrays by structured clone: transfer `ArrayBuffer`s or use `SharedArrayBuffer` (needs cross-origin isolation, see pitfalls).
- Importing the whole `d3` namespace and Plotly in the client bundle.
- `parseFloat`-style naive CSV type inference or `dynamicTyping: true` for schema detection.

## Cross-Cutting Configuration Notes

1. **Cross-origin isolation** (`Cross-Origin-Opener-Policy: same-origin`, `Cross-Origin-Embedder-Policy: require-corp`) is needed for `SharedArrayBuffer`, i.e. multi-threaded WASM in TF.js/ORT. Set via `headers()` in `next.config.ts`. Self-host fonts, wasm and model files (COEP blocks uncredentialed cross-origin loads). If the host cannot set headers (e.g. GitHub Pages), single-threaded WASM still works; do not make isolation a hard dependency. MEDIUM.
2. **Backend reachability**: an HTTPS-hosted frontend calling a user's `http://localhost:8000` backend hits CORS plus browser local-network/private-network access rules (Chrome permission prompts; Safari stricter on mixed content). Document "run frontend and backend locally" as the supported path and enable FastAPI CORS for the configured origin. MEDIUM; spike this in the backend phase.
3. **Backend selection**: feature-detect `navigator.gpu`; order TF.js backends webgpu -> webgl -> wasm -> cpu; surface the active backend in the UI so "why is this slow" is answerable.
4. **Bundle hygiene**: all ML libs (`tfjs`, `onnxruntime-web`, `transformers`, `three`) are route-level lazy imports inside workers or client islands; initial landing page must not pay for them.

## Installation

```bash
# Scaffold (Next 16 / App Router / TS / Tailwind 4 / ESLint)
pnpm create next-app@latest ml-playground --ts --tailwind --eslint --app --turbopack

# Core + framework pins
pnpm add next@16.4.5 react@19.3.1 react-dom@19.3.1
pnpm add -D typescript@~6.0.3 @types/node @types/react @types/react-dom babel-plugin-react-compiler

# ML in browser
pnpm add @tensorflow/tfjs@4.22.0 @tensorflow/tfjs-backend-wasm@4.22.0 @tensorflow/tfjs-backend-webgpu@4.22.0 \
  onnxruntime-web@1.31.1 @huggingface/transformers@4.3.2 \
  ml-matrix ml-random-forest ml-cart ml-knn ml-pca ml-kmeans ml-hclust ml-regression ml-naivebayes umap-js

# Data
pnpm add papaparse arquero simple-statistics fflate zod react-dropzone
pnpm add -D @types/papaparse
pnpm add https://cdn.sheetjs.com/xlsx-0.20.3/xlsx-0.20.3.tgz   # NOT npm "xlsx"

# State / workers / persistence / streaming
pnpm add zustand comlink dexie dexie-react-hooks eventsource-parser

# Viz / UI
pnpm add recharts d3 three@0.187.0 @react-three/fiber @react-three/drei motion \
  @tanstack/react-table @tanstack/react-virtual next-themes \
  radix-ui lucide-react clsx tailwind-merge class-variance-authority
pnpm add -D @types/d3 @types/three@0.187.0 @webgpu/types
pnpm dlx shadcn@latest init

# Test / lint
pnpm add -D vitest @vitest/browser playwright @playwright/test @testing-library/react jsdom eslint eslint-config-next typescript-eslint
```

```bash
# Optional backend (backend/ directory), Python 3.12
uv init && uv add fastapi uvicorn[standard] sse-starlette pydantic python-multipart \
  numpy pandas scikit-learn umap-learn openpyxl skl2onnx onnx
uv add torch --index https://download.pytorch.org/whl/cpu   # CPU build by default; users with CUDA swap the index
uv add --dev pytest httpx
```

## Sources

- npm registry (queried 2026-11-09): next 16.4.5, react 19.3.1, typescript 7.0.2 (latest) / 6.0.3, typescript-eslint 8.72.1 peer `typescript <6.1.0`, tailwindcss 4.3.5, @tensorflow/tfjs 4.22.0 (published 2024-10; 4.23.0-rc.0 on `next` tag 2025-01), onnxruntime-web 1.31.1, @huggingface/transformers 4.3.2, three 0.187.0, @react-three/fiber 9.8.2 (peer react `>=19 <19.4`), @react-three/drei 10.7.9, recharts 3.10.1, motion/framer-motion 14.11.1, dexie 4.4.7, arquero 8.0.3, danfojs 1.2.0 (deps: tfjs ^3.13), papaparse 5.7.0, xlsx 0.18.5, zustand 5.1.0, comlink 4.4.2, zod 4.6.7, vitest 5.1.0, playwright 1.65.0 (HIGH)
- PyPI (queried 2026-11-09): fastapi 0.162.2, uvicorn 0.54.0, scikit-learn 1.9.1 (py>=3.11), torch 2.15.0 (py>=3.11), numpy 2.5.4 (py>=3.12), pandas 3.1.1, sse-starlette 3.7.0, onnx 1.23.2, skl2onnx 1.20.0, umap-learn 0.5.12, xgboost 3.4.1, gymnasium 1.4.0 (HIGH)
- Next.js 16 release post, https://nextjs.org/blog/next-16 (Turbopack default, React Compiler stable, Node 20.9+/TS 5.1+, `next lint` removed, `middleware.ts` -> `proxy.ts`) (HIGH)
- SheetJS installation docs, https://docs.sheetjs.com/docs/getting-started/installation/nodejs (CDN tarball 0.20.3 is authoritative; npm is 0.18.5) (HIGH)
- SheetJS issue tracker, https://git.sheetjs.com/sheetjs/sheetjs/issues/2961 (CVE-2023-30533 on npm 0.18.5) (MEDIUM)
- LiteRT.js / TF.js status coverage (web search; no official TF.js deprecation notice found, TF.js treated as maintenance-mode based on release cadence) (LOW-MEDIUM)
