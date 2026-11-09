# Pitfalls Research

**Domain:** Interactive in-browser ML playground (Next.js App Router + TF.js/ONNX in Web Workers + optional FastAPI + three.js/R3F), user-uploaded data, supervised/unsupervised/RL
**Researched:** 2026-11-09
**Confidence:** MEDIUM overall. Next.js/SSR, worker, WebGL, leakage, metrics and parsing pitfalls are well-established (HIGH, domain knowledge plus the SSR search below). TF.js project health, WebGPU backend browser coverage, and Chrome local-network-access behaviour change quickly and are flagged for verification (LOW/MEDIUM). No Context7 pass was done; web search returned thin results for worker bundling and TF.js maintenance.

> **Phase labels.** No roadmap exists yet. The phase names below are indicative groupings; the roadmapper should map them onto real phase numbers:
> **F** Foundation (Next.js shell, design system, worker/compute architecture, storage) | **I** Data ingest | **C** Cleaning + EDA | **FE** Feature engineering + split | **S** Supervised models + evaluation | **U** Unsupervised | **RL** Reinforcement learning | **V** Visualization/3D | **B** FastAPI backend | **X** Experiments, comparison, export, polish.

## Critical Pitfalls

### Pitfall 1: Training (or parsing, or featurizing) on the main thread, then retrofitting workers
**What goes wrong:** UI freezes during `model.fit`, k-means, t-SNE, CSV parse, or boundary-grid prediction. Retrofitting a worker later means rewriting every ML call site because the sync, shared-memory style (calling TF.js directly from React components) does not translate to message passing.
**Why it happens:** The first demo (iris, 150 rows) runs fine on the main thread. Libraries expose simple async-looking APIs (`await model.fit`) that still hog the thread between awaits and inside sync ops like `dataSync()`.
**How to avoid:** Make the compute boundary a Phase F decision. One typed RPC layer (Comlink or a hand-rolled `{id, type, payload}` protocol) between UI and a "compute worker". UI components never import `@tensorflow/tfjs`, `ml-*` or parsers directly; they only call the client of the worker API. Every long job returns a `jobId`, streams progress, and supports `cancel(jobId)`. Define the job interface once so the FastAPI backend (Phase B) implements the same interface (`train`, `onProgress`, `cancel`), making browser and backend interchangeable.
**Warning signs:** Any `import * as tf` outside `/workers` or `/ml`; a Chrome Performance trace with long tasks >50 ms during training; the "Stop" button not responding.
**Phase to address:** F (architecture), enforced in every later phase via a lint rule (`no-restricted-imports`).

---

### Pitfall 2: Progress streaming that floods the main thread or React
**What goes wrong:** A `postMessage` per batch (hundreds/sec) plus a `setState` per message re-renders charts continuously; the UI becomes the bottleneck, not the model. Unbounded loss-history arrays grow forever and slow every re-render and structured clone.
**Why it happens:** "Live" is interpreted literally. Recharts (SVG) re-renders the entire series on each append.
**How to avoid:** Worker batches metrics and posts at most ~10-20 Hz (time-based throttle, not per-epoch). UI buffers into a ref/external store (Zustand with selectors, or `useSyncExternalStore`) and flushes in `requestAnimationFrame`. Cap history (ring buffer or LTTB downsampling to ~1-2k points). Use canvas or uPlot-style rendering for the live chart; keep Recharts for static/after-run charts. Send typed arrays as Transferables, not arrays of objects.
**Warning signs:** Fan spin-up while training a tiny model; React Profiler shows the chart component committing >30x/sec; training is slower with the chart visible than hidden.
**Phase to address:** F (progress protocol), S (first live chart), V (chart rendering strategy).

---

### Pitfall 3: Data leakage from preprocessing order (the wizard-order trap)
**What goes wrong:** The guided flow is clean, EDA, feature-engineer, then split, so imputation means, outlier caps, scalers, encoders, TF-IDF vocabularies, PCA, and feature selection are fitted on the whole dataset before the split. Test metrics are optimistically biased and the product's core promise ("honest test results") is false.
**Why it happens:** Intuitive UX order is the leaky order. Each cleaning step is implemented as an eager "mutate the dataset" operation, so there is no notion of "fit on train, apply to test".
**How to avoid:** Architect transforms as a declarative, serializable **recipe/pipeline** of steps with `fit(train)` and `transform(any)` semantics (mirrors sklearn `Pipeline`). The split is a first-class object created early (freeze it right after type detection, with a stored seed/row-index hash) and every stateful step (imputer, scaler, encoder, outlier thresholds, feature selection, vectorizer, PCA, SMOTE) fits on the train partition only. Stateless row-level fixes (dtype fixes, dedupe, trimming) may precede the split; but dedupe must happen before the split so duplicates don't straddle it. EDA may show whole-data stats but label them "full dataset (exploration only)". The before/after cleaning view should show train-fitted effects. Add a unit test: fit on a dataset where the test partition has an extreme outlier and assert the scaler's mean/std is unchanged by it.
**Warning signs:** `scaler.fit(allRows)` anywhere; the transform step objects have no `fit` vs `transform` distinction; test accuracy suspiciously equals or exceeds train accuracy; a feature with near-perfect importance (ID, target proxy).
**Phase to address:** F (data model: Dataset, Split, Recipe), C and FE (implementation), S (verification test).

---

### Pitfall 4: Target leakage, ID columns and test-set reuse
**What goes wrong:** Users pick a target and leave a leaking column in (`outcome_text`, `customer_id`, row index, a column derived from the target); accuracy hits 99% and the playground "teaches" the wrong lesson. Separately, hyperparameters or early stopping are tuned against the test set, silently contaminating it.
**Why it happens:** Playgrounds default to "all other columns are features". Showing test metrics live every epoch invites tuning on them.
**How to avoid:** Three-way split (train/validation/test) or CV for tuning; test set is evaluated once per run and clearly labelled. Live curves show train vs **validation**, not test. Heuristic warnings at feature-selection: near-unique-per-row columns (likely IDs), columns with |corr| > 0.95 or single-feature AUC > 0.98 with target, monotonic index-like columns. Default-exclude ID-like columns with a visible toggle. Group-aware or time-aware split options (no shuffle for time-ordered data; warn if a date column exists and split is random).
**Warning signs:** Single feature with >0.9 importance; test score improving as you tweak hyperparameters repeatedly; a date column present with random split.
**Phase to address:** FE (split controls and warnings), S (live curve semantics, evaluation).

---

### Pitfall 5: Misleading metrics (accuracy on imbalanced data, R^2 surprises, no baseline)
**What goes wrong:** 95% accuracy on a 95/5 dataset from a model that predicts the majority class; ROC-AUC looks great while precision on the minority class is tiny; R^2 shown as a percentage and confusing when negative; a single small test split presented as certainty.
**Why it happens:** Accuracy is the default and is easy to render; baselines and uncertainty cost extra work.
**How to avoid:** Always show a **baseline** next to every result (majority class / mean predictor, computed by the same pipeline). Task-aware default metric panel: classification shows per-class precision/recall/F1, balanced accuracy, confusion matrix, PR curve (primary when imbalance ratio > ~3:1) plus ROC; regression shows MAE/RMSE/R^2 plus residual plot. Auto-detect imbalance in EDA and surface a banner that changes the default metrics. Report test-set size and a bootstrap/CI band when n_test < ~500. Multi-class: state the averaging mode (macro vs weighted). Never show "accuracy" as the sole headline.
**Warning signs:** Headline number with no baseline; confusion matrix where one column is all zeros; metric code that returns NaN for a class with no support (division by zero in per-class F1).
**Phase to address:** S (evaluation module), C (class balance in EDA), X (comparison must use the same metrics + baseline).

---

### Pitfall 6: Imbalanced-data handling done wrong
**What goes wrong:** Random shuffle split drops all minority examples from test (or train); SMOTE/oversampling applied before the split so synthetic neighbours of test points leak into train; class weights silently ignored by some JS implementations; stratification absent.
**Why it happens:** Resampling is presented as a data-prep step alongside cleaning.
**How to avoid:** Stratified split by default for classification (error if a class has < 2 samples, with a clear message); resampling/class-weights are **model-training options applied to the train partition only**, inside the pipeline. Verify each model implementation actually honours class weights (test with a synthetic 99:1 dataset). Preview "class counts in train/test" in the split UI.
**Warning signs:** Test class counts missing a class; minority recall exactly 0; resampling step positioned before the split step.
**Phase to address:** FE (stratified split), S (class weights and resampling inside pipeline).

---

### Pitfall 7: Browser memory blowup on "any dataset"
**What goes wrong:** `FileReader.readAsText` of a 300 MB CSV, then `split('\n')`, then array-of-objects rows: memory multiplies 10-30x and the tab crashes with no recovery. Excel is worse (a `.xlsx` is a zip; parsers inflate and build whole-workbook object graphs). Image datasets decoded to float32 tensors (224x224x3 = ~600 KB each; 10k images ~ 6 GB). TF.js tensor leaks accumulate silently across runs.
**Why it happens:** "Upload anything" promise vs. a ~2-4 GB per-tab heap ceiling (less on laptops/mobile) and JS row-objects being ~10x heavier than columnar typed arrays.
**How to avoid:**
- Parse in the worker with streaming (`File.stream()` / Papa Parse chunked or `worker: true`), infer types, and store **columnar typed arrays** (`Float64Array`/`Int32Array` + dictionary-encoded strings + null bitmaps), not row objects. Consider Arquero/Apache Arrow/`danfojs` only if they don't force row objects.
- Hard, visible limits with graceful routing: e.g., > ~100-200 MB or > ~1M rows prompts "use the Python backend, or sample N rows"; offer reservoir sampling for preview/EDA.
- Images: load lazily, resize on decode to a small target (e.g., 32-64 px for the playground), use `createImageBitmap` with resize options, process in batches via `tf.data`/generators, never materialize all as one tensor.
- TF.js: wrap per-step ops in `tf.tidy`, dispose models/optimizers when a run ends, assert `tf.memory().numTensors` returns to baseline after each run in a dev-mode check; terminate and recreate the worker between runs as the nuclear option.
- Persist the dataset once (by content hash) and reference it from experiments.
**Warning signs:** Tab memory (Chrome Task Manager) grows after each training run; "Aw, Snap" on a mid-size file; `tf.memory().numTensors` climbing monotonically.
**Phase to address:** I (streaming columnar ingest and limits), FE (image tensors), F (memory instrumentation), B (overflow route).

---

### Pitfall 8: Putting the dataset in React/global state
**What goes wrong:** The full parsed dataset lives in Zustand/Context/Redux; every update triggers deep comparisons, DevTools serialization, and structured clones when posting to workers. Typing in a filter box lags.
**Why it happens:** It is the easiest place for every component to read from.
**How to avoid:** The worker owns the data (a `DatasetHandle` id). React state holds only small derived artifacts: schema, 100-row preview, column stats, histograms. Use `postMessage` with transferables or SharedArrayBuffer only if you accept the COOP/COEP requirement (see Integration Gotchas). Virtualize tables (TanStack Virtual) - never render >~200 DOM rows.
**Warning signs:** Re-render counts proportional to row count; DevTools freezing on open; slow scrolling in the data table.
**Phase to address:** F (state architecture), I (preview and virtualization).

---

### Pitfall 9: Next.js SSR and client-only library import failures
**What goes wrong:** `ReferenceError: window/self/document is not defined` at build or first request from TF.js, three.js, Plotly/D3 helpers, SheetJS, or IndexedDB wrappers imported at module top level in a component that renders on the server. Hydration mismatches from locale-dependent numbers/dates, `Math.random()` ids, or reading `localStorage` in render. Adding `'use client'` alone does **not** prevent server pre-render, so the module still executes in Node.
**Why it happens:** App Router renders Client Components on the server for the initial HTML; developers think `'use client'` means "browser only". In the App Router, `next/dynamic({ ssr: false })` is only permitted from a Client Component, so it needs a thin wrapper, and a static import of the same library elsewhere still runs server-side.
**How to avoid:** Pattern: server `page.tsx` (shell only) -> `'use client'` wrapper -> `dynamic(() => import('./Workbench'), { ssr: false })`. Keep **all** ML/3D/parsing imports behind that boundary or inside workers (workers are never server-rendered, so they are the safest home for TF.js). Never polyfill `window` on the server. Gate browser APIs in `useEffect` or `typeof window` checks; render deterministic markup on first pass and format numbers/dates after mount (or with explicit locale + timezone). Add a CI step: `next build` plus a smoke test that loads the main routes with JS disabled/SSR-only to catch server-side crashes.
**Warning signs:** Error only in `next build`/production but not dev; "self is not defined" from a transitive dependency; hydration warning in console.
**Phase to address:** F (establish the boundary pattern and CI check), then every UI phase.

---

### Pitfall 10: React StrictMode double-mount starting two workers / two trainings / two WebGL contexts
**What goes wrong:** In dev, Next's StrictMode mounts, unmounts and remounts effects; an effect that spawns a Worker, starts training, or creates a renderer without cleanup yields duplicate workers, double-counted metrics, leaked GPU contexts, and "works in prod, buggy in dev" (or the inverse). Navigating between routes also orphans workers that keep training.
**Why it happens:** Imperative resources are created in `useEffect` without symmetric teardown, or created at module scope and shared accidentally.
**How to avoid:** One worker-pool singleton owned outside React (lazy, reference-counted) and an explicit job lifecycle: `start` is triggered by user action (event handlers), not effects; effects only subscribe/unsubscribe. On unmount: unsubscribe, and `cancel` or `terminate` if the run belongs to that view. Dispose three.js geometries/materials/textures and `renderer.dispose()` in cleanup. Keep StrictMode on in dev - it is the detector.
**Warning signs:** Two "training started" logs per click in dev; multiple workers in DevTools Sources; GPU memory rising on route changes.
**Phase to address:** F (worker pool and lifecycle), V (renderer cleanup).

---

### Pitfall 11: Worker bundling, WASM asset paths and cross-origin isolation
**What goes wrong:** Worker file not found or not bundled correctly; TF.js WASM backend or onnxruntime-web fails to fetch `.wasm` files (404, wrong MIME, wrong base path after deploy); `SharedArrayBuffer`/multi-threaded WASM unavailable because the page isn't cross-origin isolated; enabling COOP/COEP then breaks third-party embeds, CDN fonts and images.
**Why it happens:** Bundler-specific asset handling (webpack vs Turbopack), static-export vs server deployment differences, and WASM runtimes resolving paths relative to the script URL.
**How to avoid:** Use the standard `new Worker(new URL('./compute.worker.ts', import.meta.url), { type: 'module' })` form. Set backend asset paths explicitly (`setWasmPaths`, `ort.env.wasm.wasmPaths`) pointing at files copied into `/public`; verify in a production build, not just `next dev`. Decide early whether to require cross-origin isolation: if multi-threaded WASM is wanted, set COOP/COEP in `next.config` headers and self-host all fonts/assets; otherwise accept single-threaded WASM and design around it. Test both `next dev` and `next build && next start`. (Worker bundling specifics were not confirmed by search; verify against current Next.js docs when implementing.)
**Warning signs:** Works in dev, 404 on `.wasm` in prod; `crossOriginIsolated === false`; console message about falling back to a single thread.
**Phase to address:** F (spike: worker + wasm + build verified end to end before any feature is built on it).

---

### Pitfall 12: Assuming TF.js (or ONNX) can run "any kind of model"
**What goes wrong:** The requirement list contains random forests, SVM, kNN, DBSCAN, hierarchical clustering, t-SNE/UMAP. TF.js is a neural-network/tensor library and has no first-class implementations of these; ONNX Runtime Web is primarily an **inference** runtime (in-browser training is limited/experimental). Teams discover this mid-build and either hand-roll buggy algorithms or quietly drop the model.
**Why it happens:** "TF.js/ONNX in browser" in the stack line is read as full ML coverage.
**How to avoid:** Decide the engine per algorithm family up front in a Phase F/S research spike:
- Neural nets, linear/logistic regression, DQN -> TF.js (or hand-written typed-array code for linear models, which is simpler and faster for small data).
- Trees/forests/kNN/k-means/DBSCAN/PCA/t-SNE/UMAP -> vetted JS libraries (`ml-*` family, `umap-js`, `tsne-js` variants) **or** Pyodide (heavy download) **or** route to FastAPI/scikit-learn.
- Make the engine an internal `Estimator` interface (`fit/predict/predictProba/getParams`) so a model can be moved between browser and backend without touching UI.
Validate every hand-rolled or JS-library algorithm against **golden fixtures generated with scikit-learn** (same data, same seed, tolerance check) - small JS ML libraries often have subtle bugs.
**Warning signs:** A model implemented in "about 80 lines"; no reference test; results differ visibly from sklearn on iris/wine.
**Phase to address:** F (engine decision spike), S and U (implementation + golden tests), B (parity).

---

### Pitfall 13: TF.js backend selection and numerical surprises (webgl / wasm / webgpu / cpu)
**What goes wrong:** Default backend picked implicitly; on machines/browsers without usable WebGL training silently falls to CPU and is 10-100x slower; the WebGPU backend is not available everywhere (browser/OS coverage and TF.js's WebGPU op coverage must be re-verified at implementation time); WebGL float precision (float16 on some mobile GPUs) yields NaNs and divergent loss; `setBackend` called after ops started or switched mid-session strands tensors; WebGL in a worker requires OffscreenCanvas support; tiny models are *faster* on WASM/CPU than WebGL because of upload overhead; results differ slightly across backends so "reproducible" runs are not.
**Why it happens:** Backend is an environment property treated as an implementation detail.
**How to avoid:** Explicit init in the worker: `await tf.setBackend(chosen); await tf.ready()` with a capability probe and fallback order (webgpu -> webgl -> wasm -> cpu) selected by a table of "model size/data size vs backend", and surface the active backend in the UI (users should see "Running on: WebGL"). Never switch backend while tensors exist; pick once per worker lifetime (new worker to change). Use WASM for small/tabular nets, GPU backends for conv nets. Detect NaN loss and stop with an actionable message. Treat TF.js as an **isolated dependency behind the Estimator interface**: project maintenance cadence is uncertain (one unverified low-reliability post claims it was wound down in favour of LiteRT.js; npm health checks show recent releases for 4.22) - do not let TF.js types leak across the codebase so a swap is feasible. Verify current status before committing to it.
**Warning signs:** Training 50x slower on a laptop without discrete GPU; `tf.getBackend()` returns `cpu`; loss turns `NaN` on one device only.
**Phase to address:** F (backend probe + fallback), S (neural-net models), RL (DQN speed).

---

### Pitfall 14: WebGL context loss and too many canvases (three.js + TF.js + charts)
**What goes wrong:** `CONTEXT_LOST_WEBGL`, black canvases, or "too many active WebGL contexts" warnings (browsers cap contexts ~8-16 and evict the oldest) when each chart/3D scene mounts its own `<Canvas>`; TF.js's WebGL backend and three.js compete for the same GPU so training starves rendering (or vice versa); memory leaks from undisposed geometries/textures; rendering 100k points as individual meshes.
**Why it happens:** One R3F `<Canvas>` per widget is the natural component design; GPU is shared invisibly with the training workload.
**How to avoid:** One shared canvas for the whole app (drei `<View>` tracking pattern) or at most one active 3D scene at a time with lazy mount/unmount; handle `webglcontextlost`/`webglcontextrestored` (show "Reset view" instead of a blank box; rebuild from state); `frameloop="demand"` and invalidate only on data change; `InstancedMesh`/`Points` with `BufferGeometry` for scatter data; subsample for display; dispose on unmount. For TF.js, prefer WASM backend or pause rendering (reduce fps) while a GPU-backend training job is hot. Test by opening 10+ visualizations in succession and by putting the machine under GPU load.
**Warning signs:** Console "Too many active WebGL contexts. Oldest context will be lost."; FPS dropping to single digits while training; VRAM climbing after navigating between views.
**Phase to address:** V (shared canvas architecture, before building many 3D views), RL (agent animation), F (decision on GPU use for training vs render).

---

### Pitfall 15: Hand-rolled CSV/Excel/JSON parsing edge cases
**What goes wrong:** Mis-parsed files that look "fine" in preview: BOM glued to the first header (`<BOM>id`); `;` or tab delimiters and decimal commas (`1,5`) from European exports; Windows-1252/UTF-16 encodings mangling accents; quoted fields containing newlines; ragged rows; duplicate or empty headers; type inference from only the first N rows (late text value turns a numeric column into garbage); leading zeros stripped (zip codes, IDs); integers above 2^53 losing precision; missing-value tokens beyond empty (`NA`, `N/A`, `null`, `-`, `?`, `#N/A`); currency/percent strings (`$1,200`, `12%`); ambiguous dates (`03/04/2024`) and Excel serial dates (plus the 1900 leap-year quirk); Excel multiple sheets, merged cells, header offset, formulas vs cached values; JSON as array-of-objects vs NDJSON vs nested; whole-column-object keys colliding with `__proto__`.
**Why it happens:** Parsing is perceived as solved; test files are clean Kaggle CSVs.
**How to avoid:** Use a robust parser (Papa Parse with `worker`/`step`, delimiter auto-detect; `jschardet`/TextDecoder with detected encoding) instead of splitting on commas. Build a **schema-confirmation step**: show detected delimiter, encoding, header row, and per-column inferred type with confidence and a per-column override, with a stated-assumptions banner for ambiguous dates and missing tokens. Infer types over a large sample (or the full column in a streaming pass), keep ID-like text as string. Excel: let the user pick the sheet and header row; read cached values; warn on merged cells; guard against huge sparse ranges (`!ref` of A1:XFD1048576). Flatten nested JSON with an explicit, previewed rule. Use `Map` or `Object.create(null)` keyed by column names. Keep a **fixture suite of ugly files** (BOM, semicolons, cp1252, ragged, embedded newlines, Excel dates, 1M-row sparse sheet).
**Warning signs:** Numeric column typed as string because of one `N/A`; first column named `<BOM>id`; row count differs from source file by a few; all dates in 1900/1970.
**Phase to address:** I (and its fixture suite); C (dtype-fix tooling relies on these).

---

### Pitfall 16: Upload and spreadsheet security (client-side-only does not mean risk-free)
**What goes wrong:**
- **SheetJS on npm is stale** (public npm stops at 0.18.5 with known prototype-pollution and ReDoS advisories; current builds are distributed from cdn.sheetjs.com). Installing `xlsx` from npm ships known-vulnerable code. (HIGH on advisory existence; verify current distribution before choosing.)
- Zip/decompression bombs: `.xlsx` and any zipped image datasets expand to GBs.
- XSS through data: column names/cell values/labels rendered via `dangerouslySetInnerHTML`, D3 `.html()`, tooltips, or unsanitized SVG uploads.
- CSV/formula injection on **export**: cells starting with `=`, `+`, `-`, `@` execute when the exported CSV is opened in Excel.
- Prototype pollution via header names (`__proto__`, `constructor`) when building objects.
- Importing a saved-experiment JSON or model file without validation.
**Why it happens:** Local-only apps feel safe; the attack surface is *the files the user opens from elsewhere* and the data they re-share.
**How to avoid:** Prefer a maintained parser (ExcelJS/`read-excel-file`, or SheetJS from its official CDN tarball pinned by hash) and cap uncompressed size / row / cell counts before parsing; parse in a worker. Render all user strings as text (React default); never `innerHTML`; show uploaded images only via `<img>`/canvas, never inline SVG. Prefix formula-leading cells with `'` on CSV/Excel export. Validate imported experiment files with Zod and a version field; reject unknown shapes. Add a Content-Security-Policy (no `unsafe-eval` unless a dependency strictly needs it - note WASM may need `wasm-unsafe-eval`).
**Warning signs:** `npm audit` flags `xlsx`; a dataset column named `<img onerror=...>` fires an alert; exported CSV opens a prompt in Excel.
**Phase to address:** I (parser choice and limits), X (import/export validation), F (CSP).

---

### Pitfall 17: FastAPI backend exposed like a web service (localhost is not private)
**What goes wrong:**
- Any website the user visits can send requests to `http://localhost:8000` (CSRF/DNS rebinding); with `allow_origins=["*"]` and no auth, a malicious page can run jobs or read results.
- Unpickling or `joblib.load` of uploaded/imported models = remote code execution; `eval`-style "custom model code" features likewise.
- Binding to `0.0.0.0` exposes the user's machine to the LAN.
- Browser constraints: an HTTPS-hosted frontend calling an HTTP localhost backend hits mixed-content/private-network rules (Chrome has been tightening "local network access" with permission prompts; Safari/Firefox differ) - re-verify current behaviour.
- Uploads proxied through Next.js Route Handlers hit body-size limits (e.g., ~4.5 MB on serverless) and double-buffer memory.
**Why it happens:** "Optional local helper" gets built like a hosted API.
**How to avoid:** Bind to `127.0.0.1` only; strict CORS allow-list of the app origin(s); a per-launch random token printed in the terminal and entered/pasted into the UI (sent as `Authorization`); no pickle/joblib import of user files (export results as JSON/ONNX instead); never accept code strings; upload directly browser -> FastAPI (not via Next.js), with size caps and temp-file cleanup; sanitize filenames. Health/version endpoint so the UI can detect "backend not running / incompatible version" and degrade gracefully. Document the exact one-line start command for Windows and macOS/Linux, and test the whole path on Windows (the dev machine).
**Warning signs:** `allow_origins=["*"]`; `--host 0.0.0.0` in docs; `pickle.load` anywhere; UI has no "backend offline" state.
**Phase to address:** B (before any endpoint ships), F (define the shared job interface).

---

### Pitfall 18: "Data stays local" privacy promise silently violated
**What goes wrong:** Analytics/error trackers capture dataset snippets in error payloads; Google Fonts/CDN scripts leak usage; pretrained models fetched at runtime from third-party hosts; backend mode uploads data without a clear indicator.
**Why it happens:** Default Next.js templates and monitoring tools phone home; "optional backend" blurs the boundary.
**How to avoid:** Self-host fonts (`next/font`), no analytics by default, scrub error reports, a persistent visible "Data location: this browser / local backend" indicator, explicit confirmation when switching to backend execution, bundled or user-approved model downloads with sizes shown.
**Warning signs:** Network tab shows third-party requests after loading a dataset; a toast or error includes row values.
**Phase to address:** F (defaults and CSP), B (opt-in UX).

---

### Pitfall 19: RL training instability and misleading reward curves
**What goes wrong:** DQN diverges or flatlines (no target network, no replay buffer, wrong learning rate, unscaled rewards, epsilon decays too fast); single-run reward curves are mostly noise yet presented as "the agent learns"; Q-learning on gridworld never converges because of step-penalty/discount misconfiguration or `done` vs `truncated` confusion; CartPole physics subtly different from Gym so results aren't comparable; training speed coupled to animation frame rate (60 steps/s ceiling) or conversely animation blocked by training; non-seedable `Math.random` makes runs irreproducible; DQN's TF.js tensors leak per step.
**Why it happens:** RL is high-variance by nature; playgrounds want an immediate "wow" and tune until one lucky run works.
**How to avoid:** Decouple the **simulation/training loop (worker, as-fast-as-possible, headless)** from the **replay/animation (UI, at human speed, of recorded or current-policy episodes)**; a speed slider and "train 100 episodes now" control. Implement a seeded PRNG (mulberry32/sfc32) passed through env, exploration, replay sampling and weight init; show seed in the UI. Implement CartPole with the exact Gym constants and Euler integration and termination thresholds, test against known trajectories. Use standard DQN components (replay buffer, target net with periodic sync, Huber loss, epsilon schedule, gradient clip), small networks, train every N env steps, and `tf.tidy` every step. Display reward as raw + moving average, optionally multiple seeds with mean +/- std; handle terminated vs truncated explicitly. Provide solid defaults per environment (a "known-good preset") and show what each hyperparameter does. Include tabular Q-learning/SARSA first - deterministic and fast - then DQN.
**Warning signs:** Reward curve flat after thousands of episodes; episode length hitting the cap every time; Q-values growing without bound or NaN; results change run to run with identical settings (expected) but there is no seed to reproduce.
**Phase to address:** RL (own phase; run a research spike on DQN-in-TF.js stability before planning it).

---

### Pitfall 20: Unscaled features, NaN/Inf and high-cardinality encodings crashing training
**What goes wrong:** kNN/SVM/NN/k-means/DBSCAN on unscaled mixed-unit features give nonsense; a single NaN/Inf makes loss `NaN` without an error; one-hot on a 10k-unique column creates 10k columns and blows memory; categories unseen in train appear in test and break the encoder; label encoding used on nominal data implies fake order; DBSCAN `eps` guessed on unscaled data.
**Why it happens:** Preprocessing is user-optional and failures are numeric rather than exceptions.
**How to avoid:** Per-model preprocessing requirements metadata (e.g., kNN/SVM/NN/k-means/PCA "requires scaling"); preflight validation before any run that blocks or auto-applies (with disclosure) scaling and rejects NaN/Inf with a pointer to the offending columns; cap one-hot cardinality with "top-K + other" and a warning; encoders handle unseen categories (ignore/other) at transform time; runtime NaN-loss guard stops the run with a message.
**Warning signs:** Loss `NaN` at epoch 1; memory spike when selecting a text column as a feature; accuracy equals majority-class baseline.
**Phase to address:** FE (encoders, scalers, preflight), S and U (model requirement metadata).

---

### Pitfall 21: Expensive visualizations quietly dominating compute (decision boundaries, t-SNE, silhouette)
**What goes wrong:** Decision-boundary plots predict a 200x200 grid (40k predictions) every epoch; t-SNE/UMAP/silhouette/hierarchical clustering are O(n^2) or worse and freeze at a few thousand points; boundaries drawn for datasets with >2 features without saying what was projected; t-SNE cluster size/distance over-interpreted by learners.
**Why it happens:** Visuals look cheap compared to "the real model".
**How to avoid:** Decision-boundary view only for 2 selected features (or a clearly labelled PCA-2D projection), at coarse resolution during training and fine resolution after; update every N epochs, in the worker, as a typed array rendered to a texture. Subsample (e.g., <= 2-5k points) for t-SNE/UMAP/silhouette with a visible "N of M points" note, seeded, with perplexity/neighbours explained; a "caveats" tooltip on t-SNE. Cap hierarchical clustering at a few thousand rows and say why. Gate algorithm/data-size combos with estimated runtime.
**Warning signs:** Epoch time dominated by plotting, not training; t-SNE run has no progress/cancel.
**Phase to address:** U (embeddings, clustering caps), V (boundary rendering), S (boundary updates).

---

### Pitfall 22: Unfair or unreproducible model comparison
**What goes wrong:** Models compared with different splits, seeds, preprocessing, or tuning effort; browser-JS model vs sklearn backend model compared as if they were the same algorithm; no record of what produced a run, so the "comparison" can't be reproduced.
**Why it happens:** Comparison is built last, on top of whatever each run happened to store.
**How to avoid:** A Run record stores: dataset hash, recipe version, split (seed + row indices hash), model id + params, seed, engine (browser/backend + library version), metrics + baseline. The comparison view groups only runs sharing the same dataset+split+target and warns (or refuses) otherwise. Include timing and engine columns. Define this schema in Phase F even though the UI comes later.
**Warning signs:** Can't answer "which split was this run on?"; comparison table mixes tasks (classification and regression) or targets.
**Phase to address:** F (Run schema), X (comparison UI).

---

### Pitfall 23: Local persistence (IndexedDB) quota, eviction and schema drift
**What goes wrong:** Datasets stored inside every experiment multiply storage use; quota errors (`QuotaExceededError`) lose the in-progress run; Safari may evict script-writable storage after inactivity; model weights can't be structured-cloned from tensors; saved experiments from v1 break after the recipe/schema changes; exported files can't be re-imported by a later version.
**Why it happens:** "Save to IndexedDB" feels trivial; no migration story because there is no backend.
**How to avoid:** Store datasets once by content hash and let experiments reference them; request `navigator.storage.persist()` and show usage via `storage.estimate()`; handle quota errors with an export-now prompt; serialize weights as `Float32Array` buffers or via `model.save('indexeddb://...')`; a `schemaVersion` on every stored/exported object plus migration functions from day one (Dexie versioning); autosave run metadata so a tab crash doesn't lose results.
**Warning signs:** Storage usage grows by the dataset size per saved run; import of last week's export fails.
**Phase to address:** F (storage layer and versioning), X (export/import).

---

### Pitfall 24: Over-complex UI that hides the learning
**What goes wrong:** Every hyperparameter shown at once; a 12-step wizard where users can't tell where they are or go back; algorithm choices that make no sense for the data (regressor on a categorical target); empty/failed states unexplained; 3D and animation for their own sake slow the app and obscure the lesson; canvas/3D charts inaccessible; color-only encodings; motion ignores `prefers-reduced-motion`.
**Why it happens:** "Full scope in v1" and "any kind of model" push towards exposing the whole feature surface.
**How to avoid:** Progressive disclosure: a **Guided** mode with sensible defaults and a one-line explanation per choice, an **Advanced** drawer for hyperparameters. Task-type auto-detection (from target dtype/cardinality) with override and a compatibility filter that hides/disables invalid model-metric combos with a reason. Persistent stepper with completion state, undo/back, and "what changed" views. Each visualization gets a plain-language caption. Design-system checks (ui-ux-pro-max) for contrast, keyboard navigation, reduced motion, and colour-blind-safe palettes (non-color cues, e.g., shapes/patterns), plus tabular fallbacks for key charts. 3D only where the third dimension carries information (e.g., 3-feature scatter, loss surface, PCA-3D), 2D otherwise.
**Warning signs:** Usability test users can't say what to do next; > ~6 controls visible per step by default; Lighthouse accessibility < 90; users don't notice the test results.
**Phase to address:** F (design system and stepper), every feature phase (defaults), X (polish and accessibility audit).

---

### Pitfall 25: Stop/cancel and error handling that doesn't actually stop or surface failure
**What goes wrong:** "Stop" only sets a flag the loop checks between epochs, so a long epoch/t-SNE can't be interrupted; worker exceptions vanish (no `onerror`/`onmessageerror` handling); tab OOM leaves the UI in "training..." forever; background-tab timer throttling stalls main-thread-driven progress loops.
**Why it happens:** Happy-path demos.
**How to avoid:** Cooperative cancel for graceful stops **and** `worker.terminate()` as the hard path (then respawn); every job has timeout/heartbeat so the UI shows "worker unresponsive"; global worker error boundary mapping to user-readable messages; drive progress from worker timers (not main-thread `setInterval`); persist last-known state.
**Warning signs:** No "Stop" test in the suite; "training" state stuck after a worker crash.
**Phase to address:** F (job lifecycle), re-verified in S, U, RL.

---

### Pitfall 26: Bundle weight and cold-start (TF.js + ORT WASM + three.js + Plotly-class charts)
**What goes wrong:** Several MB of JS plus 10+ MB of WASM downloads on the landing page; slow first paint; users bounce before seeing anything.
**Why it happens:** Eager imports and single-route architecture.
**How to avoid:** Route-level code splitting; load ML engines inside the worker on demand (first training click) with a visible "Loading engine (x MB)" state and caching; marketing/landing page ships no ML code; analyse with `@next/bundle-analyzer` in CI with a budget.
**Warning signs:** Initial JS > ~300 KB gzip on the landing route; WASM requested before user starts a run.
**Phase to address:** F (budget + analyzer), V and S (lazy loading).

---

## Technical Debt Patterns

| Shortcut | Immediate Benefit | Long-term Cost | When Acceptable |
|----------|-------------------|----------------|-----------------|
| Row-object datasets (`[{col: val}]`) | Easy to code and debug | 10x memory, slow transforms, kills "large file" goal | Only for the <=100-row preview slice |
| Calling TF.js directly from components | Fast demo | Main-thread freezes, impossible worker retrofit, backend swap impossible | Never |
| Eager "mutate dataset" cleaning steps | Simple before/after | Leakage; no reproducibility; no undo | Never (use recipe + fit/transform) |
| Single random split with no stored seed | Less UI | Unreproducible comparisons | Never; store seed + indices hash |
| Hand-rolled tree/SVM/DBSCAN with no reference tests | Zero deps | Silent wrong answers teaching wrong lessons | Only with sklearn golden fixtures |
| One `<Canvas>` per visualization | Natural component model | Context loss at ~8-16 canvases | Prototype only |
| `Math.random()` everywhere | Convenience | No reproducibility (esp. RL and splits) | Never in ML paths; use a seeded PRNG |
| Storing full dataset copy per experiment | Easy export | Quota exhaustion | Only for explicit "export bundle" |
| Skipping schema versions on saved/exported JSON | Faster v1 | Broken imports after first model/schema change | Never |
| Recharts for live training curves | Already in stack | Re-render storms at high update rates | Fine for post-run/static charts; use canvas for live |

## Integration Gotchas

| Integration | Common Mistake | Correct Approach |
|-------------|----------------|------------------|
| TF.js in Next.js | Top-level `import * as tf` in a component rendered on the server | Import inside the worker (or behind `dynamic(..., { ssr: false })` in a client wrapper) |
| TF.js WASM / ORT-web | Relying on default `.wasm` path; works in dev, 404 in prod | Copy assets to `/public`, call `setWasmPaths` / `ort.env.wasm.wasmPaths`, test a prod build |
| SharedArrayBuffer / WASM threads | Using SAB without COOP/COEP headers | Set `Cross-Origin-Opener-Policy: same-origin` + `Cross-Origin-Embedder-Policy: require-corp` and self-host assets, or avoid SAB |
| react-three-fiber in App Router | R3F `<Canvas>` in a Server Component or statically imported in a server-rendered tree | `'use client'` wrapper + `dynamic({ ssr: false })` |
| FastAPI from browser | `allow_origins=["*"]`, no auth, bound to 0.0.0.0 | 127.0.0.1, origin allow-list, per-launch token, version/health endpoint |
| FastAPI long jobs | Synchronous endpoint blocks and times out; polling hammered | Background job + SSE/WebSocket progress stream, cancel endpoint, same job protocol as in-browser worker |
| FastAPI file upload | Proxy through Next.js route handler | Upload browser -> FastAPI directly (size cap, temp-file cleanup) |
| Browser-vs-backend parity | Different preprocessing/algorithm implementations compared as equals | Single recipe spec executed by both, record engine + library version, label "browser (JS)" vs "backend (scikit-learn)" |
| SheetJS | `npm i xlsx` (stale, vulnerable on public registry) | Official CDN tarball or an alternative parser; cap sizes |
| IndexedDB | Raw API/Dexie without versions; tensors stored directly | Versioned schema, serialize typed arrays, `storage.persist()` |
| Framer Motion + R3F | Animating React state at 60 fps alongside canvas | Animate UI chrome with Framer; drive canvas via refs/`useFrame`; respect `prefers-reduced-motion` |

## Performance Traps

| Trap | Symptoms | Prevention | When It Breaks |
|------|----------|------------|----------------|
| Per-batch `postMessage` + `setState` | UI jank during "live" training | Throttle ~10-20 Hz, rAF flush, ring buffer | Any model with >50 batches/s |
| SVG charts for large series | Hover lag, high CPU | Downsample (LTTB), canvas/WebGL charts | > ~2-5k points per series |
| Rendering table rows in the DOM | Slow scroll, memory | Virtualization | > ~500 rows |
| Row-object parsing of big CSVs | Tab crash | Streaming + columnar typed arrays | > ~50-100 MB files / ~1M rows |
| Image dataset as one tensor | OOM | Resize on decode, batched generator | > ~2-5k images at 224 px |
| t-SNE/UMAP/silhouette/hierarchical on all rows | Frozen worker | Subsample + cap + progress/cancel | > ~2-5k points |
| kNN predict O(n*m) with decision-grid | Boundary view takes seconds | Coarse grid while training, KD-tree/ball-tree or subsample | > ~10k train rows with 40k grid points |
| One-hot of high-cardinality column | Memory spike | Top-K + other, hashing | > ~500-1000 categories |
| Per-step tensors not disposed (DQN) | Slow memory growth, eventual crash | `tf.tidy`, dispose, assert `numTensors` | Thousands of steps |
| One WebGL context per widget | Blank canvases | Shared canvas / single active scene | > ~8-16 contexts |
| Training on WebGL while rendering 3D | Both stutter | Prefer WASM for small nets; throttle render during training | Any integrated GPU |

## Security Mistakes

| Mistake | Risk | Prevention |
|---------|------|------------|
| `xlsx` from public npm | Known prototype pollution / ReDoS advisories | Official CDN build pinned, or alternative lib; size caps |
| Rendering cell/column text as HTML | XSS from crafted datasets | Text-only rendering; no `innerHTML`/D3 `.html()` with user data |
| Exported CSV with unescaped `=`/`+`/`-`/`@` cells | Formula injection when opened in Excel | Prefix with `'` or tab on export |
| Object keys from headers (`__proto__`) | Prototype pollution | `Map` / `Object.create(null)` |
| Loading uploaded pickle/joblib in backend | Remote code execution | Never; accept only JSON/ONNX/CSV |
| Wildcard CORS + no token on local backend | Any website drives the user's local service | Origin allow-list, per-launch token, 127.0.0.1 bind |
| Unvalidated imported experiment JSON | Crash or injection into stored state | Zod schema + version check |
| Zip/XLSX decompression bombs | Tab or backend memory exhaustion | Inspect and cap uncompressed size/entry count before parsing |
| Uploaded SVG rendered inline | Script execution | Display via `<img>` only; or reject SVG |
| Third-party scripts/fonts/analytics | Breaks "data stays local" claim | Self-host, strict CSP, no telemetry on dataset content |
| Filenames used in backend paths | Path traversal | Generate server-side IDs; ignore client filename |

## UX Pitfalls

| Pitfall | User Impact | Better Approach |
|---------|-------------|-----------------|
| Accuracy as the headline metric | Learner concludes a useless model is great | Baseline + task-aware metric panel + imbalance banner |
| Hyperparameter wall | Intimidation, random fiddling | Guided defaults + Advanced drawer + inline explanations |
| Hidden data assumptions (delimiter, date format, missing tokens) | Silent corruption users can't diagnose | Schema-confirmation step with overrides and a "what I assumed" list |
| No progress/ETA for slow ops | Users think app is dead and reload (losing state) | Progress, ETA, cancel, autosave |
| Valid-looking but nonsensical model/target combos | Garbage results | Compatibility filtering with reasons |
| Live curves showing test metrics | Teaches test-set tuning | Show train vs validation; reveal test at the end |
| Animation without control | RL agent flashes by or crawls | Speed slider, pause, step, replay best episode |
| 3D for 2D data | Slower, harder to read | 2D default; 3D when it adds a dimension |
| Color-only class encoding | Inaccessible to colour-blind users | Shape/pattern cues + accessible palette; data-table fallback |
| Loss of work on refresh/crash | Frustration | Autosave state to IndexedDB, restore prompt |
| Backend offline with no explanation | "Run on backend" silently fails | Health check, clear states, one-line start instructions |

## "Looks Done But Isn't" Checklist

- [ ] **Train/test split:** Often missing stratification, stored seed, and train-only fitting - verify scaler/imputer/encoder statistics are computed from train rows only (unit test with an outlier in test).
- [ ] **CSV upload:** Often missing BOM/encoding/delimiter/decimal-comma handling - verify against the ugly-file fixture suite.
- [ ] **Excel upload:** Often missing sheet/header-row picker and size caps - verify with a multi-sheet file and a huge sparse sheet.
- [ ] **Live training chart:** Often missing throttling and history cap - verify UI stays >30 fps while training a 1k-epoch run.
- [ ] **Stop button:** Often only cooperative - verify it halts a long t-SNE/epoch within ~1s and memory returns to baseline.
- [ ] **Metrics:** Often missing baseline, per-class values, and zero-support handling - verify on a 99:1 dataset and a class absent from predictions.
- [ ] **Neural-net training:** Often leaks tensors - verify `tf.memory().numTensors` is flat across 20 repeated runs.
- [ ] **Algorithm correctness:** Often untested against a reference - verify golden-fixture parity with scikit-learn within tolerance.
- [ ] **3D views:** Often missing context-loss handling and disposal - verify by mounting/unmounting 20 scenes and forcing context loss.
- [ ] **SSR:** Often passes in `next dev` only - verify `next build && next start` and SSR-only render of every route.
- [ ] **WASM/worker assets:** Often missing in prod - verify a production build loads workers and `.wasm` from a deployed path.
- [ ] **RL:** Often one lucky run - verify multi-seed curves, reproducibility by seed, and CartPole against known Gym trajectories.
- [ ] **Persistence:** Often no versioning or quota handling - verify import of a previous-version export and behaviour at quota.
- [ ] **Backend mode:** Often only works on the author's machine - verify clean Windows venv install, CORS, token auth, offline state.
- [ ] **Accessibility:** Often skipped for canvas - verify keyboard path, reduced motion, non-color cues, and table fallbacks.
- [ ] **Privacy:** Often leaked by defaults - verify zero third-party requests after dataset load.

## Recovery Strategies

| Pitfall | Recovery Cost | Recovery Steps |
|---------|---------------|----------------|
| ML code scattered through components (no worker boundary) | HIGH | Introduce the RPC layer, move imports into `/ml`, add lint rule, migrate feature by feature |
| Eager mutating cleaning (leakage) | HIGH | Redesign as recipe with fit/transform; rewrite cleaning/FE steps; re-validate metrics |
| Row-object dataset model | HIGH | Introduce columnar store behind the same accessor interface; migrate parsers and EDA |
| Per-widget WebGL canvases | MEDIUM | Move to shared canvas with `View` tracking; add context-restore handling |
| Wrong/unvalidated algorithm implementation | MEDIUM | Add sklearn golden fixtures, fix or swap implementation, add CI parity test |
| Unversioned saved experiments | MEDIUM | Add `schemaVersion`, ship a one-time migration that infers v0 |
| Vulnerable xlsx dependency | LOW | Swap parser/pin official build; add size caps |
| Open CORS backend | LOW | Origin allow-list + token + bind change; release notes |
| Live-chart re-render storms | LOW | Add throttle/ring buffer and canvas chart |
| TF.js maintenance risk materializes | MEDIUM | If isolated behind Estimator interface: swap engine for LiteRT.js/ORT/hand-written; otherwise HIGH |

## Pitfall-to-Phase Mapping

| Pitfall | Prevention Phase | Verification |
|---------|------------------|--------------|
| 1 Main-thread compute | F | Lint rule bans ML imports outside `/ml`; Performance trace shows no long tasks during training |
| 2 Progress flooding | F, S, V | UI >30 fps during 1k-epoch run; history capped |
| 3 Preprocessing leakage | F, C, FE | Outlier-in-test unit test; recipe has fit/transform |
| 4 Target leakage / test reuse | FE, S | Warnings fire on ID/target-proxy fixtures; curves show validation |
| 5 Misleading metrics | S | Baseline shown; 99:1 fixture shows PR/per-class metrics |
| 6 Imbalance handling | FE, S | Stratified split counts displayed; class-weight test passes |
| 7 Memory blowups | I, FE, F | 200 MB CSV and 5k-image set load or route to backend without crash; `numTensors` flat |
| 8 Dataset in React state | F, I | Re-render count independent of row count |
| 9 SSR failures | F | `next build` and SSR-only smoke test in CI |
| 10 StrictMode double mount | F, V | One worker/one job per click in dev |
| 11 Worker/WASM bundling | F | Production build runs worker + WASM end to end |
| 12 TF.js not covering all models | F (spike), S, U | Engine map per algorithm; golden tests |
| 13 TF.js backend selection | F, S, RL | Backend shown in UI; fallback tested with GPU disabled |
| 14 WebGL context loss | V, RL | Mount/unmount 20 scenes; forced context-loss recovery |
| 15 Parsing edge cases | I | Ugly-file fixture suite green |
| 16 Upload/spreadsheet security | I, X, F | `npm audit` clean; XSS and formula-injection tests |
| 17 Backend exposure | B | Cross-origin request from a foreign origin rejected; token required; bound to 127.0.0.1 |
| 18 Privacy leakage | F, B | Zero third-party requests after upload |
| 19 RL instability | RL | Multi-seed curves; seeded reproducibility; CartPole trajectory test |
| 20 Scaling/NaN/cardinality | FE, S, U | Preflight blocks NaN/Inf; unseen-category test |
| 21 Costly visualizations | U, V, S | Boundary/t-SNE caps and progress verified at limit sizes |
| 22 Unfair comparison | F, X | Comparison refuses mismatched splits/targets |
| 23 Persistence | F, X | Round-trip export/import across a schema bump; quota-error path |
| 24 Over-complex UI | F, all feature phases, X | Usability pass; accessibility audit; defaults visible <= ~6 controls |
| 25 Cancel/error handling | F, S, U, RL | Stop halts long job; crashed worker produces error state |
| 26 Bundle weight | F, V, S | Bundle budget in CI; ML code absent from landing route |

## Sources

- Next.js App Router SSR behaviour (`ssr: false` only from Client Components, `'use client'` still pre-renders, static imports elsewhere still execute server-side): web search results of 2026-11-09 (OpenReplay "window is not defined SSR", DEV Community Next.js dynamic import guides, R3F + Next.js tutorials) - MEDIUM; confirm against official Next.js docs.
- TF.js backend package health: Snyk advisor page for `@tensorflow/tfjs-backend-webgpu` (v4.22.0, "Healthy") - LOW/MEDIUM; the "TensorFlow.js killed" claim (stork.ai) is **unverified, low reliability**; check the tensorflow/tfjs repo and release notes directly before locking in.
- TF.js WASM backend and platform docs: blog.tensorflow.org WASM backend post (2020), tensorflow.org/js/guide/platform_environment - MEDIUM (dated; re-verify op coverage and worker/WebGL support).
- Domain knowledge (HIGH unless marked): scikit-learn "Common pitfalls and recommended practices" (data leakage, pipelines), imbalanced-learn guidance on resampling inside CV, SheetJS npm advisories (prototype pollution CVE-2023-30533, ReDoS CVE-2024-22363; verify current distribution), OWASP CSV injection, three.js context-limit and disposal guidance, Gymnasium CartPole spec, DQN literature (Mnih et al.) and RL reproducibility (Henderson et al., "Deep RL that Matters"), WebKit storage eviction policy, FastAPI CORS docs.
- Needs verification when reached: Chrome local-network-access / private-network permission behaviour for HTTPS site -> localhost backend (MEDIUM/LOW); WebGPU backend browser coverage (MEDIUM/LOW); Next.js/Turbopack worker + WASM asset handling (MEDIUM).

---
*Pitfalls research for: interactive in-browser ML playground (Next.js + TF.js/ONNX workers + optional FastAPI + three.js)*
*Researched: 2026-11-09*
