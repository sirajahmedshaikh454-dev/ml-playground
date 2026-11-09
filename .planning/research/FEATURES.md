# Feature Research

**Domain:** Interactive ML playground / no-code data-science + model-training visualization web app
**Researched:** 2026-11-09
**Confidence:** MEDIUM (competitor feature sets from established product knowledge: TF Playground, Teachable Machine, Orange, Weka, KNIME, Colab/Kaggle, HF Spaces, AutoML UIs, RL playgrounds. No live re-verification this session; no Context7/provider calls made. Treat per-competitor specifics as MEDIUM, category-level conclusions as HIGH.)

## Competitor Landscape (what each one teaches us)

| Product | Strength to learn from | Gap this project fills |
|---------|------------------------|------------------------|
| TensorFlow Playground | Live loss curves, decision-boundary heatmap, per-neuron activation view, tweak-and-watch loop, shareable URL state | Toy 2D datasets only; no user data, no pipeline, one model family |
| Teachable Machine | Zero-setup upload/webcam -> train in browser -> instant try-it-live preview, export | Classification only (image/audio/pose); no cleaning/EDA/metrics depth |
| Orange | Visual widget canvas, data table, rich viz (scatter, box, distributions), "Test & Score" comparison, explainers | Desktop app, Python install, widget-wiring learning curve, no live training animation |
| Weka | Broad algorithm menu, Explorer tabs (Preprocess/Classify/Cluster/Select), CV by default | Dated UX, Java desktop, no live visuals |
| KNIME | Node workflow graph, reproducibility | Heavy, enterprise-feeling, steep for learners |
| Colab / Kaggle | Real code, GPU, datasets, sharing | Requires coding; not "see ML working" |
| HF Spaces / Gradio | Try-a-model UI, shareable demos | Inference-focused, not training/learning-focused |
| DataRobot / Vertex / H2O AutoML UIs | Leaderboard, auto-preprocessing, feature impact, explainability (SHAP) | Black-box "autopilot"; hides the learning process, paywalled |
| RL playgrounds (gym demos, "RL playground" gridworld UIs, Karpathy REINFORCEjs, Unity ML demos) | Agent animation, reward curve, Q-table/value heatmap, speed slider | Isolated from the rest of data science; no unified UX |

**Positioning insight:** no existing product combines the full pipeline (upload -> clean -> EDA -> features -> train -> evaluate -> compare) with TF-Playground-style *live* visualization across supervised, unsupervised AND RL in one browser app. That unification plus liveness is the differentiator. Everything else is table stakes borrowed from Orange/Weka/AutoML.

## Feature Landscape

### Table Stakes (Users Expect These)

Missing any of these = product feels broken or toy-like.

#### Data ingestion

| Feature | Why Expected | Complexity | Notes |
|---------|--------------|------------|-------|
| CSV/TSV upload (drag-drop) with preview table | Every tool does this | LOW | PapaParse in a Worker; virtualized table |
| JSON and Excel (xlsx) upload | Stated requirement; Orange/Weka/KNIME all do it | MEDIUM | SheetJS; sheet picker for multi-sheet; JSON array-of-records and nested flatten |
| Auto schema/dtype detection (numeric, categorical, datetime, text, boolean) with user override | Orange/AutoML UIs do it; wrong dtypes poison everything downstream | MEDIUM | Sample-based inference; datetime parsing is the usual trap |
| Target column + task-type selection (classification/regression/clustering) with auto-suggestion | Needed to route to correct models/metrics | LOW | Suggest by target cardinality/dtype |
| Built-in sample datasets (Iris, Titanic, MNIST-subset, housing, blobs/moons/circles/spirals) | Learners without data bounce otherwise; TF Playground and Orange ship these | LOW | Also the first-run "aha" path; doubles as test fixtures |
| File size/row limit warnings and sampling | Browser memory is finite; users must be told, not crashed | LOW | Hand off to FastAPI backend above threshold |
| Image folder/zip upload with labels from folder names | Teachable Machine baseline for "images" | MEDIUM | Thumbnail grid preview; resize to fixed size client-side |
| Plain text upload (labeled lines / CSV with text column) | Stated requirement | LOW | Text is just a column type + featurizer |

#### Cleaning

| Feature | Why Expected | Complexity | Notes |
|---------|--------------|------------|-------|
| Missing-value report + impute (mean/median/mode/constant) or drop rows/cols | Core of any cleaning step | LOW | Show count/% per column |
| Duplicate detection and removal | Standard | LOW | |
| Outlier detection (IQR/z-score) with clip/remove/keep | Standard in Orange/Weka | MEDIUM | Visual flagging on box plot |
| dtype conversion and value fixes (trim, case, parse numbers/dates) | Real-world CSVs are messy | MEDIUM | |
| Before/after comparison view (stated requirement) | Learners must see what cleaning did | MEDIUM | Side-by-side stats + distribution overlay |
| Undo/redo and an ordered, visible list of applied steps | Destructive edits without undo = lost trust | MEDIUM | Model as immutable step list over raw data; also gives reproducibility |

#### EDA

| Feature | Why Expected | Complexity | Notes |
|---------|--------------|------------|-------|
| Summary stats per column (count, mean, std, min/max, quantiles, unique, missing) | Pandas `describe` is the baseline | LOW | |
| Histograms / bar charts per column | Baseline | LOW | Recharts/D3 |
| Correlation matrix heatmap | Baseline | LOW | Pearson; Cramér's V/ANOVA for mixed is a later upgrade |
| Class balance chart (target distribution) | Needed to interpret accuracy honestly | LOW | Warn on imbalance |
| Scatter / scatter-matrix and box plots; color by target | Orange's most-used views | MEDIUM | Cap points (sampling) for performance |
| Auto-generated "findings" callouts (skew, high-cardinality, constant column, leakage-suspect) | AutoML UIs (DataRobot) and ydata-profiling do it | MEDIUM | Rule-based; high value for learners |

#### Feature engineering and splitting

| Feature | Why Expected | Complexity | Notes |
|---------|--------------|------------|-------|
| Categorical encoding (one-hot, ordinal/label) | Required for nearly all models | LOW | |
| Scaling (standardize, min-max) | Required for kNN/SVM/NN; visible effect | LOW | |
| Feature selection (drop columns, variance/correlation filter, importance-based) | Orange/Weka Select-Attributes | MEDIUM | |
| Train/test split with ratio, shuffle, seed, stratify | Stated requirement; seed is essential for reproducibility | LOW | Optional validation split; k-fold CV toggle |
| Pipeline fit on train only, applied to test (no leakage) | Correctness: silently wrong results destroy "honest test results" core value | MEDIUM | Architecture-level constraint, not just a feature; see Pitfalls |
| Text featurization (bag-of-words/TF-IDF) | Stated requirement | MEDIUM | Simple, explainable first; embeddings are a differentiator |
| Image featurization (resize/normalize, raw pixels -> CNN input) | Stated requirement | MEDIUM | |

#### Supervised

| Feature | Why Expected | Complexity | Notes |
|---------|--------------|------------|-------|
| Linear/logistic regression, decision tree, random forest, kNN, SVM, small neural net | Stated; Weka/Orange baseline menu | HIGH (collectively) | Browser: tfjs for NN/linear; JS libs (ml.js / custom) for trees/kNN/SVM; backend sklearn for heavy. Library choice is a STACK concern |
| Hyperparameter controls with sensible defaults and tooltips | TF Playground's core interaction | MEDIUM | Plain-English explanation next to each knob = learner value |
| Start/pause/step/reset training controls | TF Playground baseline | MEDIUM | Needs Worker + cancellable loop |
| Honest hold-out evaluation (never evaluate on train only) | Core value | LOW | Show train AND test metrics side by side to expose overfitting |

#### Unsupervised

| Feature | Why Expected | Complexity | Notes |
|---------|--------------|------------|-------|
| k-means, DBSCAN, hierarchical clustering | Stated | MEDIUM | Hierarchical needs a dendrogram |
| PCA and t-SNE (UMAP as stretch) 2D/3D projection | Stated; Orange/Embedding Projector baseline | MEDIUM-HIGH | t-SNE/UMAP iterative; animate convergence |
| Anomaly detection (isolation forest / z-score / one-class) | Stated | MEDIUM | |
| Cluster quality metrics (silhouette, inertia, elbow plot) | Without these clustering is unfalsifiable | LOW | |

#### Reinforcement learning

| Feature | Why Expected | Complexity | Notes |
|---------|--------------|------------|-------|
| Gridworld and CartPole environments rendered live | Stated; the canonical RL demos | MEDIUM | Client-side sim; gridworld first (tabular, cheap) |
| Q-learning, SARSA (tabular); DQN (CartPole) | Stated | MEDIUM (tabular) / HIGH (DQN) | DQN in tfjs is feasible but fiddly; replay buffer + target net |
| Episode reward curve and agent animation with speed slider | Universal RL-demo baseline | MEDIUM | Decouple sim speed from render speed |
| Reward/penalty/hyperparameter editing (alpha, gamma, epsilon, step penalty, goal reward) | The point of an RL *playground* | LOW-MEDIUM | Editable gridworld layout (walls, pits, goal) is the fun part |
| Q-table / value heatmap overlay on gridworld | REINFORCEjs's signature view | MEDIUM | |

> Note: RL is **not** "upload your data". It runs on built-in environments, a separate track from the tabular pipeline. Do not force it through the dataset flow.

#### Live visualization

| Feature | Why Expected | Complexity | Notes |
|---------|--------------|------------|-------|
| Live loss/accuracy curves (train vs test) | TF Playground baseline | MEDIUM | Streamed from Worker via postMessage batches; throttle renders |
| Decision-boundary plot for 2D classification | TF Playground's signature | MEDIUM-HIGH | Grid prediction each N steps; only valid for 2-feature models (user picks 2 features or uses PCA) |
| Cluster plot updating per iteration | Standard k-means demo | MEDIUM | |
| UI stays responsive during training | Stated constraint; frozen tab = abandoned | MEDIUM | Web Workers mandatory; tfjs WebGPU/WebGL backend in Worker |

#### Evaluation

| Feature | Why Expected | Complexity | Notes |
|---------|--------------|------------|-------|
| Classification: accuracy, precision/recall/F1, confusion matrix, ROC/PR curves | Stated; baseline | MEDIUM | Multiclass (macro/micro) handling is the usual snag |
| Regression: MAE/RMSE/R2, predicted-vs-actual, residual plot | Stated; baseline | LOW-MEDIUM | |
| Metrics auto-selected by task type | Stated | LOW | |
| Predict on a single new row/input and see result ("try it") | Teachable Machine's hook; makes the model tangible | MEDIUM | Reuse the saved preprocessing pipeline |

#### Comparison, persistence, platform

| Feature | Why Expected | Complexity | Notes |
|---------|--------------|------------|-------|
| Run history list with config + metrics | Orange "Test & Score", AutoML leaderboard | MEDIUM | |
| Side-by-side model comparison table and overlaid curves/ROC | Stated | MEDIUM | Compare on the SAME split/seed or warn |
| Local persistence (IndexedDB) of datasets, pipelines, runs; export/import | Stated; no-accounts model | MEDIUM | Quota limits for large datasets; store processed/sampled data + step list |
| Export results (metrics CSV/JSON, chart PNG/SVG) | Expected by analysts/students writing reports | LOW | |
| Responsive, accessible, polished UI; empty/error/loading states | Stated; "production-quality" | MEDIUM | Keyboard nav, color-blind-safe palettes, reduced-motion respect |
| Clear errors for bad files / failed training (not silent) | Basic trust | LOW | |

### Differentiators (Competitive Advantage)

Align with Core Value: *see end to end how a model learns from your own data.*

| Feature | Value Proposition | Complexity | Notes |
|---------|-------------------|------------|-------|
| **One unified guided pipeline** (Upload -> Clean -> EDA -> Features -> Split -> Train -> Evaluate -> Compare) with stepper and "you are here" | Nothing else does the whole path with visuals in-browser, zero install | MEDIUM | The structural differentiator; mostly UX + state design |
| **Live training on YOUR data** (not toy 2D only) with decision boundaries via PCA/2-feature selector | TF Playground magic applied to real datasets | HIGH | Biggest technical risk; pick 2 features or PCA projection for boundary |
| **Supervised + unsupervised + RL in one app** | Unique breadth for learners | HIGH (breadth) | Keep each paradigm's UI native, share chart/run/persistence infra |
| Plain-English "what's happening" explainers and tooltips on every knob/metric | Learner audience; Orange/Weka assume knowledge | MEDIUM | Content-heavy; write once in a data file |
| Overfitting made visible (train vs test curve gap highlighted with callout) | Teaches the single most important ML lesson | LOW-MEDIUM | Cheap given curves exist; high educational value |
| 3D visualizations (3D PCA/embedding scatter, 3D loss surface or cluster view) via react-three-fiber | Stated; wow factor; genuinely useful for 3 PCs/embeddings | HIGH | Use where 3D adds info; not decoration. Performance cap on points |
| Neural-net architecture visualizer (layers, weights, activations animating) | TF Playground's other signature | HIGH | Do for small MLPs only |
| Side-by-side live training of multiple models/configs on the same data | "Race" mode; powerful comparison; unique | HIGH | Defer until single-run is solid; multiple Workers |
| Reproducible experiment export (JSON bundle of steps + config + seed + metrics; optionally generated Python/sklearn script) | Bridges no-code to real code; learner graduation path | MEDIUM-HIGH | Code export is high value/low risk if templated; JSON bundle is cheap |
| Shareable/re-importable experiment file (no server) | Collaboration without accounts | LOW-MEDIUM | Falls out of export |
| Feature importance / permutation importance / partial dependence | Explainability is a prime AutoML selling point | MEDIUM | Permutation importance works model-agnostically |
| Learning curves and validation curves (vary train size / hyperparameter) | Teaches bias-variance | MEDIUM | Needs repeated training; cheap on small data |
| Cross-validation with per-fold visualization | Honest estimates; Weka-grade rigor in a friendly UI | MEDIUM | |
| Auto-suggest next steps ("class imbalance detected, try stratify / class weights") | Guided-learner experience | MEDIUM | Rule engine over EDA findings |
| Opt-in FastAPI backend with auto-detect and one-click "run heavy on backend" | Power without losing privacy default | HIGH | Needs discoverable connection UI, health check, CORS, same protocol as browser Worker (shared training-event schema) |
| RL extras: editable environment, policy arrows, compare algorithms on same env, reward-shaping sandbox | Makes "reward/punishment" tangible | MEDIUM | Gridworld editor is the cheap high-value piece |
| Animated transitions (Framer Motion) between pipeline steps and data changes | Perceived quality; shows what changed | LOW-MEDIUM | Respect prefers-reduced-motion |
| Mini AutoML "try several models, show leaderboard" | DataRobot-style convenience, but transparent | MEDIUM | Only after comparison exists; show each model's live run, never black-box |

### Anti-Features (Commonly Requested, Often Problematic)

| Feature | Why Requested | Why Problematic | Alternative |
|---------|---------------|-----------------|-------------|
| User accounts / cloud save / social sharing | Seems standard | Out of scope per PROJECT.md; auth, storage, privacy liability | IndexedDB + export/import file |
| Model serving / deploy endpoint / REST API for trained model | AutoML UIs have it | Out of scope; turns learning tool into MLOps | Export model file / generated script |
| Hosted GPU / distributed training | "Train big models" | Out of scope; cost and ops | Optional user-run FastAPI backend |
| Full visual node-graph workflow editor (KNIME/Orange canvas) | Looks powerful | Huge build cost, steep learning curve, undermines guided linear flow for learners | Linear stepper with branching via run history |
| In-app code editor / notebook | Colab familiarity | Doubles scope (sandboxing, kernel); competes with Colab itself | Export generated Python script |
| Black-box "one-click AutoML" hiding the process | Fast results | Contradicts core value (see how it learns) | Transparent leaderboard with live runs |
| Arbitrary custom environments/Gym import for RL | Flexibility | Needs Python runtime, sandboxing, rendering contracts | Built-in gridworld (editable) + CartPole; maybe MountainCar later |
| Large pretrained models / LLM fine-tuning / transformers training | Trendy | Not browser-feasible; scope explosion | Out for v1; maybe inference-only feature later |
| Real-time collaborative editing | Modern feel | Needs servers/CRDTs; conflicts with no-accounts | Export/import files |
| Full time-series forecasting suite | Common data type | Large separate domain (lags, CV schemes, leakage) | Allow datetime column as feature; defer proper TS to v2 |
| Pixel-perfect everything on 1M+ row data in-browser | "Any dataset" | Browser memory/CPU limits; UI freezes | Sampling for display, row-limit warnings, backend offload |
| Every algorithm under the sun (gradient boosting variants, 50+ models) | Weka-style breadth | Maintenance and UX clutter; most don't add learning value | Curated ~12-15 with excellent viz; add boosting (GBM) later |
| Real-time 60fps updates of every chart on every training step | "Live" feel | Kills performance; React re-render storm | Throttle to ~10-20 Hz; batch Worker messages; canvas for heavy plots |
| Heavy tooltips/modals interrupting flow | Education | Annoying for repeat users | Inline expandable explainers, "learn mode" toggle |

## Feature Dependencies

```
Upload + schema detection
    └──requires──> Preview table (virtualized)
    └──enables──> Cleaning ──requires──> Step list (immutable, undoable)
                     └──enables──> EDA
                                      └──enables──> Feature engineering
                                                         └──requires──> Train/test split (fit-on-train pipeline)
                                                                            └──enables──> Supervised / Unsupervised training

Web Worker training harness + streamed event schema
    ├──required by──> Live curves, decision boundary, cluster animation
    ├──required by──> Start/pause/step/reset controls
    └──required by (shared schema)──> FastAPI backend offload

Training + Evaluation metrics
    └──requires──> Honest hold-out evaluation
    └──enables──> Run history ──enables──> Model comparison ──enables──> Race mode, mini-AutoML leaderboard

Saved preprocessing pipeline ──enables──> "Predict on new input" (try it)

Local persistence (IndexedDB) ──enables──> Run history, Export/import, Reproducible bundle ──enables──> Code export

RL environments (gridworld/CartPole)  [independent track; shares: Worker harness, chart components, persistence]
    └──requires──> Env simulator + renderer ──requires──> Q-learning/SARSA ──enables──> DQN

3D viz ──enhances──> PCA/t-SNE/UMAP, cluster plots
Explainers ──enhances──> every step
Overfitting callout ──requires──> Train + test curves

Image/text upload ──requires──> Featurizers ──requires──> NN training path (images)
Decision boundary ──requires──> 2-feature (or PCA-2D) projection ──conflicts──> >2 raw features without projection
Model comparison ──conflicts──> different splits/seeds (invalid comparison; must warn or lock split)
```

### Dependency Notes

- **Step list underpins cleaning, features, reproducibility and export:** design it once as an ordered, serializable transform list; undo, before/after, pipeline replay for "try it", and export all fall out of it.
- **Worker harness + event schema is the keystone:** every live visual and the FastAPI backend depend on one shared "training event" contract (epoch, metrics, snapshot). Build this before any individual model.
- **RL is a parallel track:** it needs only the Worker harness, chart components, and persistence, not the data pipeline. It can be scheduled independently of the data-cleaning phases.
- **Fit-on-train-only pipeline must precede evaluation:** otherwise leakage invalidates the "honest results" core value.
- **Decision boundary conflicts with high-dimensional data:** requires the user to choose 2 features or a PCA projection; design the UI around that, do not hide it.
- **Comparison requires a locked split/seed:** store split seed in each run; warn when comparing runs across different splits.
- **Race mode and mini-AutoML require single-run comparison first.**

## MVP Definition

Project decision is "full scope in v1, delivered in phases," so MVP here means the first *shippable vertical slice* that proves Core Value, not the whole of v1.

### Launch With (first vertical slice)

- [ ] CSV upload + sample datasets + preview + schema detection: entry point, zero-friction demo
- [ ] Missing/duplicate/outlier handling with before/after + undo step list: proves the pipeline concept
- [ ] Auto EDA (stats, histograms, correlation, class balance): needed to interpret anything
- [ ] Encoding, scaling, train/test split (seeded, stratified) with leak-free pipeline: correctness foundation
- [ ] Worker training harness + live loss/accuracy curves: the Core Value moment
- [ ] 3-4 supervised models (logistic/linear, tree, kNN, small NN) with decision boundary on 2D projection: the "TF Playground on your data" hook
- [ ] Confusion matrix + metrics by task type + train-vs-test gap: honest results
- [ ] IndexedDB run history + simple compare table + JSON export: persistence/comparison baseline
- [ ] Design-system UI shell, stepper, accessible, responsive

### Add After Validation (v1.x, remaining v1 scope in later phases)

- [ ] JSON/Excel/image/text upload and featurizers: breadth, once CSV flow is solid
- [ ] Remaining supervised models (forest, SVM) + ROC/PR, residuals, CV
- [ ] Unsupervised: k-means -> DBSCAN/hierarchical -> PCA/t-SNE/UMAP, anomaly detection
- [ ] RL track: gridworld + Q-learning/SARSA first, then CartPole + DQN
- [ ] 3D viz (PCA/embedding 3D scatter first)
- [ ] FastAPI backend offload (sklearn/PyTorch) with shared event schema
- [ ] Overlay comparison of curves/ROC, feature importance, learning curves
- [ ] Explainers/learn mode content pass

### Future Consideration (v2+)

- [ ] Race mode (multi-model live) and mini-AutoML leaderboard: after comparison is proven
- [ ] Python/sklearn code export: after pipeline/step-list is stable
- [ ] Neural-net architecture visualizer, additional RL envs, gradient boosting, basic time-series
- [ ] Inference-only pretrained model hooks (HF-style): scope-heavy

## Feature Prioritization Matrix

| Feature | User Value | Implementation Cost | Priority |
|---------|------------|---------------------|----------|
| CSV upload + preview + schema detection | HIGH | LOW | P1 |
| Sample datasets | HIGH | LOW | P1 |
| Cleaning with before/after + undo | HIGH | MEDIUM | P1 |
| Auto EDA | HIGH | MEDIUM | P1 |
| Encoding/scaling/split, leak-free | HIGH | MEDIUM | P1 |
| Worker harness + live curves | HIGH | MEDIUM | P1 |
| Core supervised models + metrics | HIGH | HIGH | P1 |
| Decision boundary (2D) | HIGH | MEDIUM-HIGH | P1 |
| Run history + compare + local persistence/export | HIGH | MEDIUM | P1 |
| JSON/Excel upload | MEDIUM | MEDIUM | P2 |
| Image + text upload and featurizers | MEDIUM | MEDIUM | P2 |
| Clustering + projection (PCA/t-SNE) | HIGH | MEDIUM-HIGH | P2 |
| RL gridworld + Q-learning/SARSA | HIGH | MEDIUM | P2 |
| CartPole + DQN | MEDIUM | HIGH | P2 |
| FastAPI backend offload | MEDIUM | HIGH | P2 |
| 3D visualizations | MEDIUM | HIGH | P2 |
| Feature importance, learning curves, CV | MEDIUM | MEDIUM | P2 |
| Explainers / learn mode | HIGH | MEDIUM | P2 |
| Race mode / mini-AutoML | MEDIUM | HIGH | P3 |
| Python code export | MEDIUM | MEDIUM | P3 |
| NN architecture visualizer | MEDIUM | HIGH | P3 |
| UMAP, anomaly extras, more algorithms | LOW | MEDIUM | P3 |

**Priority key:** P1 = must have for the first shippable slice; P2 = rest of v1 scope, later phases; P3 = nice to have / v2.

## Competitor Feature Analysis

| Feature | TF Playground / Teachable Machine | Orange / Weka / KNIME | AutoML UIs (DataRobot etc.) | Our Approach |
|---------|-----------------------------------|-----------------------|-----------------------------|--------------|
| Own-data upload | Teachable Machine only (images/audio); TFP none | Yes, many formats | Yes | Many formats, local-only |
| Cleaning/EDA | None | Yes (widgets / Preprocess tab) | Auto, mostly hidden | Guided, visible, undoable, before/after |
| Live training viz | Yes (TFP strong) | No / minimal | Progress bars only | First-class: curves, boundary, clusters, RL animation |
| Algorithm breadth | NN only | Very broad | Broad, automatic | Curated ~12-15 + RL |
| Unsupervised | No | Yes | Limited | Yes, animated |
| RL | No | No | No | Yes (gridworld, CartPole) |
| Evaluation | Minimal | Strong (Test & Score) | Strong + explainability | Strong + train-vs-test gap education |
| Comparison | None | Yes | Leaderboard | Run history + overlay; race mode later |
| Setup | None (browser) | Install | SaaS/paid | None (browser), optional backend |
| Explainability for learners | Visual intuition | Moderate | SHAP/feature impact | Plain-English explainers + permutation importance |
| Export | Model/code (TM) | Workflow files | Deploy API | JSON bundle, later Python script |

## Sources

- TensorFlow Playground (playground.tensorflow.org): live loss, decision boundary, neuron viz, URL-state sharing. MEDIUM (product knowledge, not re-fetched).
- Teachable Machine (teachablemachine.withgoogle.com): upload/webcam, in-browser train, live preview, export. MEDIUM.
- Orange Data Mining widget catalog and Test & Score; Weka Explorer tabs; KNIME node workflow model. MEDIUM.
- Colab/Kaggle notebooks, Hugging Face Spaces/Gradio: code-first / inference-first comparison points. MEDIUM.
- DataRobot / H2O / Vertex AutoML UIs: leaderboard, feature impact, auto-preprocessing. MEDIUM.
- REINFORCEjs (Karpathy) gridworld/Q-learning demos, OpenAI Gym CartPole as canonical RL demo environments. MEDIUM.
- PROJECT.md in this repo: scope, out-of-scope, constraints. HIGH.
- Recommendation: spot-verify competitor current features (esp. Teachable Machine and TFP updates) before finalizing requirements if any single claim becomes load-bearing.

---
*Feature research for: interactive ML playground / no-code data-science and training visualization*
*Researched: 2026-11-09*
