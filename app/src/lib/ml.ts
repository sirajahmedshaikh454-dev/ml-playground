// Every model is an async generator of Progress events. The runner decides how fast
// to pull from it, so pause / step / stop / speed work the same for every algorithm.
import { DecisionTreeClassifier, DecisionTreeRegression } from "ml-cart";
import { PCA } from "ml-pca";
import { type Predictor, type Progress, type Task, dist2, dot, mean, mulberry32, std } from "./core";
import { classReport, regReport } from "./metrics";

export interface ParamDef { key: string; label: string; min: number; max: number; step: number; def: number; hint?: string }
export interface Ctx {
  task: Task;
  Xtr: number[][]; ytr: number[];
  Xva: number[][]; yva: number[];
  k: number; // number of classes
  params: Record<string, number>;
  seed: number;
}
export interface Engine {
  id: string;
  label: string;
  blurb: string;
  family: "supervised" | "unsupervised";
  tasks: Task[];
  params: ParamDef[];
  remote?: boolean;
  run: (c: Ctx) => AsyncGenerator<Progress, void, unknown>;
}

// ---------- helpers ----------
function fitStd(X: number[][]) {
  const d = X[0]?.length ?? 0;
  const m = Array.from({ length: d }, (_, j) => mean(X.map((r) => r[j])));
  const s = Array.from({ length: d }, (_, j) => std(X.map((r) => r[j])) || 1);
  return (x: number[]) => x.map((v, j) => (v - m[j]) / s[j]);
}
const softmax = (z: number[]) => {
  const mx = Math.max(...z);
  const e = z.map((v) => Math.exp(v - mx));
  const s = e.reduce((a, b) => a + b, 0);
  return e.map((v) => v / s);
};
const argmax = (a: number[]) => a.reduce((bi, v, i) => (v > a[bi] ? i : bi), 0);
const xent = (P: number[][], y: number[]) => mean(P.map((p, i) => -Math.log(Math.max(1e-12, p[y[i]] ?? 1e-12))));

// standard score bundle for a predictor on train + validation
function score(c: Ctx, pr: Predictor, extra: Record<string, number> = {}): Record<string, number> {
  if (c.task === "regression") {
    const t = regReport(c.ytr, pr.predict(c.Xtr));
    const v = regReport(c.yva, pr.predict(c.Xva));
    return { loss: t.rmse, val_loss: v.rmse, r2: t.r2, val_r2: v.r2, ...extra };
  }
  const out: Record<string, number> = {
    acc: classReport(c.ytr, pr.predict(c.Xtr), c.k).accuracy,
    val_acc: classReport(c.yva, pr.predict(c.Xva), c.k).accuracy,
    ...extra,
  };
  if (pr.proba && !("loss" in extra)) {
    out.loss = xent(pr.proba(c.Xtr), c.ytr);
    out.val_loss = xent(pr.proba(c.Xva), c.yva);
  }
  return out;
}

// ---------- supervised ----------
async function* linear(c: Ctx): AsyncGenerator<Progress> {
  const { epochs, lr, l2 } = c.params;
  const f = fitStd(c.Xtr);
  const Z = c.Xtr.map(f);
  const ym = mean(c.ytr), ys = std(c.ytr) || 1;
  const t = c.ytr.map((v) => (v - ym) / ys);
  const d = Z[0].length, n = Z.length;
  const w = new Array<number>(d).fill(0);
  let b = 0;
  const pr: Predictor = { predict: (X) => X.map((x) => (b + dot(w, f(x))) * ys + ym) };
  for (let e = 1; e <= epochs; e++) {
    const gw = new Array<number>(d).fill(0);
    let gb = 0;
    for (let i = 0; i < n; i++) {
      const err = b + dot(w, Z[i]) - t[i];
      gb += err;
      for (let j = 0; j < d; j++) gw[j] += err * Z[i][j];
    }
    for (let j = 0; j < d; j++) w[j] -= lr * (gw[j] / n + l2 * w[j]);
    b -= (lr * gb) / n;
    yield { step: e, total: epochs, metrics: score(c, pr), predictor: pr };
  }
}

async function* logistic(c: Ctx): AsyncGenerator<Progress> {
  const { epochs, lr, l2 } = c.params;
  const f = fitStd(c.Xtr);
  const Z = c.Xtr.map(f);
  const d = Z[0].length, n = Z.length, k = c.k;
  const W = Array.from({ length: k }, () => new Array<number>(d).fill(0));
  const b = new Array<number>(k).fill(0);
  const probs = (z: number[]) => softmax(W.map((wk, i) => b[i] + dot(wk, z)));
  const pr: Predictor = {
    proba: (X) => X.map((x) => probs(f(x))),
    predict: (X) => X.map((x) => argmax(probs(f(x)))),
  };
  for (let e = 1; e <= epochs; e++) {
    const gW = Array.from({ length: k }, () => new Array<number>(d).fill(0));
    const gb = new Array<number>(k).fill(0);
    for (let i = 0; i < n; i++) {
      const p = probs(Z[i]);
      for (let cl = 0; cl < k; cl++) {
        const g = p[cl] - (c.ytr[i] === cl ? 1 : 0);
        gb[cl] += g;
        for (let j = 0; j < d; j++) gW[cl][j] += g * Z[i][j];
      }
    }
    for (let cl = 0; cl < k; cl++) {
      for (let j = 0; j < d; j++) W[cl][j] -= lr * (gW[cl][j] / n + l2 * W[cl][j]);
      b[cl] -= (lr * gb[cl]) / n;
    }
    yield { step: e, total: epochs, metrics: score(c, pr), predictor: pr };
  }
}

async function* mlp(c: Ctx): AsyncGenerator<Progress> {
  const tf = await import("@tensorflow/tfjs");
  await tf.ready();
  const { epochs, lr, hidden, layers, batch } = c.params;
  const cls = c.task === "classification";
  const d = c.Xtr[0].length;
  const ym = cls ? 0 : mean(c.ytr), ys = cls ? 1 : std(c.ytr) || 1;
  const model = tf.sequential();
  for (let l = 0; l < layers; l++)
    model.add(tf.layers.dense(l === 0 ? { units: hidden, activation: "relu", inputShape: [d] } : { units: hidden, activation: "relu" }));
  model.add(tf.layers.dense({ units: cls ? c.k : 1, activation: cls ? "softmax" : "linear" }));
  model.compile({ optimizer: tf.train.adam(lr), loss: cls ? "categoricalCrossentropy" : "meanSquaredError", metrics: cls ? ["accuracy"] : [] });
  const xs = tf.tensor2d(c.Xtr);
  const target = (y: number[]) => cls
    ? tf.oneHot(tf.tensor1d(y, "int32"), c.k).toFloat()
    : tf.tensor2d(y.map((v) => [(v - ym) / ys]));
  const yt = target(c.ytr);
  const xv = tf.tensor2d(c.Xva), yv = target(c.yva);
  const vVar = std(c.yva) ** 2 || 1, tVar = (std(c.ytr) || 1) ** 2;
  const raw = (X: number[][]) =>
    X.length ? (tf.tidy(() => (model.predict(tf.tensor2d(X)) as import("@tensorflow/tfjs").Tensor).arraySync()) as number[][]) : [];
  const pr: Predictor = cls
    ? { proba: raw, predict: (X) => raw(X).map(argmax), dispose: () => model.dispose() }
    : { predict: (X) => raw(X).map((r) => r[0] * ys + ym), dispose: () => model.dispose() };
  try {
    for (let e = 1; e <= epochs; e++) {
      // train + validation metrics are computed in-graph by fit(): no extra GPU readbacks per epoch
      const h = (await model.fit(xs, yt, { epochs: 1, batchSize: batch, shuffle: true, verbose: 0, validationData: [xv, yv] })).history;
      const g = (k: string) => Number((h[k] ?? h[k.replace("acc", "accuracy")] ?? [NaN])[0]);
      const metrics: Record<string, number> = cls
        ? { loss: g("loss"), val_loss: g("val_loss"), acc: g("acc"), val_acc: g("val_acc") }
        : { loss: Math.sqrt(g("loss")) * ys, val_loss: Math.sqrt(g("val_loss")) * ys, r2: 1 - (g("loss") * ys * ys) / tVar, val_r2: 1 - (g("val_loss") * ys * ys) / vVar };
      yield { step: e, total: epochs, metrics, predictor: pr };
    }
  } finally {
    tf.dispose([xs, yt, xv, yv]);
  }
}

async function* knn(c: Ctx): AsyncGenerator<Progress> {
  // shortcut: brute-force kNN on up to 3000 train rows; use a KD-tree or the backend beyond that
  const cap = mulberry32(c.seed);
  const ids = c.Xtr.map((_, i) => i).sort(() => cap() - 0.5).slice(0, 3000);
  const X = ids.map((i) => c.Xtr[i]), Y = ids.map((i) => c.ytr[i]);
  const K = Math.min(c.params.k, X.length);
  const near = (x: number[]) => X.map((r, i) => [dist2(r, x), i]).sort((a, b) => a[0] - b[0]).slice(0, K).map((p) => p[1]);
  const pr: Predictor = c.task === "regression"
    ? { predict: (Q) => Q.map((x) => mean(near(x).map((i) => Y[i]))) }
    : {
        proba: (Q) => Q.map((x) => { const v = new Array<number>(c.k).fill(0); near(x).forEach((i) => v[Y[i]]++); return v.map((n) => n / K); }),
        predict: (Q) => Q.map((x) => { const v = new Array<number>(c.k).fill(0); near(x).forEach((i) => v[Y[i]]++); return argmax(v); }),
      };
  yield { step: 1, total: 1, metrics: score(c, pr), predictor: pr };
}

async function* naiveBayes(c: Ctx): AsyncGenerator<Progress> {
  const d = c.Xtr[0].length;
  const stats = Array.from({ length: c.k }, (_, cl) => {
    const rows = c.Xtr.filter((_, i) => c.ytr[i] === cl);
    return {
      prior: Math.log((rows.length + 1) / (c.Xtr.length + c.k)),
      m: Array.from({ length: d }, (_, j) => mean(rows.map((r) => r[j]))),
      v: Array.from({ length: d }, (_, j) => (std(rows.map((r) => r[j])) ** 2 || 0) + 1e-3),
    };
  });
  const proba = (Q: number[][]) => Q.map((x) => softmax(stats.map((s) => s.prior + x.reduce((a, xv, j) => a - 0.5 * Math.log(2 * Math.PI * s.v[j]) - (xv - s.m[j]) ** 2 / (2 * s.v[j]), 0))));
  const pr: Predictor = { proba, predict: (Q) => proba(Q).map(argmax) };
  yield { step: 1, total: 1, metrics: score(c, pr), predictor: pr };
}

// Grows the tree one depth at a time so you can watch it start to overfit.
async function* tree(c: Ctx): AsyncGenerator<Progress> {
  const cls = c.task === "classification";
  for (let depth = 1; depth <= c.params.maxDepth; depth++) {
    const t = cls
      ? new DecisionTreeClassifier({ gainFunction: "gini", maxDepth: depth, minNumSamples: c.params.minSamples })
      : new DecisionTreeRegression({ maxDepth: depth, minNumSamples: c.params.minSamples });
    t.train(c.Xtr, c.ytr);
    const pr: Predictor = { predict: (X) => (X.length ? t.predict(X) : []) };
    yield { step: depth, total: c.params.maxDepth, metrics: score(c, pr, { depth }), predictor: pr };
  }
}

// Random forest = bagged trees on random feature subspaces, added one tree per step.
async function* forest(c: Ctx): AsyncGenerator<Progress> {
  const r = mulberry32(c.seed);
  const cls = c.task === "classification";
  const d = c.Xtr[0].length, n = c.Xtr.length;
  const m = Math.max(1, Math.round(cls ? Math.sqrt(d) : d / 3));
  const trees: { cols: number[]; t: DecisionTreeClassifier | DecisionTreeRegression }[] = [];
  const pick = (x: number[], cols: number[]) => cols.map((j) => x[j]);
  const votes = (X: number[][]) => X.map((x) => {
    const v = new Array<number>(cls ? c.k : 1).fill(0);
    trees.forEach(({ cols, t }) => { const p = t.predict([pick(x, cols)])[0]; if (cls) v[p]++; else v[0] += p; });
    return v.map((s) => s / trees.length);
  });
  const pr: Predictor = cls
    ? { proba: (X) => votes(X), predict: (X) => votes(X).map(argmax) }
    : { predict: (X) => votes(X).map((v) => v[0]) };
  for (let i = 1; i <= c.params.trees; i++) {
    const rows = Array.from({ length: n }, () => Math.floor(r() * n));
    const cols = Array.from({ length: d }, (_, j) => j).sort(() => r() - 0.5).slice(0, m);
    const t = cls
      ? new DecisionTreeClassifier({ gainFunction: "gini", maxDepth: c.params.maxDepth, minNumSamples: 2 })
      : new DecisionTreeRegression({ maxDepth: c.params.maxDepth, minNumSamples: 2 });
    t.train(rows.map((k) => pick(c.Xtr[k], cols)), rows.map((k) => c.ytr[k]));
    trees.push({ cols, t });
    yield { step: i, total: c.params.trees, metrics: score(c, pr, { trees: i }), predictor: pr };
  }
}

// ---------- unsupervised ----------
export function silhouette(X: number[][], labels: number[], seed = 1) {
  const r = mulberry32(seed);
  const ids = X.map((_, i) => i).filter((i) => labels[i] >= 0).sort(() => r() - 0.5).slice(0, 400);
  const ks = [...new Set(ids.map((i) => labels[i]))];
  if (ks.length < 2) return 0;
  return mean(ids.map((i) => {
    const byK = new Map<number, number[]>();
    ids.forEach((j) => { if (j !== i) { const l = labels[j]; if (!byK.has(l)) byK.set(l, []); byK.get(l)!.push(Math.sqrt(dist2(X[i], X[j]))); } });
    const a = mean(byK.get(labels[i]) ?? [0]);
    const b = Math.min(...ks.filter((k) => k !== labels[i]).map((k) => mean(byK.get(k) ?? [Infinity])));
    return Math.max(a, b) ? (b - a) / Math.max(a, b) : 0;
  }));
}

async function* kmeans(c: Ctx): AsyncGenerator<Progress> {
  const X = c.Xtr, K = c.params.k, r = mulberry32(c.seed);
  // k-means++ init
  const C: number[][] = [X[Math.floor(r() * X.length)].slice()];
  while (C.length < K) {
    const d = X.map((x) => Math.min(...C.map((cc) => dist2(x, cc))));
    let t = r() * d.reduce((a, b) => a + b, 0);
    const i = d.findIndex((v) => (t -= v) <= 0);
    C.push(X[i < 0 ? X.length - 1 : i].slice());
  }
  let labels = new Array<number>(X.length).fill(0);
  for (let it = 1; it <= c.params.iters; it++) {
    const next = X.map((x) => { let bi = 0, bd = Infinity; C.forEach((cc, k) => { const dd = dist2(x, cc); if (dd < bd) { bd = dd; bi = k; } }); return bi; });
    const changed = next.some((l, i) => l !== labels[i]);
    labels = next;
    for (let k = 0; k < K; k++) {
      const mem = X.filter((_, i) => labels[i] === k);
      if (mem.length) C[k] = C[k].map((_, j) => mean(mem.map((m) => m[j])));
    }
    const inertia = X.reduce((s, x, i) => s + dist2(x, C[labels[i]]), 0);
    const done = !changed || it === c.params.iters;
    yield {
      step: it, total: c.params.iters, labels: labels.slice(), centroids: C.map((cc) => cc.slice()),
      metrics: done ? { inertia, silhouette: silhouette(X, labels, c.seed) } : { inertia },
    };
    if (done) return;
  }
}

async function* dbscan(c: Ctx): AsyncGenerator<Progress> {
  const X = c.Xtr, eps2 = c.params.eps ** 2, minPts = c.params.minPts, n = X.length;
  const labels = new Array<number>(n).fill(-2); // -2 unvisited, -1 noise
  const region = (i: number) => { const o: number[] = []; for (let j = 0; j < n; j++) if (dist2(X[i], X[j]) <= eps2) o.push(j); return o; };
  let cid = 0, visited = 0;
  const chunk = Math.max(25, Math.floor(n / 30));
  for (let i = 0; i < n; i++) {
    if (labels[i] !== -2) continue;
    const nb = region(i);
    visited++;
    if (nb.length < minPts) { labels[i] = -1; continue; }
    labels[i] = cid;
    const q = nb.filter((j) => j !== i);
    while (q.length) {
      const j = q.pop()!;
      if (labels[j] === -1) labels[j] = cid;
      if (labels[j] !== -2) continue;
      labels[j] = cid;
      visited++;
      const nb2 = region(j);
      if (nb2.length >= minPts) q.push(...nb2);
      if (visited % chunk === 0) yield { step: visited, total: n, labels: labels.map((l) => (l === -2 ? -1 : l)), metrics: { clusters: cid + 1 } };
    }
    cid++;
  }
  const noise = labels.filter((l) => l < 0).length;
  yield { step: n, total: n, labels: labels.slice(), metrics: { clusters: cid, noise_pct: noise / n, silhouette: silhouette(X, labels, c.seed) } };
}

async function* pca(c: Ctx): AsyncGenerator<Progress> {
  const p = new PCA(c.Xtr);
  const nc = Math.min(3, c.Xtr[0].length);
  const proj = p.predict(c.Xtr, { nComponents: nc }).to2DArray().map((r) => [r[0] ?? 0, r[1] ?? 0, r[2] ?? 0]);
  const explained = p.getExplainedVariance();
  yield {
    step: 1, total: 1, proj, explained,
    metrics: { pc1: explained[0] ?? 0, pc2: explained[1] ?? 0, pc3: explained[2] ?? 0, cumulative_3: explained.slice(0, 3).reduce((a, b) => a + b, 0) },
  };
}

async function* anomaly(c: Ctx): AsyncGenerator<Progress> {
  const X = c.Xtr, K = c.params.k;
  const scores = X.map((x, i) => {
    const d = X.map((y, j) => (i === j ? Infinity : dist2(x, y))).sort((a, b) => a - b);
    return Math.sqrt(d[Math.min(K, d.length) - 1]);
  });
  const cut = [...scores].sort((a, b) => b - a)[Math.max(0, Math.floor(X.length * c.params.contamination) - 1)];
  const labels = scores.map((s) => (s >= cut ? 1 : 0));
  yield { step: 1, total: 1, labels, metrics: { anomalies: labels.filter(Boolean).length, threshold: cut } };
}

const SUP: Task[] = ["classification", "regression"];
export const ENGINES: Engine[] = [
  { id: "linear", label: "Linear Regression", blurb: "Gradient descent on squared error. Watch the loss fall each epoch.", family: "supervised", tasks: ["regression"], run: linear,
    params: [{ key: "epochs", label: "Epochs", min: 10, max: 600, step: 10, def: 150 }, { key: "lr", label: "Learning rate", min: 0.005, max: 0.5, step: 0.005, def: 0.08 }, { key: "l2", label: "L2 penalty", min: 0, max: 0.5, step: 0.01, def: 0 }] },
  { id: "logistic", label: "Logistic Regression", blurb: "Softmax classifier trained by gradient descent. Draws linear boundaries.", family: "supervised", tasks: ["classification"], run: logistic,
    params: [{ key: "epochs", label: "Epochs", min: 10, max: 600, step: 10, def: 150 }, { key: "lr", label: "Learning rate", min: 0.01, max: 2, step: 0.01, def: 0.3 }, { key: "l2", label: "L2 penalty", min: 0, max: 0.5, step: 0.01, def: 0.001 }] },
  { id: "mlp", label: "Neural Network (TF.js)", blurb: "Multi-layer perceptron trained with Adam on TensorFlow.js (WebGL).", family: "supervised", tasks: SUP, run: mlp,
    params: [{ key: "epochs", label: "Epochs", min: 5, max: 300, step: 5, def: 60 }, { key: "hidden", label: "Units / layer", min: 4, max: 128, step: 4, def: 32 }, { key: "layers", label: "Hidden layers", min: 1, max: 4, step: 1, def: 2 }, { key: "lr", label: "Learning rate", min: 0.0005, max: 0.05, step: 0.0005, def: 0.01 }, { key: "batch", label: "Batch size", min: 8, max: 256, step: 8, def: 32 }] },
  { id: "tree", label: "Decision Tree", blurb: "CART tree grown one depth at a time. See where validation peaks.", family: "supervised", tasks: SUP, run: tree,
    params: [{ key: "maxDepth", label: "Max depth", min: 1, max: 20, step: 1, def: 10 }, { key: "minSamples", label: "Min samples / leaf", min: 1, max: 30, step: 1, def: 3 }] },
  { id: "forest", label: "Random Forest", blurb: "Bagged trees on random feature subsets, added one tree at a time.", family: "supervised", tasks: SUP, run: forest,
    params: [{ key: "trees", label: "Trees", min: 1, max: 100, step: 1, def: 30 }, { key: "maxDepth", label: "Max depth", min: 2, max: 20, step: 1, def: 8 }] },
  { id: "knn", label: "k-Nearest Neighbors", blurb: "No training: predicts from the k closest examples.", family: "supervised", tasks: SUP, run: knn,
    params: [{ key: "k", label: "k", min: 1, max: 40, step: 1, def: 7 }] },
  { id: "nb", label: "Gaussian Naive Bayes", blurb: "Probabilistic baseline assuming independent features.", family: "supervised", tasks: ["classification"], run: naiveBayes, params: [] },
  { id: "kmeans", label: "k-Means", blurb: "Centroids move each iteration until clusters stop changing.", family: "unsupervised", tasks: ["clustering"], run: kmeans,
    params: [{ key: "k", label: "Clusters (k)", min: 2, max: 10, step: 1, def: 3 }, { key: "iters", label: "Max iterations", min: 1, max: 100, step: 1, def: 40 }] },
  { id: "dbscan", label: "DBSCAN", blurb: "Density clustering that finds odd shapes and marks noise.", family: "unsupervised", tasks: ["clustering"], run: dbscan,
    params: [{ key: "eps", label: "Radius (eps)", min: 0.05, max: 3, step: 0.05, def: 0.6 }, { key: "minPts", label: "Min points", min: 2, max: 30, step: 1, def: 5 }] },
  { id: "pca", label: "PCA", blurb: "Projects data onto the directions of most variance.", family: "unsupervised", tasks: ["clustering"], run: pca, params: [] },
  { id: "anomaly", label: "Anomaly Detection", blurb: "Flags points far from their k-th nearest neighbour.", family: "unsupervised", tasks: ["clustering"], run: anomaly,
    params: [{ key: "k", label: "Neighbour k", min: 2, max: 30, step: 1, def: 5 }, { key: "contamination", label: "Expected anomalies", min: 0.01, max: 0.3, step: 0.01, def: 0.05 }] },
];

export function projection(X: number[][]) {
  if (!X.length) return { pts: [] as number[][], explained: [] as number[] };
  if (X[0].length <= 3) return { pts: X.map((r) => [r[0] ?? 0, r[1] ?? 0, r[2] ?? 0]), explained: [] };
  const p = new PCA(X);
  return { pts: p.predict(X, { nComponents: 3 }).to2DArray(), explained: p.getExplainedVariance() };
}
