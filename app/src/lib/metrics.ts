import { mean, sum } from "./core";

export function confusion(y: number[], p: number[], k: number) {
  const m = Array.from({ length: k }, () => new Array<number>(k).fill(0));
  y.forEach((t, i) => { if (p[i] >= 0 && p[i] < k) m[t][p[i]]++; });
  return m;
}

export function classReport(y: number[], p: number[], k: number) {
  const cm = confusion(y, p, k);
  const per = Array.from({ length: k }, (_, c) => {
    const tp = cm[c][c];
    const fp = sum(cm.map((r) => r[c])) - tp;
    const fn = sum(cm[c]) - tp;
    const precision = tp + fp ? tp / (tp + fp) : 0;
    const recall = tp + fn ? tp / (tp + fn) : 0;
    return { precision, recall, f1: precision + recall ? (2 * precision * recall) / (precision + recall) : 0, support: tp + fn };
  });
  return {
    cm, per,
    accuracy: y.length ? y.filter((t, i) => t === p[i]).length / y.length : 0,
    macroF1: mean(per.map((x) => x.f1)),
    // majority-class baseline: a model has to beat this to mean anything
    baseline: y.length ? Math.max(...per.map((x) => x.support)) / y.length : 0,
  };
}

export function roc(scores: number[], positive: boolean[]) {
  const idx = scores.map((_, i) => i).sort((a, b) => scores[b] - scores[a]);
  const P = positive.filter(Boolean).length;
  const N = positive.length - P;
  if (!P || !N) return { pts: [{ fpr: 0, tpr: 0 }, { fpr: 1, tpr: 1 }], auc: NaN };
  let tp = 0, fp = 0, auc = 0, pf = 0, pt = 0;
  const pts = [{ fpr: 0, tpr: 0 }];
  for (const i of idx) {
    if (positive[i]) tp++; else fp++;
    const f = fp / N, t = tp / P;
    auc += ((f - pf) * (t + pt)) / 2;
    pf = f; pt = t;
    pts.push({ fpr: f, tpr: t });
  }
  const step = Math.max(1, Math.floor(pts.length / 150));
  return { pts: pts.filter((_, i) => i % step === 0 || i === pts.length - 1), auc };
}

export function regReport(y: number[], p: number[]) {
  const my = mean(y);
  const ssr = sum(y.map((t, i) => (t - p[i]) ** 2));
  const sst = sum(y.map((t) => (t - my) ** 2));
  return {
    rmse: Math.sqrt(ssr / (y.length || 1)),
    mae: mean(y.map((t, i) => Math.abs(t - p[i]))),
    r2: sst ? 1 - ssr / sst : 0,
    baselineRmse: Math.sqrt(sst / (y.length || 1)), // always-predict-the-mean baseline
  };
}

export function selfCheck() {
  const r = classReport([0, 0, 1, 1], [0, 1, 1, 1], 2);
  console.assert(r.accuracy === 0.75 && r.cm[0][1] === 1, "classReport");
  console.assert(Math.abs(roc([0.9, 0.8, 0.2, 0.1], [true, true, false, false]).auc - 1) < 1e-9, "roc auc perfect");
  console.assert(regReport([1, 2, 3], [1, 2, 3]).r2 === 1, "r2 perfect");
}
