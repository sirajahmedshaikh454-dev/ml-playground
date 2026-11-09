import { type Dataset, type Row, type Task, mean, mulberry32, shuffle, std } from "./core";

export interface FeatureCfg {
  target: string | null;
  task: Task;
  features: string[];
  encode: "onehot" | "label";
  scale: "standard" | "minmax" | "none";
  testSize: number; // 0..0.5
  seed: number;
  stratify: boolean;
  textFeatures: number;
}

export interface Prepared {
  Xtr: number[][]; ytr: number[];
  Xte: number[][]; yte: number[];
  Xall: number[][]; yall: number[];
  names: string[];
  classes: string[];
  nTrain: number; nTest: number;
  dropped: number;
}

const STOP = new Set("the and for are but not you all any can had her was one our out has have this that with from they will would there their what about which when your said them been than then into some could other".split(" "));
const tok = (s: string) => s.toLowerCase().split(/[^a-z0-9]+/).filter((w) => w.length > 2 && !STOP.has(w));

type Col =
  | { kind: "num"; name: string; mean: number; std: number; min: number; max: number }
  | { kind: "cat"; name: string; levels: string[] }
  | { kind: "text"; name: string; vocab: string[] };

export function defaultCfg(ds: Dataset): FeatureCfg {
  const text = ds.types["text"] === "text" && ds.columns.includes("label");
  const target = text ? "label" : ds.columns[ds.columns.length - 1];
  const t = ds.types[target];
  const task: Task = t === "number" && new Set(ds.rows.map((r) => r[target])).size > 10 ? "regression" : "classification";
  return {
    target, task,
    features: ds.columns.filter((c) => c !== target),
    encode: "onehot", scale: "standard", testSize: 0.2, seed: 42, stratify: true, textFeatures: 100,
  };
}

// Every stateful transform (levels, means, vocab) is fitted on TRAIN rows only -> no leakage.
export function prepare(ds: Dataset, cfg: FeatureCfg): Prepared {
  const supervised = cfg.task !== "clustering" && !!cfg.target;
  const all = supervised ? ds.rows.filter((r) => r[cfg.target!] != null) : ds.rows;
  const dropped = ds.rows.length - all.length;
  const feats = cfg.features.filter((f) => f !== cfg.target && ds.columns.includes(f));

  const classes = supervised && cfg.task === "classification"
    ? [...new Set(all.map((r) => String(r[cfg.target!])))].sort()
    : [];
  const yOf = (r: Row): number => {
    if (!supervised) return 0;
    const v = r[cfg.target!];
    return cfg.task === "classification" ? classes.indexOf(String(v)) : Number(v);
  };

  // split (seeded, optionally stratified)
  const rng = mulberry32(cfg.seed);
  const idx = all.map((_, i) => i);
  const test: number[] = [];
  if (supervised && cfg.testSize > 0) {
    if (cfg.task === "classification" && cfg.stratify) {
      const groups = new Map<string, number[]>();
      idx.forEach((i) => {
        const k = String(all[i][cfg.target!]);
        if (!groups.has(k)) groups.set(k, []);
        groups.get(k)!.push(i);
      });
      for (const g of groups.values())
        if (g.length > 1) test.push(...shuffle(g, rng).slice(0, Math.max(1, Math.round(g.length * cfg.testSize))));
    } else test.push(...shuffle(idx, rng).slice(0, Math.round(all.length * cfg.testSize)));
  }
  const testSet = new Set(test);
  const train = idx.filter((i) => !testSet.has(i));
  const trainRows = train.map((i) => all[i]);

  const cols: Col[] = feats.map((name): Col => {
    const t = ds.types[name];
    if (t === "number" || t === "boolean") {
      const v = trainRows.map((r) => (r[name] == null ? NaN : Number(r[name]))).filter(Number.isFinite);
      return { kind: "num", name, mean: mean(v), std: std(v) || 1, min: v.length ? Math.min(...v) : 0, max: v.length ? Math.max(...v) : 1 };
    }
    if (t === "text") {
      const df = new Map<string, number>();
      trainRows.forEach((r) => new Set(tok(String(r[name] ?? ""))).forEach((w) => df.set(w, (df.get(w) ?? 0) + 1)));
      return { kind: "text", name, vocab: [...df.entries()].sort((a, b) => b[1] - a[1]).slice(0, cfg.textFeatures).map((e) => e[0]) };
    }
    const cnt = new Map<string, number>();
    trainRows.forEach((r) => { if (r[name] != null) cnt.set(String(r[name]), (cnt.get(String(r[name])) ?? 0) + 1); });
    return { kind: "cat", name, levels: [...cnt.entries()].sort((a, b) => b[1] - a[1]).slice(0, 25).map((e) => e[0]) };
  });

  const names: string[] = [];
  cols.forEach((c) => {
    if (c.kind === "num") names.push(c.name);
    else if (c.kind === "text") c.vocab.forEach((w) => names.push(`${c.name}:${w}`));
    else if (cfg.encode === "label") names.push(c.name);
    else c.levels.forEach((l) => names.push(`${c.name}=${l}`));
  });

  const enc = (r: Row): number[] => {
    const out: number[] = [];
    for (const c of cols) {
      if (c.kind === "num") {
        const raw = r[c.name] == null ? NaN : Number(r[c.name]);
        const v = Number.isFinite(raw) ? raw : c.mean; // impute with train mean
        out.push(cfg.scale === "standard" ? (v - c.mean) / c.std : cfg.scale === "minmax" ? (v - c.min) / (c.max - c.min || 1) : v);
      } else if (c.kind === "text") {
        const w = tok(String(r[c.name] ?? ""));
        const n = w.length || 1;
        const cnt = new Map<string, number>();
        w.forEach((x) => cnt.set(x, (cnt.get(x) ?? 0) + 1));
        c.vocab.forEach((term) => out.push((cnt.get(term) ?? 0) / n));
      } else if (cfg.encode === "label") out.push(Math.max(0, c.levels.indexOf(String(r[c.name]))));
      else c.levels.forEach((l) => out.push(String(r[c.name]) === l ? 1 : 0));
    }
    return out;
  };

  const part = (ids: number[]) => ({ X: ids.map((i) => enc(all[i])), y: ids.map((i) => yOf(all[i])) });
  const tr = part(train);
  const te = test.length ? part(test) : tr;
  const al = part(idx);
  return { Xtr: tr.X, ytr: tr.y, Xte: te.X, yte: te.y, Xall: al.X, yall: al.y, names, classes, nTrain: train.length, nTest: test.length, dropped };
}

// shortcut: self-check runs in dev only, move to a test runner if the suite grows.
export function selfCheck() {
  const ds: Dataset = {
    name: "t", columns: ["a", "c", "y"], types: { a: "number", c: "category", y: "category" },
    rows: Array.from({ length: 40 }, (_, i) => ({ a: i, c: i % 2 ? "x" : "z", y: i % 2 ? "p" : "q" })),
  };
  const p = prepare(ds, { ...defaultCfg(ds), testSize: 0.25 });
  console.assert(p.nTrain + p.nTest === 40 && p.nTest === 10, "split sizes");
  console.assert(p.names.join() === "a,c=x,c=z" || p.names.join() === "a,c=z,c=x", "one-hot names");
  console.assert(Math.abs(p.Xtr.reduce((s, r) => s + r[0], 0)) < 1e-9, "train standardized to mean 0");
}
