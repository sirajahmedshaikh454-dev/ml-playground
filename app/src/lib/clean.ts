import { type ColType, type Dataset, type Row, mean, quantile, std } from "./core";

// Cleaning is a serializable list of steps: replayable, undoable, exportable.
export type CleanOp =
  | { op: "dedupe" }
  | { op: "dropCol"; col: string }
  | { op: "dropMissing"; col?: string }
  | { op: "fill"; col: string; how: "mean" | "median" | "mode" | "zero" | "unknown" }
  | { op: "outliers"; col: string; how: "clip" | "remove"; k: number }
  | { op: "cast"; col: string; to: ColType };

export function describeOp(o: CleanOp): string {
  switch (o.op) {
    case "dedupe": return "Remove duplicate rows";
    case "dropCol": return `Drop column ${o.col}`;
    case "dropMissing": return o.col ? `Drop rows missing ${o.col}` : "Drop rows with any missing value";
    case "fill": return `Fill ${o.col} missing with ${o.how}`;
    case "outliers": return `${o.how === "clip" ? "Clip" : "Remove"} outliers in ${o.col} (IQR x${o.k})`;
    case "cast": return `Convert ${o.col} to ${o.to}`;
  }
}

const nums = (rows: Row[], c: string) =>
  rows.map((r) => r[c]).filter((v): v is number => typeof v === "number" && Number.isFinite(v));

function fences(rows: Row[], c: string, k: number): [number, number] {
  const s = nums(rows, c).sort((a, b) => a - b);
  const q1 = quantile(s, 0.25), q3 = quantile(s, 0.75), i = q3 - q1;
  return [q1 - k * i, q3 + k * i];
}

export function applyOps(ds: Dataset, ops: CleanOp[]): Dataset {
  let d: Dataset = { ...ds };
  for (const o of ops) {
    const rows = d.rows;
    switch (o.op) {
      case "dedupe": {
        const seen = new Set<string>();
        d = { ...d, rows: rows.filter((r) => { const k = JSON.stringify(d.columns.map((c) => r[c])); return seen.has(k) ? false : (seen.add(k), true); }) };
        break;
      }
      case "dropCol":
        d = { ...d, columns: d.columns.filter((c) => c !== o.col), rows: rows.map((r) => { const n = { ...r }; delete n[o.col]; return n; }) };
        break;
      case "dropMissing":
        d = { ...d, rows: rows.filter((r) => (o.col ? r[o.col] != null : d.columns.every((c) => r[c] != null))) };
        break;
      case "fill": {
        let v: string | number | boolean | null = null;
        const vals = rows.map((r) => r[o.col]).filter((x) => x != null);
        if (o.how === "zero") v = 0;
        else if (o.how === "unknown") v = "unknown";
        else if (o.how === "mean") v = mean(nums(rows, o.col));
        else if (o.how === "median") v = quantile(nums(rows, o.col).sort((a, b) => a - b), 0.5);
        else {
          const cnt = new Map<unknown, number>();
          vals.forEach((x) => cnt.set(x, (cnt.get(x) ?? 0) + 1));
          v = [...cnt.entries()].sort((a, b) => b[1] - a[1])[0]?.[0] as string | number | boolean ?? null;
        }
        const types = { ...d.types };
        if (o.how === "unknown" && types[o.col] === "number") types[o.col] = "category";
        d = { ...d, types, rows: rows.map((r) => (r[o.col] == null ? { ...r, [o.col]: v } : r)) };
        break;
      }
      case "outliers": {
        const [lo, hi] = fences(rows, o.col, o.k);
        d = {
          ...d,
          rows: o.how === "remove"
            ? rows.filter((r) => typeof r[o.col] !== "number" || ((r[o.col] as number) >= lo && (r[o.col] as number) <= hi))
            : rows.map((r) => (typeof r[o.col] === "number" ? { ...r, [o.col]: Math.min(hi, Math.max(lo, r[o.col] as number)) } : r)),
        };
        break;
      }
      case "cast": {
        const f = (v: Row[string]) => {
          if (v == null) return null;
          if (o.to === "number") { const n = Number(v); return Number.isFinite(n) ? n : null; }
          if (o.to === "boolean") return String(v).toLowerCase() === "true" || v === 1 || v === "1";
          return String(v);
        };
        d = { ...d, types: { ...d.types, [o.col]: o.to }, rows: rows.map((r) => ({ ...r, [o.col]: f(r[o.col]) })) };
        break;
      }
    }
  }
  return d;
}

export interface ColProfile {
  name: string;
  type: ColType;
  missing: number;
  unique: number;
  mean?: number; std?: number; min?: number; max?: number; median?: number;
  outliers?: number;
  top?: [string, number][];
}

export function profile(ds: Dataset): ColProfile[] {
  return ds.columns.map((name) => {
    const type = ds.types[name];
    const vals = ds.rows.map((r) => r[name]);
    const present = vals.filter((v) => v != null);
    const p: ColProfile = { name, type, missing: vals.length - present.length, unique: new Set(present).size };
    if (type === "number") {
      const s = nums(ds.rows, name).sort((a, b) => a - b);
      if (s.length) {
        const [lo, hi] = fences(ds.rows, name, 1.5);
        Object.assign(p, { mean: mean(s), std: std(s), min: s[0], max: s[s.length - 1], median: quantile(s, 0.5), outliers: s.filter((v) => v < lo || v > hi).length });
      }
    } else {
      const cnt = new Map<string, number>();
      present.forEach((v) => cnt.set(String(v), (cnt.get(String(v)) ?? 0) + 1));
      p.top = [...cnt.entries()].sort((a, b) => b[1] - a[1]).slice(0, 6);
    }
    return p;
  });
}

export function histogram(values: number[], bins = 20) {
  if (!values.length) return [];
  const lo = Math.min(...values), hi = Math.max(...values);
  const w = (hi - lo) / bins || 1;
  const out = Array.from({ length: bins }, (_, i) => ({ x: +(lo + i * w).toFixed(3), count: 0 }));
  values.forEach((v) => { out[Math.min(bins - 1, Math.floor((v - lo) / w))].count++; });
  return out;
}

export function correlation(ds: Dataset) {
  const cols = ds.columns.filter((c) => ds.types[c] === "number").slice(0, 14);
  const m = cols.map((a) =>
    cols.map((b) => {
      const xs: number[] = [], ys: number[] = [];
      for (const r of ds.rows) if (typeof r[a] === "number" && typeof r[b] === "number") { xs.push(r[a] as number); ys.push(r[b] as number); }
      const mx = mean(xs), my = mean(ys);
      let sxy = 0, sxx = 0, syy = 0;
      xs.forEach((x, i) => { sxy += (x - mx) * (ys[i] - my); sxx += (x - mx) ** 2; syy += (ys[i] - my) ** 2; });
      return sxx && syy ? sxy / Math.sqrt(sxx * syy) : 0;
    }),
  );
  return { cols, m };
}
