// Shared types + small numeric helpers.
export type ColType = "number" | "category" | "text" | "boolean";
export type Cell = string | number | boolean | null;
export type Row = Record<string, Cell>;
export interface Dataset {
  name: string;
  columns: string[];
  types: Record<string, ColType>;
  rows: Row[];
  note?: string;
}
export type Task = "classification" | "regression" | "clustering";

export interface Predictor {
  predict: (X: number[][]) => number[];
  proba?: (X: number[][]) => number[][];
  dispose?: () => void;
}
export interface Progress {
  step: number;
  total: number;
  metrics: Record<string, number>;
  predictor?: Predictor;
  // clustering payloads
  labels?: number[];
  centroids?: number[][];
}

export function mulberry32(seed: number) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
export const gauss = (r: () => number) =>
  Math.sqrt(-2 * Math.log(1 - r())) * Math.cos(2 * Math.PI * r());
export function shuffle<T>(a: T[], r: () => number) {
  const b = a.slice();
  for (let i = b.length - 1; i > 0; i--) {
    const j = Math.floor(r() * (i + 1));
    [b[i], b[j]] = [b[j], b[i]];
  }
  return b;
}

export const sum = (a: number[]) => a.reduce((s, v) => s + v, 0);
export const mean = (a: number[]) => (a.length ? sum(a) / a.length : 0);
export const std = (a: number[]) => {
  if (a.length < 2) return 0;
  const m = mean(a);
  return Math.sqrt(sum(a.map((v) => (v - m) ** 2)) / (a.length - 1));
};
export function quantile(sorted: number[], q: number) {
  if (!sorted.length) return NaN;
  const p = (sorted.length - 1) * q;
  const lo = Math.floor(p);
  const hi = Math.ceil(p);
  return sorted[lo] + (sorted[hi] - sorted[lo]) * (p - lo);
}
export const dot = (a: number[], b: number[]) => {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i] * b[i];
  return s;
};
export const dist2 = (a: number[], b: number[]) => {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += (a[i] - b[i]) ** 2;
  return s;
};
export const isMissing = (v: unknown) =>
  v == null ||
  (typeof v === "string" &&
    ["", "na", "n/a", "nan", "null", "none", "?", "-"].includes(v.trim().toLowerCase()));

export const PALETTE = [
  "#60A5FA", "#F59E0B", "#34D399", "#F472B6",
  "#A78BFA", "#22D3EE", "#FB7185", "#A3E635",
];
