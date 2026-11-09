"use client";
import Dexie, { type Table } from "dexie";
import { create } from "zustand";
import { applyOps, type CleanOp } from "./clean";
import type { Dataset, Predictor, Task } from "./core";
import { defaultCfg, type FeatureCfg } from "./features";

// ---------- persisted experiment runs (IndexedDB) ----------
export interface RunRecord {
  id?: number;
  createdAt: number;
  dataset: string;
  rows: number;
  engine: string;
  engineLabel: string;
  task: Task;
  params: Record<string, number>;
  seed: number;
  testSize: number;
  features: number;
  test: Record<string, number>;
  history: Record<string, number>[];
  ms: number;
}
class DB extends Dexie {
  runs!: Table<RunRecord, number>;
  constructor() {
    super("ml-playground");
    this.version(1).stores({ runs: "++id, createdAt, dataset, engine" });
  }
}
let db: DB | null = null;
export const getDb = () => (db ??= new DB());

// ---------- in-memory lab state ----------
export interface EvalResult {
  task: Task;
  engine: string;
  y: number[];
  pred: number[];
  proba?: number[][];
  classes: string[];
  predictor: Predictor;
}

interface Lab {
  raw: Dataset | null;
  ops: CleanOp[];
  data: Dataset | null; // raw + ops
  cfg: FeatureCfg | null;
  step: number;
  evalResult: EvalResult | null;
  load: (d: Dataset) => void;
  addOp: (o: CleanOp) => void;
  undo: () => void;
  resetOps: () => void;
  setCfg: (c: Partial<FeatureCfg>) => void;
  setStep: (s: number) => void;
  setEval: (e: EvalResult | null) => void;
}

export const useLab = create<Lab>((set, get) => ({
  raw: null, ops: [], data: null, cfg: null, step: 0, evalResult: null,
  load: (d) => {
    get().evalResult?.predictor.dispose?.();
    set({ raw: d, ops: [], data: d, cfg: defaultCfg(d), step: 1, evalResult: null });
  },
  addOp: (o) => {
    const ops = [...get().ops, o];
    const data = applyOps(get().raw!, ops);
    const cfg = get().cfg!;
    set({ ops, data, cfg: { ...cfg, features: cfg.features.filter((f) => data.columns.includes(f)), target: cfg.target && data.columns.includes(cfg.target) ? cfg.target : data.columns[data.columns.length - 1] } });
  },
  undo: () => {
    const ops = get().ops.slice(0, -1);
    set({ ops, data: applyOps(get().raw!, ops) });
  },
  resetOps: () => set({ ops: [], data: get().raw }),
  setCfg: (c) => set({ cfg: { ...get().cfg!, ...c } }),
  setStep: (step) => set({ step }),
  setEval: (e) => {
    const prev = get().evalResult;
    if (prev && prev.predictor !== e?.predictor) prev.predictor.dispose?.();
    set({ evalResult: e });
  },
}));
