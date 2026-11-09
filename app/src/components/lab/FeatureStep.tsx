"use client";
import { SlidersHorizontal, Split } from "lucide-react";
import { useMemo } from "react";
import type { Task } from "@/lib/core";
import { prepare } from "@/lib/features";
import { useLab } from "@/lib/store";
import { Bars, colorOf } from "../charts";
import { Badge, Card, Field, Select, Slider, Stat, cx } from "../ui";

export function usePrepared() {
  const { data, cfg } = useLab();
  return useMemo(() => {
    if (!data || !cfg) return { p: null, error: "Load data first" };
    if (cfg.task === "regression" && cfg.target && data.types[cfg.target] !== "number")
      return { p: null, error: `Regression needs a numeric target; "${cfg.target}" is ${data.types[cfg.target]}.` };
    if (!cfg.features.length) return { p: null, error: "Select at least one feature." };
    try {
      const p = prepare(data, cfg);
      if (!p.Xtr.length || !p.names.length) return { p: null, error: "No usable rows or features after preparation." };
      if (cfg.task === "classification" && p.classes.length < 2) return { p: null, error: "Classification needs at least two classes in the target." };
      if (cfg.task === "classification" && p.classes.length > 50) return { p: null, error: `Target has ${p.classes.length} classes — did you mean regression?` };
      return { p, error: null };
    } catch (e) { return { p: null, error: (e as Error).message }; }
  }, [data, cfg]);
}

export function FeatureStep() {
  const { data, cfg, setCfg } = useLab();
  const { p, error } = usePrepared();
  if (!data || !cfg) return null;
  const supervised = cfg.task !== "clustering";
  const balance = p?.classes.map((c, i) => ({ x: c, count: p.ytr.filter((y) => y === i).length })) ?? [];

  return (
    <div className="grid grid-cols-1 gap-5 lg:grid-cols-[1fr_1.1fr]">
      <Card title="What should the model learn?" icon={<SlidersHorizontal className="size-4" />}>
        <div className="space-y-4">
          <div role="radiogroup" aria-label="Task" className="grid grid-cols-3 gap-2">
            {([["classification", "Classify", "Predict a category"], ["regression", "Regress", "Predict a number"], ["clustering", "Discover", "Find structure, no labels"]] as [Task, string, string][]).map(([t, l, d]) => (
              <button key={t} role="radio" aria-checked={cfg.task === t} onClick={() => setCfg({ task: t })}
                className={cx("cursor-pointer rounded-xl border p-3 text-left transition-colors", cfg.task === t ? "border-accent bg-accent/10" : "border-border bg-surface/60 hover:border-primary/50")}>
                <div className="text-sm font-medium">{l}</div>
                <div className="text-[11px] text-dim">{d}</div>
              </button>
            ))}
          </div>

          {supervised && (
            <Select label="Target column" value={cfg.target ?? ""} onChange={(v) => setCfg({ target: v, features: cfg.features.filter((f) => f !== v) })}
              options={data.columns.map((c) => ({ value: c, label: `${c} (${data.types[c]})` }))} />
          )}

          <Field label={`Features (${cfg.features.length} selected)`}>
            <div className="scroll-thin flex max-h-40 flex-wrap gap-1.5 overflow-auto rounded-lg border border-border bg-surface/60 p-2">
              {data.columns.filter((c) => !supervised || c !== cfg.target).map((c) => {
                const on = cfg.features.includes(c);
                return (
                  <button key={c} aria-pressed={on} onClick={() => setCfg({ features: on ? cfg.features.filter((f) => f !== c) : [...cfg.features, c] })}
                    className={cx("cursor-pointer rounded-md border px-2 py-1 text-xs transition-colors", on ? "border-primary/60 bg-primary/15 text-fg" : "border-border text-dim hover:text-fg")}>
                    {c}
                  </button>
                );
              })}
            </div>
          </Field>

          <div className="grid grid-cols-2 gap-3">
            <Select label="Categorical encoding" value={cfg.encode} onChange={(v) => setCfg({ encode: v })}
              options={[{ value: "onehot", label: "One-hot" }, { value: "label", label: "Label (ordinal)" }]} />
            <Select label="Scaling" value={cfg.scale} onChange={(v) => setCfg({ scale: v })}
              options={[{ value: "standard", label: "Standardize (z-score)" }, { value: "minmax", label: "Min-max 0..1" }, { value: "none", label: "None" }]} />
          </div>
          {data.columns.some((c) => data.types[c] === "text") && (
            <Slider label="Text vocabulary size" value={cfg.textFeatures} min={10} max={500} step={10} onChange={(v) => setCfg({ textFeatures: v })} />
          )}
        </div>
      </Card>

      <div className="space-y-5">
        <Card title="Train / test split" icon={<Split className="size-4" />}>
          {supervised ? (
            <div className="space-y-4">
              <Slider label="Test set size" value={cfg.testSize} min={0.1} max={0.5} step={0.05} onChange={(v) => setCfg({ testSize: v })} fmt={(v) => `${Math.round(v * 100)}%`} />
              <div className="grid grid-cols-2 gap-3">
                <Field label="Random seed">
                  <input type="number" value={cfg.seed} onChange={(e) => setCfg({ seed: Number(e.target.value) || 0 })}
                    className="num h-10 w-full rounded-lg border border-border bg-surface px-3 text-sm" />
                </Field>
                {cfg.task === "classification" && (
                  <label className="mt-5 flex cursor-pointer items-center gap-2 text-sm">
                    <input type="checkbox" checked={cfg.stratify} onChange={(e) => setCfg({ stratify: e.target.checked })} className="size-4 accent-amber-500" />
                    Stratify by class
                  </label>
                )}
              </div>
              <p className="text-xs text-dim">The split happens <b className="text-sub">before</b> scaling and encoding, which are fitted on the training rows only — so the test score is honest.</p>
            </div>
          ) : <p className="text-sm text-dim">Clustering uses every row; there is no held-out set.</p>}
        </Card>

        <Card title="Ready to train?">
          {error ? <p role="alert" className="rounded-lg border border-bad/40 bg-bad/10 px-3 py-2 text-sm text-bad">{error}</p> : p && (
            <div className="space-y-4">
              <div className="grid grid-cols-3 gap-3">
                <Stat label="Features" value={p.names.length} sub={`from ${cfg.features.length} columns`} />
                <Stat label="Train rows" value={p.nTrain.toLocaleString()} />
                <Stat label="Test rows" value={supervised ? p.nTest.toLocaleString() : "—"} />
              </div>
              {p.dropped > 0 && <p className="text-xs text-accent">{p.dropped} rows without a target were left out.</p>}
              {cfg.task === "classification" && (
                <div>
                  <div className="mb-1 flex items-center gap-2 text-xs text-dim">Class balance (train) {balance.length && Math.max(...balance.map((b) => b.count)) > 3 * Math.min(...balance.map((b) => b.count)) && <Badge tone="amber">imbalanced — watch F1, not just accuracy</Badge>}</div>
                  <Bars data={balance} xKey="x" yKey="count" color={colorOf(1)} height={130} label="Class balance in the training set" />
                </div>
              )}
            </div>
          )}
        </Card>
      </div>
    </div>
  );
}
