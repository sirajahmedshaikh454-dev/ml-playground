"use client";
import dynamic from "next/dynamic";
import { Activity, ArrowRight, Cpu, Pause, Play, Square, StepForward, TriangleAlert } from "lucide-react";
import { useEffect, useMemo, useRef, useState } from "react";
import { type Progress, mean, mulberry32, predictAll, shuffle } from "@/lib/core";
import type { Prepared } from "@/lib/features";
import { classReport, regReport } from "@/lib/metrics";
import { type Ctx, type Engine, projection } from "@/lib/ml";
import { BACKEND, REMOTE_ENGINES, backendModels } from "@/lib/remote";
import { WORKER_ENGINES } from "@/lib/workerEngine";
import { getDb, type RunRecord, useLab } from "@/lib/store";
import { useRunner } from "@/lib/useRunner";
import { Bars, type Bg, Curves, type Pt, Scatter2D, colorOf } from "../charts";
import { Badge, Button, Card, Select, Slider, Stat, cx, fmt, pct } from "../ui";
import { usePrepared } from "./FeatureStep";

const PointCloud3D = dynamic(() => import("../Scene3D"), { ssr: false, loading: () => <div className="h-[300px] animate-pulse rounded-xl bg-muted/40" /> });

const SPEEDS = [{ value: "0", label: "Max speed" }, { value: "40", label: "Fast" }, { value: "150", label: "Normal" }, { value: "450", label: "Slow-mo" }];
const RES = 48;

interface Live { ctx: Ctx; p: Prepared; engine: Engine; t0: number; X: number[][] }

export function TrainStep() {
  const { data, cfg, setEval, setStep } = useLab();
  const { p, error } = usePrepared();
  const runner = useRunner<Progress>();
  const task = cfg?.task ?? "classification";
  const [pyModels, setPyModels] = useState<string[] | null>(null);
  useEffect(() => { void backendModels().then(setPyModels); }, []);
  const engines = [...WORKER_ENGINES, ...REMOTE_ENGINES.filter((e) => pyModels?.includes(e.id.slice(3)))].filter((e) => e.tasks.includes(task));
  const [engineId, setEngineId] = useState(engines[0]?.id);
  const engine = engines.find((e) => e.id === engineId) ?? engines[0];
  const [paramState, setParamState] = useState<Record<string, Record<string, number>>>({});
  const [axes, setAxes] = useState<[number, number]>([0, 1]);
  const [axisMode, setAxisMode] = useState<"features" | "pca">("pca");
  const [speed, setSpeed] = useState("0");
  const [saved, setSaved] = useState<number | null>(null);
  const [L, setL] = useState<Live | null>(null); // the run being shown (state, not a ref: render reads it)
  const busy = runner.status === "running" || runner.status === "paused";

  const param = (k: string, def: number) => paramState[engine.id]?.[k] ?? def;
  const supervised = task !== "clustering";
  const names = p?.names ?? [];
  const fx = Math.min(axes[0], names.length - 1), fy = Math.min(axes[1], Math.max(0, names.length - 1));

  const start = () => {
    if (!p || !cfg || !data) return;
    const params = Object.fromEntries(engine.params.map((q) => [q.key, param(q.key, q.def)]));
    let ctx: Ctx;
    if (supervised) {
      // carve a validation split out of TRAIN for the live curves; test stays untouched until the end
      const ids = shuffle(p.Xtr.map((_, i) => i), mulberry32(cfg.seed + 1));
      const nVal = Math.max(1, Math.round(ids.length * 0.15));
      const va = ids.slice(0, nVal), fit = ids.slice(nVal);
      ctx = { task, Xtr: fit.map((i) => p.Xtr[i]), ytr: fit.map((i) => p.ytr[i]), Xva: va.map((i) => p.Xtr[i]), yva: va.map((i) => p.ytr[i]), k: p.classes.length, params, seed: cfg.seed };
    } else {
      // shortcut: O(n^2) clustering capped at 4000 rows; switch to the backend for more
      const X = p.Xall.length > 4000 ? shuffle(p.Xall, mulberry32(cfg.seed)).slice(0, 4000) : p.Xall;
      ctx = { task, Xtr: X, ytr: [], Xva: [], yva: [], k: 0, params, seed: cfg.seed };
    }
    // eslint-disable-next-line react-hooks/purity -- click handler, not render
    const run: Live = { ctx, p, engine, t0: performance.now(), X: ctx.Xtr };
    setL(run);
    setSaved(null);
    if (supervised) setEval(null);
    runner.setDelay(Number(speed));
    runner.start(engine.run(ctx), (all) => void finish(all, run));
  };

  const finish = async (all: Progress[], L: Live) => {
    const last = all[all.length - 1];
    if (!last || !cfg || !data) return;
    let test: Record<string, number> = { ...last.metrics };
    if (L.ctx.task !== "clustering" && last.predictor) {
      const { pred, proba } = await predictAll(last.predictor, L.p.Xte);
      if (L.ctx.task === "classification") {
        const r = classReport(L.p.yte, pred, L.p.classes.length);
        test = { accuracy: r.accuracy, macro_f1: r.macroF1, baseline: r.baseline };
      } else {
        const r = regReport(L.p.yte, pred);
        test = { r2: r.r2, rmse: r.rmse, mae: r.mae, baseline_rmse: r.baselineRmse };
      }
      setEval({ task: L.ctx.task, engine: L.engine.label, y: L.p.yte, pred, proba, classes: L.p.classes, predictor: last.predictor });
    }
    const every = Math.max(1, Math.ceil(all.length / 200));
    /* eslint-disable react-hooks/purity -- runs after training completes, never during render */
    const rec: RunRecord = {
      createdAt: Date.now(), dataset: data.name, rows: data.rows.length, engine: L.engine.id, engineLabel: L.engine.label, task: L.ctx.task,
      params: L.ctx.params, seed: cfg.seed, testSize: cfg.testSize, features: L.p.names.length, test,
      history: all.filter((_, i) => i % every === 0 || i === all.length - 1).map((e) => ({ step: e.step, ...e.metrics })),
      ms: Math.round(performance.now() - L.t0),
    };
    /* eslint-enable react-hooks/purity */
    try { setSaved(await getDb().runs.add(rec)); } catch { /* storage blocked (private mode): run still shown */ }
  };

  const events = runner.events;
  const last = events[events.length - 1];
  const history = useMemo(() => events.map((e) => ({ step: e.step, ...e.metrics })), [events]);

  // ----- throttled decision boundary / regression fit -----
  const vizKey = `${fx}:${fy}:${engine.id}`;
  const [viz, setViz] = useState<{ key: string; bg?: Bg; line?: [number, number][] }>({ key: "" });
  const bg = viz.key === vizKey ? viz.bg ?? null : null;
  const fitLine = viz.key === vizKey ? viz.line : undefined;
  const timer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const latest = useRef(last);
  useEffect(() => { latest.current = last; });
  // leaving with a run in flight (new dataset): free its worker / TF.js memory
  useEffect(() => () => latest.current?.predictor?.dispose?.(), []);
  const base = useMemo(() => (L?.ctx.task !== "clustering" && L ? L.ctx.Xtr[0].map((_, j) => mean(L.ctx.Xtr.map((r) => r[j]))) : []), [L]);
  const [redraw, setRedraw] = useState(0);
  useEffect(() => {
    if (!L || L.ctx.task === "clustering" || !last?.predictor || timer.current) return;
    const key = vizKey;
    timer.current = setTimeout(async () => {
      const seen = latest.current;
      const pr = seen?.predictor;
      if (!pr) { timer.current = null; return; }
      const xs = L.p.Xall.map((r) => r[fx]), ys = L.p.Xall.map((r) => r[fy]);
      const pad = (a: number[]) => { const lo = Math.min(...a), hi = Math.max(...a), d = (hi - lo) * 0.08 || 1; return [lo - d, hi + d]; };
      const [x0, x1] = pad(xs), [y0, y1] = pad(ys);
      if (L.ctx.task === "classification") {
        const rows: number[][] = [];
        for (let gy = 0; gy < RES; gy++) for (let gx = 0; gx < RES; gx++) {
          const r = base.slice();
          r[fx] = x0 + ((gx + 0.5) / RES) * (x1 - x0);
          if (fy !== fx) r[fy] = y0 + ((gy + 0.5) / RES) * (y1 - y0);
          rows.push(r);
        }
        const { pred } = await predictAll(pr, rows).catch(() => ({ pred: [] as number[] }));
        timer.current = null;
        if (pred.length) setViz({ key, bg: { res: RES, x0, x1, y0, y1, colors: pred.map((c) => `${colorOf(c)}3a`) } });
      } else {
        const line: [number, number][] = [];
        const rows = Array.from({ length: 60 }, (_, i) => { const r = base.slice(); r[fx] = x0 + (i / 59) * (x1 - x0); return r; });
        const { pred } = await predictAll(pr, rows).catch(() => ({ pred: [] as number[] }));
        timer.current = null;
        pred.forEach((v, i) => line.push([rows[i][fx], v]));
        if (pred.length) setViz({ key, line });
      }
      if (latest.current !== seen) setRedraw((n) => n + 1); // newer step arrived mid-draw: draw it too
    }, 160);
  }, [last, L, fx, fy, base, redraw, vizKey]);
  useEffect(() => () => { if (timer.current) clearTimeout(timer.current); }, []);

  // ----- clustering projection (computed once per run) -----
  const proj = useMemo(() => (L && L.ctx.task === "clustering" ? projection(L.X) : null), [L]);

  if (!data || !cfg) return null;
  if (error || !p) return <Card title="Train"><p role="alert" className="text-sm text-bad">{error} Fix it in the Features step.</p></Card>;

  const sameRun = L?.engine.id === engine.id;
  const points: Pt[] = (() => {
    if (!L || !sameRun) return [];
    if (L.ctx.task === "clustering") {
      const lab = last?.labels;
      const src = engine.id === "pca" && last?.proj ? last.proj : axisMode === "pca" && proj ? proj.pts : L.X.map((r) => [r[fx], r[fy]]);
      return src.slice(0, 2500).map((r, i) => ({ x: r[0], y: r[1] ?? 0, c: engine.id === "pca" ? (L.p.classes.length ? colorOf(L.p.yall[i]) : "#60a5fa") : engine.id === "anomaly" ? (lab?.[i] ? "#f87171" : "#475569") : colorOf(lab?.[i] ?? -1), r: lab?.[i] === 1 && engine.id === "anomaly" ? 4.5 : 3 }));
    }
    const tr = L.p.Xtr.slice(0, 1500).map((r, i) => ({ x: r[fx], y: L.ctx.task === "regression" ? L.p.ytr[i] : r[fy], c: L.ctx.task === "regression" ? "#60a5fa" : colorOf(L.p.ytr[i]) }));
    const te = L.p.Xte.slice(0, 500).map((r, i) => ({ x: r[fx], y: L.ctx.task === "regression" ? L.p.yte[i] : r[fy], c: L.ctx.task === "regression" ? "#f59e0b" : colorOf(L.p.yte[i]), ring: true }));
    return [...tr, ...te];
  })();
  const marks = L?.ctx.task === "clustering" && axisMode === "features" && last?.centroids
    ? last.centroids.map((c, i) => ({ x: c[fx], y: c[fy], c: colorOf(i) })) : undefined;

  const m = last?.metrics ?? {};
  const gap = supervised && runner.status === "done" ? (task === "classification" ? (m.acc ?? 0) - (m.val_acc ?? 0) : (m.r2 ?? 0) - (m.val_r2 ?? 0)) : 0;

  return (
    <div className="space-y-5">
      <Card title="Pick a model" icon={<Cpu className="size-4" />}>
        <div className="grid gap-2 sm:grid-cols-2 xl:grid-cols-4">
          {engines.map((e) => (
            <button key={e.id} disabled={busy} onClick={() => setEngineId(e.id)} aria-pressed={e.id === engine.id}
              className={cx("cursor-pointer rounded-xl border p-3 text-left transition-[border,background] disabled:cursor-not-allowed disabled:opacity-50",
                e.id === engine.id ? "border-accent bg-accent/10" : "border-border bg-surface/60 hover:border-primary/50")}>
              <div className="flex items-center gap-1.5 text-sm font-medium">{e.label.replace(" · Python", "")}{e.remote && <Badge tone="green">Python</Badge>}</div>
              <div className="mt-0.5 text-[11px] leading-snug text-dim">{e.blurb}</div>
            </button>
          ))}
        </div>
        {pyModels && !pyModels.length && supervised && (
          <p className="mt-3 text-xs text-dim">
            Want SVM, Gradient Boosting or bigger forests? Start the optional Python backend (<code className="num text-sub">backend/</code>, see README) and it will appear here automatically at <span className="num">{BACKEND}</span>.
          </p>
        )}
      </Card>

      <div className="grid grid-cols-1 gap-5 xl:grid-cols-[320px_1fr]">
        <Card title="Hyperparameters">
          <div className="space-y-4">
            {engine.params.length ? engine.params.map((q) => (
              <Slider key={q.key} label={q.label} min={q.min} max={q.max} step={q.step} value={param(q.key, q.def)} disabled={busy}
                onChange={(v) => setParamState((s) => ({ ...s, [engine.id]: { ...s[engine.id], [q.key]: v } }))} />
            )) : <p className="text-sm text-dim">No knobs — this model has a closed-form fit.</p>}
            <Select label="Playback speed" value={speed} options={SPEEDS} onChange={(v) => { setSpeed(v); runner.setDelay(Number(v)); }} />
            <div className="flex flex-wrap gap-2 pt-1">
              {!busy && <Button variant="accent" onClick={start}><Play className="size-4" aria-hidden />{runner.status === "done" ? "Train again" : "Train"}</Button>}
              {runner.status === "running" && <Button onClick={runner.pause}><Pause className="size-4" aria-hidden />Pause</Button>}
              {runner.status === "paused" && <Button variant="accent" onClick={runner.resume}><Play className="size-4" aria-hidden />Resume</Button>}
              {busy && <Button onClick={() => void runner.step()} aria-label="Advance one step"><StepForward className="size-4" aria-hidden />Step</Button>}
              {busy && <Button variant="danger" onClick={runner.stop}><Square className="size-4" aria-hidden />Stop</Button>}
            </div>
            {runner.error && <p role="alert" className="text-sm text-bad">{runner.error}</p>}
          </div>
        </Card>

        <Card title="Live training" icon={<Activity className="size-4" />}
          actions={<>
            <Badge tone={runner.status === "running" ? "green" : runner.status === "done" ? "blue" : "default"}>{runner.status}</Badge>
            {last && <span className="num text-xs text-dim">step {last.step}/{last.total}</span>}
          </>}>
          <div className="mb-4 h-1.5 overflow-hidden rounded-full bg-muted" role="progressbar" aria-valuemin={0} aria-valuemax={100} aria-valuenow={last ? Math.round((last.step / last.total) * 100) : 0} aria-label="Training progress">
            <div className="h-full bg-gradient-to-r from-primary to-accent transition-[width] duration-150" style={{ width: `${last ? (last.step / last.total) * 100 : 0}%` }} />
          </div>
          {!last ? (
            <div className="grid-bg grid h-64 place-items-center rounded-xl border border-dashed border-border text-sm text-dim">Press Train to watch {engine.label} learn.</div>
          ) : (
            <div className="space-y-4">
              <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
                {Object.entries(m).slice(0, 4).map(([k, v]) => (
                  <Stat key={k} label={k.replace("_", " ")} value={/acc|pct|pc\d|cumul/.test(k) ? pct(v) : fmt(v)} tone={k.startsWith("val") ? "accent" : undefined} />
                ))}
              </div>
              {sameRun && supervised && history.length > 1 && (
                <div className="grid gap-4 lg:grid-cols-2">
                  <Curves data={history} xLabel={engine.id === "tree" ? "depth" : engine.id === "forest" ? "trees" : "epoch"}
                    keys={[{ key: "loss", label: task === "regression" ? "train RMSE" : "train loss", color: "#60a5fa" }, { key: "val_loss", label: task === "regression" ? "val RMSE" : "val loss", color: "#f59e0b", dashed: true }]} />
                  <Curves data={history} keys={task === "regression"
                    ? [{ key: "r2", label: "train R²", color: "#60a5fa" }, { key: "val_r2", label: "val R²", color: "#f59e0b", dashed: true }]
                    : [{ key: "acc", label: "train acc", color: "#60a5fa" }, { key: "val_acc", label: "val acc", color: "#f59e0b", dashed: true }]} />
                </div>
              )}
              {sameRun && engine.id === "kmeans" && history.length > 1 && <Curves data={history} keys={[{ key: "inertia", label: "inertia (within-cluster spread)", color: "#f59e0b" }]} height={180} />}
              {gap > 0.1 && (
                <p className="flex items-start gap-2 rounded-lg border border-accent/40 bg-accent/10 px-3 py-2 text-sm text-accent">
                  <TriangleAlert className="mt-0.5 size-4 shrink-0" aria-hidden />
                  Overfitting: training score beats validation by {pct(gap)}. Try more regularisation, a shallower model, or fewer epochs.
                </p>
              )}
            </div>
          )}
        </Card>
      </div>

      {last && sameRun && (
        <div className="grid grid-cols-1 gap-5 xl:grid-cols-[1.4fr_1fr]">
          <Card title={supervised ? (task === "classification" ? "Decision boundary" : "Model fit") : engine.id === "pca" ? "PC1 vs PC2" : "Clusters"}
            actions={
              <div className="flex flex-wrap items-end gap-2">
                {!supervised && engine.id !== "pca" && (
                  <Select label="Axes" value={axisMode} onChange={setAxisMode} options={[{ value: "pca", label: "PCA projection" }, { value: "features", label: "Two features" }]} className="h-8 text-xs" />
                )}
                {(supervised || axisMode === "features") && engine.id !== "pca" && <>
                  <Select label="X axis" value={String(fx)} onChange={(v) => setAxes([Number(v), fy])} options={names.map((n, i) => ({ value: String(i), label: n }))} className="h-8 max-w-40 text-xs" />
                  {task !== "regression" && <Select label="Y axis" value={String(fy)} onChange={(v) => setAxes([fx, Number(v)])} options={names.map((n, i) => ({ value: String(i), label: n }))} className="h-8 max-w-40 text-xs" />}
                </>}
              </div>
            }>
            <Scatter2D points={points} bg={task === "classification" ? bg : null} line={task === "regression" ? fitLine : undefined} marks={marks}
              xLabel={engine.id === "pca" || (!supervised && axisMode === "pca") ? "PC1" : names[fx]}
              yLabel={task === "regression" ? cfg.target ?? "target" : engine.id === "pca" || (!supervised && axisMode === "pca") ? "PC2" : names[fy]}
              label={`${engine.label} visualisation`} height={380} />
            <p className="mt-2 text-xs text-dim">
              {task === "classification" && "Shaded regions = what the model predicts if every other feature sits at its average. Ringed dots are test rows."}
              {task === "regression" && "Amber line = prediction along this feature (others held at their average). Amber dots are test rows."}
              {!supervised && engine.id === "anomaly" && "Red points are flagged as anomalies."}
              {!supervised && (engine.id === "kmeans" || engine.id === "dbscan") && "Colours are discovered clusters; grey = DBSCAN noise. ✕ marks centroids (feature axes)."}
            </p>
          </Card>

          <div className="space-y-5">
            {engine.id === "pca" && last.explained && (
              <Card title="Explained variance (scree)">
                <Bars data={last.explained.slice(0, 12).map((v, i) => ({ x: `PC${i + 1}`, count: +(v * 100).toFixed(1) }))} xKey="x" yKey="count" color="#f59e0b" label="Explained variance per principal component, percent" />
              </Card>
            )}
            {!supervised && runner.status === "done" && (proj || last.proj) && (
              <Card title="3D view">
                <PointCloud3D height={300}
                  pts={(engine.id === "pca" ? last.proj! : proj!.pts).slice(0, 3000)}
                  colors={(engine.id === "pca" ? last.proj! : proj!.pts).slice(0, 3000).map((_, i) => engine.id === "pca" ? (L!.p.classes.length ? colorOf(L!.p.yall[i]) : "#60a5fa") : engine.id === "anomaly" ? (last.labels?.[i] ? "#f87171" : "#475569") : colorOf(last.labels?.[i] ?? -1))}
                  label="3D projection coloured by result" />
              </Card>
            )}
            {runner.status === "done" && (
              <Card title="Run complete">
                <p className="text-sm text-sub">{saved ? `Saved as experiment #${saved}.` : "Run finished."} {supervised ? "Now see how it does on the test set it never saw." : "Compare it with other runs in Experiments."}</p>
                <div className="mt-3 flex gap-2">
                  {supervised ? <Button variant="accent" onClick={() => setStep(5)}>Evaluate on test set<ArrowRight className="size-4" aria-hidden /></Button>
                    : <a href="/compare" className="inline-flex h-10 items-center gap-2 rounded-lg bg-accent px-4 text-sm font-medium text-black">Open experiments<ArrowRight className="size-4" aria-hidden /></a>}
                </div>
              </Card>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
