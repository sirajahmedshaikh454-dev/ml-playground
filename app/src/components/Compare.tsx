"use client";
import { Download, FlaskConical, Trash2, TriangleAlert, Upload } from "lucide-react";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { getDb, type RunRecord } from "@/lib/store";
import type { Task } from "@/lib/core";
import { Bars, Curves, colorOf } from "./charts";
import { Badge, Button, Card, Select, cx, fmt } from "./ui";

const PRIMARY: Record<Task, { key: string; label: string; higher: boolean }> = {
  classification: { key: "accuracy", label: "Test accuracy", higher: true },
  regression: { key: "r2", label: "Test R²", higher: true },
  clustering: { key: "silhouette", label: "Silhouette", higher: true },
};

export function Compare() {
  const [runs, setRuns] = useState<RunRecord[] | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [dataset, setDataset] = useState("");
  const [sel, setSel] = useState<number[]>([]);
  const fileIn = useRef<HTMLInputElement>(null);

  const load = useCallback(async () => {
    try { setRuns(await getDb().runs.orderBy("createdAt").reverse().toArray()); }
    catch { setErr("Browser storage is unavailable (private window?). Runs from this session cannot be listed."); setRuns([]); }
  }, []);
  useEffect(() => {
    let alive = true;
    getDb().runs.orderBy("createdAt").reverse().toArray()
      .then((r) => { if (alive) setRuns(r); })
      .catch(() => { if (alive) { setErr("Browser storage is unavailable (private window?). Runs from this session cannot be listed."); setRuns([]); } });
    return () => { alive = false; };
  }, []);

  const datasets = useMemo(() => [...new Set(runs?.map((r) => `${r.dataset}::${r.task}`))], [runs]);
  const key = dataset || datasets[0] || "";
  const group = useMemo(() => runs?.filter((r) => `${r.dataset}::${r.task}` === key) ?? [], [runs, key]);
  const task = group[0]?.task;
  const prim = task ? PRIMARY[task] : null;
  const ranked = prim ? [...group].sort((a, b) => (b.test[prim.key] ?? -Infinity) - (a.test[prim.key] ?? -Infinity)) : group;
  const metricKeys = [...new Set(group.flatMap((r) => Object.keys(r.test)))];
  const splits = new Set(group.filter((r) => r.task !== "clustering").map((r) => `${r.seed}/${r.testSize}`));
  const chosen = group.filter((r) => sel.includes(r.id!));
  const curveKey = task === "classification" ? "val_acc" : task === "regression" ? "val_r2" : "inertia";
  const overlay = useMemo(() => {
    const len = Math.max(0, ...chosen.map((r) => r.history.length));
    return Array.from({ length: len }, (_, i) => Object.fromEntries([["step", i + 1], ...chosen.map((r) => [`r${r.id}`, r.history[i]?.[curveKey]])]));
  }, [chosen, curveKey]);

  const exportAll = () => {
    const blob = new Blob([JSON.stringify({ schemaVersion: 1, exportedAt: new Date().toISOString(), runs }, null, 2)], { type: "application/json" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = `ml-playground-experiments-${new Date().toISOString().slice(0, 10)}.json`;
    a.click();
    URL.revokeObjectURL(a.href);
  };
  const importRuns = async (f: File) => {
    try {
      const j = JSON.parse(await f.text()) as { schemaVersion?: number; runs?: RunRecord[] };
      if (j.schemaVersion !== 1 || !Array.isArray(j.runs)) throw new Error("Not an ML Playground export");
      // validate shape before writing to the DB (trust boundary)
      const ok = j.runs.filter((r) => typeof r.dataset === "string" && typeof r.engine === "string" && r.test && typeof r.test === "object" && Array.isArray(r.history));
      await getDb().runs.bulkAdd(ok.map((r) => { const copy = { ...r }; delete copy.id; return copy; })); // fresh ids
      await load();
    } catch (e) { setErr((e as Error).message); }
  };

  if (!runs) return <div className="mx-auto max-w-7xl px-4 py-6"><div className="h-64 animate-pulse rounded-2xl bg-muted/40" /></div>;

  return (
    <div className="mx-auto max-w-7xl space-y-5 px-4 py-6">
      <div className="flex flex-wrap items-center gap-3">
        <h1 className="text-2xl font-semibold tracking-tight">Experiments</h1>
        <span className="text-sm text-dim">{runs.length} saved runs, stored only in this browser</span>
        <div className="ml-auto flex flex-wrap gap-2">
          <Button size="sm" onClick={() => fileIn.current?.click()}><Upload className="size-3.5" aria-hidden />Import</Button>
          <Button size="sm" onClick={exportAll} disabled={!runs.length}><Download className="size-3.5" aria-hidden />Export JSON</Button>
          <Button size="sm" variant="danger" disabled={!runs.length} onClick={async () => { if (confirm(`Delete all ${runs.length} runs? This cannot be undone.`)) { await getDb().runs.clear(); await load(); } }}>
            <Trash2 className="size-3.5" aria-hidden />Clear all
          </Button>
          <input ref={fileIn} type="file" accept="application/json,.json" hidden onChange={(e) => { const f = e.target.files?.[0]; if (f) void importRuns(f); e.target.value = ""; }} />
        </div>
      </div>
      {err && <p role="alert" className="rounded-lg border border-bad/40 bg-bad/10 px-3 py-2 text-sm text-bad">{err}</p>}

      {!runs.length ? (
        <Card><div className="grid place-items-center gap-3 py-16 text-center">
          <FlaskConical className="size-10 text-primary" aria-hidden />
          <p className="text-lg font-medium">No experiments yet</p>
          <p className="max-w-md text-sm text-dim">Every model you train in the Data Lab is saved here automatically, so you can line them up and see which one really wins.</p>
          <a href="/lab" className="mt-2 inline-flex h-10 items-center rounded-lg bg-accent px-4 text-sm font-medium text-black">Open the Data Lab</a>
        </div></Card>
      ) : (
        <>
          <Select label="Dataset & task" value={key} onChange={(v) => { setDataset(v); setSel([]); }} className="max-w-md"
            options={datasets.map((d) => { const [n, t] = d.split("::"); return { value: d, label: `${n} — ${t}` }; })} />
          {splits.size > 1 && (
            <p className="flex items-center gap-2 rounded-lg border border-accent/40 bg-accent/10 px-3 py-2 text-sm text-accent">
              <TriangleAlert className="size-4 shrink-0" aria-hidden />These runs used different seeds or test sizes, so their test sets differ. Compare with care.
            </p>
          )}
          <div className="grid grid-cols-1 gap-5 xl:grid-cols-[1.2fr_1fr]">
            <Card title={`Leaderboard · ${prim?.label}`}>
              <Bars data={ranked.slice(0, 15).map((r) => ({ x: `#${r.id} ${r.engineLabel.split(" ")[0]}`, count: +(r.test[prim!.key] ?? 0).toFixed(4) }))} xKey="x" yKey="count" color="#f59e0b" height={220} label={`${prim?.label} per run`} />
            </Card>
            <Card title="Learning curves" actions={<span className="text-xs text-dim">tick runs below to overlay</span>}>
              {chosen.length ? (
                <Curves data={overlay} keys={chosen.map((r, i) => ({ key: `r${r.id}`, label: `#${r.id} ${r.engineLabel}`, color: colorOf(i) }))} height={220} xLabel="checkpoint" />
              ) : <p className="grid h-[220px] place-items-center text-sm text-dim">Select runs in the table to compare their {curveKey} curves.</p>}
            </Card>
          </div>
          <Card title="All runs">
            <div className="scroll-thin overflow-x-auto">
              <table className="num w-full min-w-[820px] text-left text-sm">
                <caption className="sr-only">Saved runs for the selected dataset</caption>
                <thead className="text-xs text-dim"><tr className="border-b border-border">
                  <th className="px-2 py-2" scope="col"><span className="sr-only">Overlay</span></th>
                  {["#", "Model", ...metricKeys, "Params", "Time", ""].map((h, i) => <th key={i} scope="col" className="px-2 py-2 font-medium">{h}</th>)}
                </tr></thead>
                <tbody>
                  {ranked.map((r, rank) => (
                    <tr key={r.id} className={cx("border-b border-border/50 hover:bg-muted/30", rank === 0 && "bg-accent/5")}>
                      <td className="px-2 py-2"><input type="checkbox" aria-label={`Overlay run ${r.id}`} checked={sel.includes(r.id!)} onChange={(e) => setSel(e.target.checked ? [...sel, r.id!] : sel.filter((s) => s !== r.id))} className="size-4 accent-amber-500" /></td>
                      <td className="px-2 py-2 text-dim">{r.id}</td>
                      <td className="px-2 py-2 font-sans">{r.engineLabel} {rank === 0 && <Badge tone="amber">best</Badge>}</td>
                      {metricKeys.map((k) => <td key={k} className="px-2 py-2">{fmt(r.test[k])}</td>)}
                      <td className="max-w-56 truncate px-2 py-2 text-xs text-dim" title={JSON.stringify(r.params)}>{Object.entries(r.params).map(([k, v]) => `${k}=${v}`).join(" ") || "—"}</td>
                      <td className="px-2 py-2 text-dim">{r.ms < 1000 ? `${r.ms}ms` : `${(r.ms / 1000).toFixed(1)}s`}</td>
                      <td className="px-2 py-2"><Button size="sm" variant="danger" aria-label={`Delete run ${r.id}`} onClick={async () => { await getDb().runs.delete(r.id!); await load(); }}><Trash2 className="size-3.5" aria-hidden /></Button></td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>
        </>
      )}
    </div>
  );
}
