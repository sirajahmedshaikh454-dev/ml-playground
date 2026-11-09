"use client";
import { Eraser, History, RotateCcw, Undo2, Wand2 } from "lucide-react";
import { useMemo } from "react";
import { type CleanOp, describeOp, profile } from "@/lib/clean";
import type { ColType } from "@/lib/core";
import { useLab } from "@/lib/store";
import { Badge, Button, Card, Stat, cx, fmt } from "../ui";
import { Preview } from "./DataStep";

export function CleanStep() {
  const { raw, data, ops, addOp, undo, resetOps } = useLab();
  const prof = useMemo(() => (data ? profile(data) : []), [data]);
  const dupes = useMemo(() => {
    if (!data) return 0;
    const s = new Set(data.rows.map((r) => JSON.stringify(data.columns.map((c) => r[c]))));
    return data.rows.length - s.size;
  }, [data]);
  const rawMissing = useMemo(() => (raw ? profile(raw).reduce((s, p) => s + p.missing, 0) : 0), [raw]);
  if (!raw || !data) return null;

  const missing = prof.reduce((s, p) => s + p.missing, 0);
  const outl = prof.reduce((s, p) => s + (p.outliers ?? 0), 0);

  // one-click "suggested" cleaning: dedupe, fill, clip
  const autoClean = () => {
    const suggested: CleanOp[] = [];
    if (dupes) suggested.push({ op: "dedupe" });
    prof.forEach((p) => {
      if (p.missing) suggested.push(p.type === "number" ? { op: "fill", col: p.name, how: "median" } : { op: "fill", col: p.name, how: "mode" });
      if (p.outliers) suggested.push({ op: "outliers", col: p.name, how: "clip", k: 1.5 });
    });
    suggested.forEach(addOp);
  };

  return (
    <div className="space-y-5">
      <div className="grid grid-cols-2 gap-3 lg:grid-cols-4">
        <Stat label="Rows" value={data.rows.length.toLocaleString()} sub={`was ${raw.rows.length.toLocaleString()}`} />
        <Stat label="Missing cells" value={missing.toLocaleString()} sub={`was ${rawMissing.toLocaleString()}`} tone={missing ? "bad" : "good"} />
        <Stat label="Duplicate rows" value={dupes} tone={dupes ? "bad" : "good"} />
        <Stat label="Outliers (IQR 1.5)" value={outl} tone={outl ? "accent" : "good"} />
      </div>

      <Card title="Column health" icon={<Eraser className="size-4" />}
        actions={<>
          <Button size="sm" variant="accent" onClick={autoClean} disabled={!missing && !dupes && !outl}><Wand2 className="size-3.5" aria-hidden />Auto-clean</Button>
          <Button size="sm" onClick={() => addOp({ op: "dedupe" })} disabled={!dupes}>Remove {dupes} duplicates</Button>
          <Button size="sm" onClick={() => addOp({ op: "dropMissing" })} disabled={!missing}>Drop rows with missing</Button>
        </>}>
        <div className="scroll-thin overflow-x-auto">
          <table className="w-full min-w-[760px] text-left text-sm">
            <caption className="sr-only">Per-column profile and cleaning actions</caption>
            <thead className="text-xs text-dim">
              <tr className="border-b border-border">
                {["Column", "Type", "Missing", "Unique", "Summary", "Outliers", "Fix"].map((h) => <th key={h} scope="col" className="px-2 py-2 font-medium">{h}</th>)}
              </tr>
            </thead>
            <tbody>
              {prof.map((p) => (
                <tr key={p.name} className="border-b border-border/50 hover:bg-muted/30">
                  <td className="px-2 py-2 font-medium">{p.name}</td>
                  <td className="px-2 py-2">
                    <select aria-label={`Type of ${p.name}`} value={p.type} onChange={(e) => addOp({ op: "cast", col: p.name, to: e.target.value as ColType })}
                      className="cursor-pointer rounded-md border border-border bg-surface px-1.5 py-1 text-xs">
                      {["number", "category", "text", "boolean"].map((t) => <option key={t}>{t}</option>)}
                    </select>
                  </td>
                  <td className={cx("num px-2 py-2", p.missing ? "text-bad" : "text-dim")}>{p.missing} <span className="text-dim">({((p.missing / data.rows.length) * 100).toFixed(0)}%)</span></td>
                  <td className="num px-2 py-2 text-dim">{p.unique}</td>
                  <td className="num px-2 py-2 text-xs text-sub">
                    {p.type === "number" ? `μ ${fmt(p.mean, 2)} · σ ${fmt(p.std, 2)} · [${fmt(p.min, 1)}, ${fmt(p.max, 1)}]` : p.top?.slice(0, 3).map(([v, n]) => `${v.slice(0, 14)} (${n})`).join(", ")}
                  </td>
                  <td className="num px-2 py-2">{p.outliers ? <Badge tone="amber">{p.outliers}</Badge> : <span className="text-dim">—</span>}</td>
                  <td className="px-2 py-2">
                    <div className="flex flex-wrap gap-1">
                      {p.missing > 0 && (
                        <select aria-label={`Fill missing in ${p.name}`} defaultValue="" onChange={(e) => { if (e.target.value) addOp({ op: "fill", col: p.name, how: e.target.value as "mean" }); e.target.value = ""; }}
                          className="cursor-pointer rounded-md border border-border bg-surface px-1.5 py-1 text-xs">
                          <option value="">Fill…</option>
                          {(p.type === "number" ? ["mean", "median", "zero"] : ["mode", "unknown"]).map((h) => <option key={h} value={h}>{h}</option>)}
                        </select>
                      )}
                      {!!p.outliers && <>
                        <Button size="sm" onClick={() => addOp({ op: "outliers", col: p.name, how: "clip", k: 1.5 })}>Clip</Button>
                        <Button size="sm" onClick={() => addOp({ op: "outliers", col: p.name, how: "remove", k: 1.5 })}>Remove</Button>
                      </>}
                      <Button size="sm" variant="danger" onClick={() => addOp({ op: "dropCol", col: p.name })} aria-label={`Drop column ${p.name}`}>Drop</Button>
                    </div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>

      <Card title="Cleaning recipe" icon={<History className="size-4" />}
        actions={<>
          <Button size="sm" onClick={undo} disabled={!ops.length}><Undo2 className="size-3.5" aria-hidden />Undo</Button>
          <Button size="sm" onClick={resetOps} disabled={!ops.length}><RotateCcw className="size-3.5" aria-hidden />Reset</Button>
        </>}>
        {ops.length ? (
          <ol className="space-y-1.5 text-sm">
            {ops.map((o, i) => (
              <li key={i} className="flex items-center gap-2"><Badge tone="blue">{i + 1}</Badge>{describeOp(o)}</li>
            ))}
          </ol>
        ) : <p className="text-sm text-dim">No steps yet. Each fix is recorded here so it can be undone and replayed.</p>}
      </Card>

      <Card title="After cleaning"><Preview ds={data} max={25} /></Card>
    </div>
  );
}
