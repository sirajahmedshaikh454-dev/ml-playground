"use client";
import dynamic from "next/dynamic";
import { BarChart3, Box, Grid3x3 } from "lucide-react";
import { useMemo } from "react";
import { correlation, histogram } from "@/lib/clean";
import { projection } from "@/lib/ml";
import { prepare } from "@/lib/features";
import { useLab } from "@/lib/store";
import { Bars, Heat, colorOf } from "../charts";
import { Card } from "../ui";

const PointCloud3D = dynamic(() => import("../Scene3D"), { ssr: false, loading: () => <div className="h-[340px] animate-pulse rounded-xl bg-muted/40" /> });

export function ExploreStep() {
  const { data, cfg } = useLab();
  const corr = useMemo(() => (data ? correlation(data) : null), [data]);
  const cols = useMemo(() => (data ? data.columns.filter((c) => data.types[c] !== "text").slice(0, 12) : []), [data]);
  const cloud = useMemo(() => {
    if (!data || !cfg) return null;
    try {
      const p = prepare(data, { ...cfg, task: cfg.task === "clustering" ? "clustering" : cfg.task, testSize: 0 });
      if (!p.Xall.length || !p.Xall[0].length) return null;
      const sample = p.Xall.slice(0, 3000);
      const { pts, explained } = projection(sample);
      return { pts, explained, colors: sample.map((_, i) => (p.classes.length ? colorOf(p.yall[i]) : "#60a5fa")), classes: p.classes };
    } catch { return null; }
  }, [data, cfg]);
  if (!data) return null;

  return (
    <div className="space-y-5">
      <Card title="Distributions" icon={<BarChart3 className="size-4" />}>
        <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-3">
          {cols.map((c) => {
            const isNum = data.types[c] === "number";
            const vals = data.rows.map((r) => r[c]);
            const d = isNum
              ? histogram(vals.filter((v): v is number => typeof v === "number"), 20)
              : (() => { const m = new Map<string, number>(); vals.forEach((v) => v != null && m.set(String(v), (m.get(String(v)) ?? 0) + 1)); return [...m.entries()].sort((a, b) => b[1] - a[1]).slice(0, 12).map(([x, count]) => ({ x, count })); })();
            return (
              <div key={c} className="rounded-xl border border-border bg-surface/50 p-3">
                <div className="mb-1 flex items-center justify-between text-sm"><span className="font-medium">{c}</span><span className="text-xs text-dim">{data.types[c]}</span></div>
                <Bars data={d} xKey="x" yKey="count" color={isNum ? "#3b82f6" : "#f59e0b"} label={`Distribution of ${c}`} />
              </div>
            );
          })}
        </div>
      </Card>

      <div className="grid gap-5 xl:grid-cols-2">
        <Card title="Correlation (Pearson)" icon={<Grid3x3 className="size-4" />}>
          {corr && corr.cols.length > 1
            ? <Heat rows={corr.cols} cols={corr.cols} m={corr.m} diverging caption="Correlation matrix of numeric columns" />
            : <p className="text-sm text-dim">Needs at least two numeric columns.</p>}
          <p className="mt-3 text-xs text-dim">Blue = move together, red = move opposite. Strongly correlated features carry overlapping information.</p>
        </Card>
        <Card title="3D shape of your data" icon={<Box className="size-4" />}>
          {cloud ? (
            <>
              <PointCloud3D pts={cloud.pts} colors={cloud.colors} label="3D projection of the dataset, coloured by target" />
              <div className="mt-2 flex flex-wrap items-center gap-3 text-xs text-dim">
                <span>{cloud.explained.length ? `PCA, ${(cloud.explained.slice(0, 3).reduce((a, b) => a + b, 0) * 100).toFixed(0)}% of variance shown` : "Raw features"} · drag to rotate</span>
                {cloud.classes.slice(0, 8).map((c, i) => <span key={c} className="flex items-center gap-1"><span className="size-2.5 rounded-full" style={{ background: colorOf(i) }} />{c}</span>)}
              </div>
            </>
          ) : <p className="text-sm text-dim">Choose features in the next step to see the 3D view.</p>}
        </Card>
      </div>
    </div>
  );
}
