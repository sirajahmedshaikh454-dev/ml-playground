"use client";
import { useEffect, useRef, useState } from "react";
import {
  Area, AreaChart, Bar, BarChart, CartesianGrid, Legend, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis,
} from "recharts";
import { PALETTE } from "@/lib/core";
import { fmt } from "./ui";

const axis = { stroke: "#64748b", fontSize: 11, tickLine: false } as const;
const tip = { contentStyle: { background: "#0f172a", border: "1px solid #263247", borderRadius: 10, fontSize: 12 }, labelStyle: { color: "#cbd5e1" } };

export function Curves({ data, keys, xKey = "step", height = 240, xLabel }: { data: Record<string, number>[]; keys: { key: string; label: string; color: string; dashed?: boolean }[]; xKey?: string; height?: number; xLabel?: string }) {
  const shown = keys.filter((k) => data.some((d) => Number.isFinite(d[k.key])));
  return (
    <div style={{ height }} role="img" aria-label={`Line chart of ${shown.map((k) => k.label).join(", ")} over ${xLabel ?? xKey}`}>
      <ResponsiveContainer>
        <LineChart data={data} margin={{ top: 8, right: 12, bottom: 4, left: -8 }}>
          <CartesianGrid stroke="#1e293b" vertical={false} />
          <XAxis dataKey={xKey} {...axis} />
          <YAxis {...axis} width={48} tickFormatter={(v: number) => fmt(v, 2)} domain={["auto", "auto"]} />
          <Tooltip {...tip} formatter={(v) => fmt(Number(v), 4)} />
          <Legend wrapperStyle={{ fontSize: 12 }} />
          {shown.map((k) => (
            <Line key={k.key} type="monotone" dataKey={k.key} name={k.label} stroke={k.color} strokeWidth={2} dot={false}
              strokeDasharray={k.dashed ? "5 4" : undefined} isAnimationActive={false} />
          ))}
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}

export function Bars({ data, xKey, yKey, color = "#3b82f6", height = 160, label }: { data: Record<string, string | number>[]; xKey: string; yKey: string; color?: string; height?: number; label?: string }) {
  return (
    <div style={{ height }} role="img" aria-label={label ?? `Bar chart of ${yKey}`}>
      <ResponsiveContainer>
        <BarChart data={data} margin={{ top: 4, right: 4, bottom: 0, left: -20 }}>
          <XAxis dataKey={xKey} {...axis} interval="preserveStartEnd" tickFormatter={(v) => (typeof v === "number" ? fmt(v, 1) : String(v).slice(0, 10))} />
          <YAxis {...axis} width={40} />
          <Tooltip {...tip} cursor={{ fill: "#1e293b" }} />
          <Bar dataKey={yKey} fill={color} radius={[3, 3, 0, 0]} isAnimationActive={false} />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}

export function RocChart({ pts, auc }: { pts: { fpr: number; tpr: number }[]; auc: number }) {
  return (
    <div style={{ height: 240 }} role="img" aria-label={`ROC curve, area under curve ${fmt(auc)}`}>
      <ResponsiveContainer>
        <AreaChart data={pts} margin={{ top: 8, right: 12, bottom: 4, left: -8 }}>
          <CartesianGrid stroke="#1e293b" />
          <XAxis dataKey="fpr" type="number" domain={[0, 1]} {...axis} tickFormatter={(v: number) => v.toFixed(1)} />
          <YAxis dataKey="tpr" domain={[0, 1]} {...axis} width={40} tickFormatter={(v: number) => v.toFixed(1)} />
          <Tooltip {...tip} formatter={(v) => fmt(Number(v))} labelFormatter={(v) => `FPR ${fmt(Number(v))}`} />
          <Area type="stepAfter" dataKey="tpr" stroke="#f59e0b" fill="#f59e0b" fillOpacity={0.15} strokeWidth={2} isAnimationActive={false} />
        </AreaChart>
      </ResponsiveContainer>
    </div>
  );
}

export function Heat({ rows, cols, m, diverging, fmtCell = (v: number) => fmt(v, 2), caption }: { rows: string[]; cols: string[]; m: number[][]; diverging?: boolean; fmtCell?: (v: number) => string; caption: string }) {
  const max = Math.max(1e-9, ...m.flat().map(Math.abs));
  const color = (v: number) => {
    const t = Math.abs(v) / max;
    return diverging ? (v >= 0 ? `rgb(59 130 246 / ${0.08 + t * 0.8})` : `rgb(248 113 113 / ${0.08 + t * 0.8})`) : `rgb(245 158 11 / ${0.06 + t * 0.85})`;
  };
  return (
    <div className="scroll-thin overflow-x-auto">
      <table className="num border-separate border-spacing-0.5 text-[11px]">
        <caption className="sr-only">{caption}</caption>
        <thead>
          <tr>
            <th />
            {cols.map((c) => <th key={c} scope="col" className="max-w-16 truncate px-1 pb-1 font-normal text-dim" title={c}>{c.slice(0, 8)}</th>)}
          </tr>
        </thead>
        <tbody>
          {m.map((r, i) => (
            <tr key={rows[i]}>
              <th scope="row" className="max-w-28 truncate pr-2 text-right font-normal text-dim" title={rows[i]}>{rows[i].slice(0, 14)}</th>
              {r.map((v, j) => (
                <td key={j} className="h-9 min-w-11 rounded text-center text-fg" style={{ background: color(v) }} title={`${rows[i]} / ${cols[j]}: ${fmtCell(v)}`}>
                  {fmtCell(v)}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export interface Pt { x: number; y: number; c: string; r?: number; ring?: boolean }
export interface Bg { res: number; x0: number; x1: number; y0: number; y1: number; colors: string[] }

// Canvas scatter: decision regions, regression lines, residuals and cluster plots all use this.
export function Scatter2D({ points, bg, line, hline, marks, xLabel, yLabel, height = 340, label }: {
  points: Pt[]; bg?: Bg | null; line?: [number, number][]; hline?: number; marks?: Pt[]; xLabel?: string; yLabel?: string; height?: number; label: string;
}) {
  const wrap = useRef<HTMLDivElement>(null);
  const cv = useRef<HTMLCanvasElement>(null);
  const [w, setW] = useState(600);
  useEffect(() => {
    const ro = new ResizeObserver(([e]) => setW(Math.max(200, e.contentRect.width)));
    if (wrap.current) ro.observe(wrap.current);
    return () => ro.disconnect();
  }, []);
  useEffect(() => {
    const c = cv.current;
    if (!c) return;
    const dpr = window.devicePixelRatio || 1;
    c.width = w * dpr; c.height = height * dpr;
    const g = c.getContext("2d")!;
    g.setTransform(dpr, 0, 0, dpr, 0, 0);
    g.clearRect(0, 0, w, height);
    const P = { l: 44, r: 10, t: 10, b: 30 };
    const xs = points.map((p) => p.x).concat(bg ? [bg.x0, bg.x1] : []).concat(line?.map((p) => p[0]) ?? []);
    const ys = points.map((p) => p.y).concat(bg ? [bg.y0, bg.y1] : []).concat(line?.map((p) => p[1]) ?? []).concat(hline != null ? [hline] : []);
    if (!xs.length) return;
    let x0 = Math.min(...xs), x1 = Math.max(...xs), y0 = Math.min(...ys), y1 = Math.max(...ys);
    if (!bg) { const px = (x1 - x0) * 0.05 || 1, py = (y1 - y0) * 0.05 || 1; x0 -= px; x1 += px; y0 -= py; y1 += py; }
    const X = (v: number) => P.l + ((v - x0) / (x1 - x0 || 1)) * (w - P.l - P.r);
    const Y = (v: number) => height - P.b - ((v - y0) / (y1 - y0 || 1)) * (height - P.t - P.b);
    if (bg) {
      const cw = (w - P.l - P.r) / bg.res, ch = (height - P.t - P.b) / bg.res;
      bg.colors.forEach((col, i) => {
        const gx = i % bg.res, gy = Math.floor(i / bg.res);
        g.fillStyle = col;
        // snap to whole pixels so neighbouring cells meet without seams
        const x = Math.floor(P.l + gx * cw), y = Math.floor(height - P.b - (gy + 1) * ch);
        g.fillRect(x, y, Math.ceil(P.l + (gx + 1) * cw) - x, Math.ceil(height - P.b - gy * ch) - y);
      });
    }
    // axes + ticks
    g.strokeStyle = "#334155"; g.lineWidth = 1; g.fillStyle = "#64748b"; g.font = "10px var(--font-mono), monospace";
    g.beginPath(); g.moveTo(P.l, P.t); g.lineTo(P.l, height - P.b); g.lineTo(w - P.r, height - P.b); g.stroke();
    for (let i = 0; i <= 4; i++) {
      const vx = x0 + ((x1 - x0) * i) / 4, vy = y0 + ((y1 - y0) * i) / 4;
      g.textAlign = "center"; g.fillText(fmt(vx, 1), X(vx), height - P.b + 13);
      g.textAlign = "right"; g.fillText(fmt(vy, 1), P.l - 4, Y(vy) + 3);
    }
    if (xLabel) { g.textAlign = "center"; g.fillStyle = "#94a3b8"; g.fillText(xLabel, (P.l + w - P.r) / 2, height - 3); }
    if (yLabel) { g.save(); g.translate(10, (P.t + height - P.b) / 2); g.rotate(-Math.PI / 2); g.textAlign = "center"; g.fillStyle = "#94a3b8"; g.fillText(yLabel, 0, 0); g.restore(); }
    if (hline != null) { g.setLineDash([4, 4]); g.strokeStyle = "#94a3b8"; g.beginPath(); g.moveTo(P.l, Y(hline)); g.lineTo(w - P.r, Y(hline)); g.stroke(); g.setLineDash([]); }
    for (const p of points) {
      g.beginPath(); g.arc(X(p.x), Y(p.y), p.r ?? 3.2, 0, Math.PI * 2);
      g.fillStyle = p.c; g.globalAlpha = 0.9; g.fill(); g.globalAlpha = 1;
      g.lineWidth = p.ring ? 2 : 0.6; g.strokeStyle = p.ring ? "#f8fafc" : "rgb(11 17 32 / 0.8)"; g.stroke();
    }
    if (line?.length) {
      g.strokeStyle = "#f59e0b"; g.lineWidth = 2.5; g.beginPath();
      line.forEach(([a, b], i) => (i ? g.lineTo(X(a), Y(b)) : g.moveTo(X(a), Y(b)))); g.stroke();
    }
    for (const m of marks ?? []) {
      const cx = X(m.x), cy = Y(m.y);
      g.lineWidth = 3; g.strokeStyle = "#0b1120"; g.beginPath(); g.moveTo(cx - 8, cy - 8); g.lineTo(cx + 8, cy + 8); g.moveTo(cx + 8, cy - 8); g.lineTo(cx - 8, cy + 8); g.stroke();
      g.lineWidth = 2; g.strokeStyle = m.c; g.stroke();
    }
  }, [points, bg, line, hline, marks, w, height, xLabel, yLabel]);
  return (
    <div ref={wrap} className="w-full">
      <canvas ref={cv} style={{ width: w, height }} role="img" aria-label={label} />
    </div>
  );
}

export const colorOf = (i: number) => (i < 0 ? "#475569" : PALETTE[i % PALETTE.length]);
