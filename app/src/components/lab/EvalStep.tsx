"use client";
import { ClipboardCheck, Target } from "lucide-react";
import { useMemo, useState } from "react";
import { classReport, regReport, roc } from "@/lib/metrics";
import { useLab } from "@/lib/store";
import { Heat, RocChart, Scatter2D } from "../charts";
import { Badge, Card, Select, Stat, fmt, pct } from "../ui";

export function EvalStep() {
  const { evalResult: e } = useLab();
  const [pos, setPos] = useState(0);
  const cls = useMemo(() => (e?.task === "classification" ? classReport(e.y, e.pred, e.classes.length) : null), [e]);
  const reg = useMemo(() => (e?.task === "regression" ? regReport(e.y, e.pred) : null), [e]);
  const rocRes = useMemo(() => (e?.proba && cls ? roc(e.proba.map((p) => p[pos] ?? 0), e.y.map((y) => y === pos)) : null), [e, cls, pos]);

  if (!e) return <Card title="Evaluate"><p className="text-sm text-dim">Train a supervised model first. The test set stays locked until a run finishes.</p></Card>;

  if (cls) {
    const beat = cls.accuracy - cls.baseline;
    return (
      <div className="space-y-5">
        <div className="grid grid-cols-2 gap-3 lg:grid-cols-4">
          <Stat label="Test accuracy" value={pct(cls.accuracy)} tone={beat > 0.05 ? "good" : "bad"} sub={`${e.y.length} unseen rows`} />
          <Stat label="Macro F1" value={fmt(cls.macroF1)} sub="balances every class equally" />
          <Stat label="Majority baseline" value={pct(cls.baseline)} sub="always guess the biggest class" />
          <Stat label="Lift over baseline" value={`${beat >= 0 ? "+" : ""}${(beat * 100).toFixed(1)} pts`} tone={beat > 0.05 ? "good" : "bad"} />
        </div>
        <div className="grid gap-5 xl:grid-cols-2">
          <Card title="Confusion matrix" icon={<Target className="size-4" />} actions={<Badge tone="blue">{e.engine}</Badge>}>
            <Heat rows={e.classes.map((c) => `true ${c}`)} cols={e.classes.map((c) => `pred ${c}`)} m={cls.cm} fmtCell={(v) => String(v)} caption="Confusion matrix: rows are true classes, columns predicted classes" />
            <p className="mt-3 text-xs text-dim">The diagonal is correct. Anything off the diagonal shows which classes the model mixes up.</p>
          </Card>
          <Card title="ROC curve" actions={e.classes.length > 2 && <Select label="Positive class (one-vs-rest)" value={String(pos)} onChange={(v) => setPos(Number(v))} options={e.classes.map((c, i) => ({ value: String(i), label: c }))} className="h-8 text-xs" />}>
            {rocRes ? <>
              <RocChart pts={rocRes.pts} auc={rocRes.auc} />
              <p className="mt-2 text-sm">AUC <span className="num font-semibold text-accent">{fmt(rocRes.auc)}</span> <span className="text-xs text-dim">— 0.5 is a coin flip, 1.0 is perfect ranking</span></p>
            </> : <p className="text-sm text-dim">{e.engine} gives hard labels only, so there is no score to rank by. Try Logistic, kNN, Forest, Naive Bayes or the Neural Network.</p>}
          </Card>
        </div>
        <Card title="Per-class report" icon={<ClipboardCheck className="size-4" />}>
          <div className="scroll-thin overflow-x-auto">
            <table className="num w-full min-w-[480px] text-left text-sm">
              <caption className="sr-only">Precision, recall, F1 and support per class</caption>
              <thead className="text-xs text-dim"><tr className="border-b border-border">{["Class", "Precision", "Recall", "F1", "Support"].map((h) => <th key={h} scope="col" className="px-2 py-2 font-medium">{h}</th>)}</tr></thead>
              <tbody>
                {cls.per.map((r, i) => (
                  <tr key={e.classes[i]} className="border-b border-border/50">
                    <td className="px-2 py-2 font-sans">{e.classes[i]}</td>
                    <td className="px-2 py-2">{fmt(r.precision)}</td><td className="px-2 py-2">{fmt(r.recall)}</td>
                    <td className="px-2 py-2">
                      <div className="flex items-center gap-2"><span className="w-12">{fmt(r.f1)}</span><span className="h-1.5 w-24 rounded-full bg-muted"><span className="block h-full rounded-full bg-primary" style={{ width: `${r.f1 * 100}%` }} /></span></div>
                    </td>
                    <td className="px-2 py-2 text-dim">{r.support}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>
      </div>
    );
  }

  const r = reg!;
  const lo = Math.min(...e.y, ...e.pred), hi = Math.max(...e.y, ...e.pred);
  return (
    <div className="space-y-5">
      <div className="grid grid-cols-2 gap-3 lg:grid-cols-4">
        <Stat label="Test R²" value={fmt(r.r2)} tone={r.r2 > 0.5 ? "good" : "bad"} sub="1 = perfect, 0 = predicts the mean" />
        <Stat label="RMSE" value={fmt(r.rmse)} sub="typical error, target units" />
        <Stat label="MAE" value={fmt(r.mae)} sub="average absolute error" />
        <Stat label="Mean baseline RMSE" value={fmt(r.baselineRmse)} tone="accent" sub={`model is ${fmt((1 - r.rmse / r.baselineRmse) * 100, 0)}% better`} />
      </div>
      <div className="grid gap-5 xl:grid-cols-2">
        <Card title="Predicted vs actual" actions={<Badge tone="blue">{e.engine}</Badge>}>
          <Scatter2D points={e.y.map((y, i) => ({ x: y, y: e.pred[i], c: "#60a5fa" }))} line={[[lo, lo], [hi, hi]]} xLabel="actual" yLabel="predicted" label="Scatter of predicted against actual values with the ideal diagonal" />
          <p className="mt-2 text-xs text-dim">Points on the amber diagonal are perfect predictions.</p>
        </Card>
        <Card title="Residuals">
          <Scatter2D points={e.pred.map((p, i) => ({ x: p, y: e.y[i] - p, c: Math.abs(e.y[i] - p) > 2 * r.rmse ? "#f87171" : "#34d399" }))} hline={0} xLabel="predicted" yLabel="actual − predicted" label="Residual plot" />
          <p className="mt-2 text-xs text-dim">A shapeless band around zero is good. Curves or funnels mean the model is missing structure. Red points are more than 2×RMSE off.</p>
        </Card>
      </div>
    </div>
  );
}
