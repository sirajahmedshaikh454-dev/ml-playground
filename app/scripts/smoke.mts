// Runs every engine on the samples headlessly: `npx tsx scripts/smoke.ts`
import { SAMPLES } from "../src/lib/samples";
import { applyOps } from "../src/lib/clean";
import { defaultCfg, prepare, selfCheck as fcheck } from "../src/lib/features";
import { selfCheck as mcheck } from "../src/lib/metrics";
import { ENGINES } from "../src/lib/ml";
import type { Progress } from "../src/lib/core";

fcheck(); mcheck();
let fail = 0;
for (const s of SAMPLES) {
  const ds = applyOps(s.make(), [{ op: "dedupe" }, { op: "dropMissing" }, ...(s.id === "houses" ? [{ op: "outliers" as const, col: "price_k", how: "remove" as const, k: 3 }] : [])]);
  for (const task of [defaultCfg(ds).task, "clustering"] as const) {
    const cfg = { ...defaultCfg(ds), task };
    const p = prepare(ds, cfg);
    for (const e of ENGINES.filter((e) => e.tasks.includes(task))) {
      const params = Object.fromEntries(e.params.map((q) => [q.key, e.id === "mlp" && q.key === "epochs" ? 5 : q.def]));
      const ctx = { task, Xtr: task === "clustering" ? p.Xall : p.Xtr, ytr: task === "clustering" ? p.yall : p.ytr, Xva: p.Xte, yva: p.yte, k: p.classes.length, params, seed: 1 };
      try {
        let last: Progress | undefined;
        for await (const ev of e.run(ctx)) last = ev;
        const m = last!.metrics;
        const bad = Object.values(m).some((v) => !Number.isFinite(v));
        if (bad) fail++;
        console.log(`${bad ? "FAIL" : "ok  "} ${s.id.padEnd(7)} ${task.padEnd(14)} ${e.id.padEnd(9)} ${Object.entries(m).map(([k, v]) => `${k}=${(+v).toFixed(3)}`).join(" ")}`);
        last!.predictor?.dispose?.();
      } catch (err) { fail++; console.log(`FAIL ${s.id} ${task} ${e.id}: ${(err as Error).message}`); }
    }
  }
}
process.exit(fail ? 1 : 0);
