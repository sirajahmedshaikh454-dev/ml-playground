// Client for the optional Python backend (backend/main.py). Same Engine interface as the
// in-browser models, so the Train step treats them identically.
import type { Predictor, Progress } from "./core";
import type { Ctx, Engine } from "./ml";

export const BACKEND = process.env.NEXT_PUBLIC_ML_BACKEND ?? "http://127.0.0.1:8000";

export async function backendModels(): Promise<string[]> {
  try {
    const r = await fetch(`${BACKEND}/health`, { signal: AbortSignal.timeout(1500) });
    return r.ok ? ((await r.json()).models as string[]) : [];
  } catch {
    return []; // not running: the Python engines simply stay hidden
  }
}

const post = async (path: string, body: unknown, signal?: AbortSignal) => {
  const r = await fetch(`${BACKEND}${path}`, { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify(body), signal });
  if (!r.ok) throw new Error((await r.json().catch(() => null))?.detail?.toString() ?? `Backend error ${r.status}`);
  return r;
};

const remotePredictor = (id: string): Predictor => ({
  predict: () => { throw new Error("Remote model: use predictAsync"); },
  predictAsync: async (X) => {
    if (!X.length) return { pred: [] };
    const j = await (await post("/predict", { id, X })).json();
    return { pred: j.pred, proba: j.proba ?? undefined };
  },
});

function remote(model: string, label: string, blurb: string, params: Engine["params"]): Engine {
  return {
    id: `py-${model}`, label: `${label} · Python`, blurb, family: "supervised", tasks: ["classification", "regression"], params, remote: true,
    run: async function* (c: Ctx): AsyncGenerator<Progress> {
      const ctrl = new AbortController(); // aborting closes the stream; the server then stops training
      try {
        const res = await post("/train", { task: c.task, model, params: c.params, Xtr: c.Xtr, ytr: c.ytr, Xva: c.Xva, yva: c.yva, k: c.k }, ctrl.signal);
        const reader = res.body!.pipeThrough(new TextDecoderStream()).getReader();
        let buf = "";
        let last: Progress | undefined;
        for (;;) {
          const { value, done } = await reader.read();
          if (done) throw new Error("Backend closed the stream early");
          buf += value;
          for (let i = buf.indexOf("\n\n"); i >= 0; i = buf.indexOf("\n\n")) {
            const line = buf.slice(0, i);
            buf = buf.slice(i + 2);
            if (!line.startsWith("data: ")) continue;
            const ev = JSON.parse(line.slice(6));
            if (ev.type === "error") throw new Error(ev.message);
            if (ev.type === "progress") yield (last = { step: ev.step, total: ev.total, metrics: ev.metrics });
            if (ev.type === "done") { yield { ...(last ?? { step: 1, total: 1, metrics: {} }), predictor: remotePredictor(ev.id) }; return; }
          }
        }
      } finally {
        ctrl.abort();
      }
    },
  };
}

export const REMOTE_ENGINES: Engine[] = [
  remote("svm", "SVM (RBF)", "Support vector machine with an RBF kernel. Strong on small, non-linear data.", [
    { key: "C", label: "C (margin hardness)", min: 0.1, max: 20, step: 0.1, def: 1 },
    { key: "gamma", label: "Kernel width γ", min: 0.01, max: 2, step: 0.01, def: 0.2 },
  ]),
  remote("gboost", "Gradient Boosting", "Trees fitted to each other's mistakes, replayed stage by stage.", [
    { key: "trees", label: "Trees", min: 10, max: 500, step: 10, def: 150 },
    { key: "lr", label: "Learning rate", min: 0.01, max: 0.5, step: 0.01, def: 0.1 },
    { key: "depth", label: "Tree depth", min: 1, max: 8, step: 1, def: 3 },
  ]),
  remote("forest", "Random Forest (large)", "scikit-learn forest on all cores, grown in chunks.", [
    { key: "trees", label: "Trees", min: 10, max: 1000, step: 10, def: 300 },
    { key: "depth", label: "Max depth (0 = unlimited)", min: 0, max: 30, step: 1, def: 0 },
  ]),
  remote("mlp", "MLP", "scikit-learn neural network trained one epoch at a time.", [
    { key: "epochs", label: "Epochs", min: 10, max: 300, step: 10, def: 80 },
    { key: "hidden", label: "Units / layer", min: 8, max: 256, step: 8, def: 64 },
    { key: "layers", label: "Hidden layers", min: 1, max: 4, step: 1, def: 2 },
    { key: "lr", label: "Learning rate", min: 0.0005, max: 0.02, step: 0.0005, def: 0.002 },
    { key: "l2", label: "L2 penalty", min: 0, max: 0.01, step: 0.0001, def: 0.0001 },
  ]),
];
