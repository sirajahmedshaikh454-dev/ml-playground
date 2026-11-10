// Wraps a browser engine so it trains in a Web Worker. Same Engine interface; the trained
// model stays in the worker and predicts via messages (predictAsync), like the Python ones.
import type { Predictor, Progress } from "./core";
import { type Ctx, ENGINES, type Engine } from "./ml";
import type { FromWorker, ToWorker } from "./train.worker";

async function* runInWorker(engine: string, ctx: Ctx): AsyncGenerator<Progress> {
  const w = new Worker(new URL("./train.worker.ts", import.meta.url), { type: "module" });
  const send = (m: ToWorker) => w.postMessage(m);
  const inbox: FromWorker[] = [];
  let wake: (() => void) | null = null;
  const preds = new Map<number, (r: { pred: number[]; proba?: number[][] }) => void>();
  let nextId = 0, disposed = false;

  w.onmessage = (ev: MessageEvent<FromWorker>) => {
    const m = ev.data;
    if (m.type === "pred") { preds.get(m.id)?.({ pred: m.pred, proba: m.proba }); preds.delete(m.id); return; }
    inbox.push(m);
    wake?.();
  };
  w.onerror = (e) => { inbox.push({ type: "error", message: e.message || "Training worker failed to start" }); wake?.(); };
  const receive = async () => {
    while (!inbox.length) await new Promise<void>((r) => (wake = r));
    return inbox.shift()!;
  };

  const dispose = () => {
    if (disposed) return;
    disposed = true;
    w.terminate();
    preds.forEach((r) => r({ pred: [] })); // nobody waits forever on a dead worker
    preds.clear();
  };
  const predictor: Predictor = {
    predict: () => { throw new Error("Worker model: use predictAsync"); },
    predictAsync: (X) => disposed ? Promise.resolve({ pred: [] }) : new Promise((r) => { const id = ++nextId; preds.set(id, r); send({ type: "predict", id, X }); }),
    dispose,
  };

  let handedOut = false, finished = false;
  send({ type: "train", engine, ctx });
  try {
    for (;;) {
      send({ type: "next" });
      const m = await receive();
      if (m.type === "error") throw new Error(m.message);
      if (m.type === "done") { finished = true; return; }
      if (m.type === "progress") {
        if (m.hasPredictor) handedOut = true;
        yield { ...m.p, predictor: m.hasPredictor ? predictor : undefined };
      }
    }
  } finally {
    // no model handed out (clustering, or failed early) -> free the worker now;
    // otherwise it lives until the predictor is disposed (next run / new dataset)
    if (!handedOut) dispose();
    else if (!finished && !disposed) send({ type: "stop" });
  }
}

export const WORKER_ENGINES: Engine[] = ENGINES.map((e) => ({ ...e, run: (c: Ctx) => runInWorker(e.id, c) }));
