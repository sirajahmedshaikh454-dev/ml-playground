// Runs an in-browser engine off the main thread. Pull-based: the page asks for each
// step ("next"), so pause/step/speed keep working and the worker never races ahead.
import type { Predictor, Progress } from "./core";
import { type Ctx, ENGINES } from "./ml";

export type ToWorker =
  | { type: "train"; engine: string; ctx: Ctx }
  | { type: "next" }
  | { type: "stop" }
  | { type: "predict"; id: number; X: number[][] };
export type FromWorker =
  | { type: "progress"; p: Omit<Progress, "predictor">; hasPredictor: boolean }
  | { type: "done" }
  | { type: "error"; message: string }
  | { type: "pred"; id: number; pred: number[]; proba?: number[][] };

// TF.js yields between batches via requestAnimationFrame, which Chrome also exposes in workers
// but stops firing while the page is occluded -> training would hang. Nothing renders here,
// so a macrotask is the right yield. Must run before TF.js is (dynamically) imported.
self.requestAnimationFrame = (cb: FrameRequestCallback) => setTimeout(() => cb(performance.now()), 0) as unknown as number;

let gen: AsyncGenerator<Progress> | null = null;
let pr: Predictor | undefined;
const post = (m: FromWorker) => self.postMessage(m);

async function handle(m: ToWorker) {
  switch (m.type) {
    case "train": {
      const engine = ENGINES.find((e) => e.id === m.engine);
      if (!engine) throw new Error(`Unknown engine ${m.engine}`);
      gen = engine.run(m.ctx);
      return;
    }
    case "next": {
      const r = await gen!.next();
      if (r.done) return post({ type: "done" });
      const { predictor, ...p } = r.value;
      if (predictor) pr = predictor;
      return post({ type: "progress", p, hasPredictor: !!predictor });
    }
    case "stop":
      await gen?.return(undefined); // keeps the trained model alive for predictions
      return;
    case "predict":
      try {
        const out = !pr ? { pred: [] } : pr.predictAsync ? await pr.predictAsync(m.X) : { pred: pr.predict(m.X), proba: pr.proba?.(m.X) };
        return post({ type: "pred", id: m.id, ...out });
      } catch {
        return post({ type: "pred", id: m.id, pred: [] });
      }
  }
}

// serialize: a predict never runs in the middle of a training step (e.g. inside tf fit)
let chain = Promise.resolve();
self.onmessage = (ev: MessageEvent<ToWorker>) => {
  chain = chain.then(() => handle(ev.data)).catch((e: unknown) => post({ type: "error", message: e instanceof Error ? e.message : String(e) }));
};
