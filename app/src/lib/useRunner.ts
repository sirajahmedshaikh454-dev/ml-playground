"use client";
import { useCallback, useEffect, useRef, useState } from "react";

export type RunStatus = "idle" | "running" | "paused" | "done" | "error";

// Drives any (async) generator from the UI: run / pause / step / stop at an adjustable speed.
// Pulls as many steps as fit in a ~14ms frame budget, then commits one React update,
// so a 600-epoch run does not mean 600 re-renders of every chart.
export function useRunner<T>() {
  const gen = useRef<AsyncIterator<T> | Iterator<T> | null>(null);
  const statusRef = useRef<RunStatus>("idle");
  const delayRef = useRef(0);
  const [status, setStatusState] = useState<RunStatus>("idle");
  const [events, setEvents] = useState<T[]>([]);
  const [error, setError] = useState<string | null>(null);
  const onDone = useRef<((all: T[]) => void) | null>(null);
  const buf = useRef<T[]>([]);

  const setStatus = (s: RunStatus) => { statusRef.current = s; setStatusState(s); };

  const pull = useCallback(async (budgetMs: number) => {
    const g = gen.current;
    if (!g) return true;
    const t0 = performance.now();
    const fresh: T[] = [];
    let finished = false;
    try {
      do {
        const r = await g.next();
        if (r.done) { finished = true; break; }
        fresh.push(r.value);
      } while (performance.now() - t0 < budgetMs);
    } catch (e) {
      setError((e as Error).message);
      setStatus("error");
      return true;
    }
    if (fresh.length) {
      buf.current = buf.current.concat(fresh);
      setEvents(buf.current);
    }
    if (finished) {
      gen.current = null;
      setStatus("done");
      onDone.current?.(buf.current);
    }
    return finished;
  }, []);

  const loop = useCallback(async () => {
    while (statusRef.current === "running") {
      const finished = await pull(delayRef.current > 0 ? 0 : 14);
      if (finished) return;
      await new Promise((r) => (delayRef.current > 0 ? setTimeout(r, delayRef.current) : requestAnimationFrame(() => r(null))));
    }
  }, [pull]);

  const start = useCallback((g: AsyncIterator<T> | Iterator<T>, done?: (all: T[]) => void) => {
    void gen.current?.return?.();
    gen.current = g;
    onDone.current = done ?? null;
    buf.current = [];
    setEvents([]);
    setError(null);
    setStatus("running");
    void loop();
  }, [loop]);

  const pause = useCallback(() => setStatus("paused"), []);
  const resume = useCallback(() => { if (gen.current) { setStatus("running"); void loop(); } }, [loop]);
  const step = useCallback(async () => { if (gen.current) { setStatus("paused"); await pull(0); } }, [pull]);
  const stop = useCallback(() => {
    void gen.current?.return?.();
    gen.current = null;
    if (buf.current.length) { setStatus("done"); onDone.current?.(buf.current); } else setStatus("idle");
  }, []);
  const setDelay = useCallback((ms: number) => { delayRef.current = ms; }, []);

  useEffect(() => () => { statusRef.current = "idle"; void gen.current?.return?.(); }, []);

  return { status, events, error, start, pause, resume, step, stop, setDelay };
}
