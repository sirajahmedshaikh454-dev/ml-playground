"use client";
import { motion } from "framer-motion";
import { Bot, Eye, Gamepad2, Pause, Play, RotateCcw, Shuffle, Square, StepForward } from "lucide-react";
import { useEffect, useMemo, useRef, useState } from "react";
import {
  ARROWS, CART_DEFAULTS, type CartProgress, type CartState, GRID_DEFAULTS, type Grid, type GridParams, type GridProgress, type Tile,
  cartRollout, cartStep, defaultGrid, greedyPath, trainCart, trainGrid,
} from "@/lib/rl";
import { mulberry32 } from "@/lib/core";
import { useRunner } from "@/lib/useRunner";
import { Curves } from "../charts";
import { Badge, Button, Card, Select, Slider, Stat, cx, fmt, pct } from "../ui";

const SPEEDS = [{ value: "0", label: "Max speed" }, { value: "15", label: "Fast" }, { value: "60", label: "Normal" }, { value: "250", label: "Slow-mo" }];
const smooth = (a: number[], w = 20) => a.map((_, i) => { const s = a.slice(Math.max(0, i - w + 1), i + 1); return s.reduce((x, y) => x + y, 0) / s.length; });

function Controls({ runner, onStart, speed, setSpeed }: { runner: ReturnType<typeof useRunner<never>>; onStart: () => void; speed: string; setSpeed: (v: string) => void }) {
  const busy = runner.status === "running" || runner.status === "paused";
  return (
    <div className="space-y-3">
      <Select label="Playback speed" value={speed} options={SPEEDS} onChange={(v) => { setSpeed(v); runner.setDelay(Number(v)); }} />
      <div className="flex flex-wrap gap-2">
        {!busy && <Button variant="accent" onClick={onStart}><Play className="size-4" aria-hidden />{runner.status === "done" ? "Train again" : "Train agent"}</Button>}
        {runner.status === "running" && <Button onClick={runner.pause}><Pause className="size-4" aria-hidden />Pause</Button>}
        {runner.status === "paused" && <Button variant="accent" onClick={runner.resume}><Play className="size-4" aria-hidden />Resume</Button>}
        {busy && <Button onClick={() => void runner.step()} aria-label="Advance one episode"><StepForward className="size-4" aria-hidden />Step</Button>}
        {busy && <Button variant="danger" onClick={runner.stop}><Square className="size-4" aria-hidden />Stop</Button>}
      </div>
    </div>
  );
}

// ---------------- Gridworld ----------------
const TILE_STYLE: Record<Tile, string> = {
  empty: "bg-surface", wall: "bg-slate-600", trap: "bg-bad/30 ring-1 ring-inset ring-bad/60", goal: "bg-good/25 ring-1 ring-inset ring-good/70", start: "bg-primary/25 ring-1 ring-inset ring-primary/70",
};
const TOOLS: { t: Tile; label: string }[] = [{ t: "wall", label: "Wall" }, { t: "trap", label: "Trap (−)" }, { t: "goal", label: "Goal (+)" }, { t: "start", label: "Start" }, { t: "empty", label: "Erase" }];

function GridWorld() {
  const [grid, setGrid] = useState<Grid>(defaultGrid);
  const [tool, setTool] = useState<Tile>("wall");
  const [p, setP] = useState<GridParams>(GRID_DEFAULTS);
  const [speed, setSpeed] = useState("15");
  const runner = useRunner<GridProgress>();
  const [path, setPath] = useState<number[] | null>(null);
  const [agent, setAgent] = useState<number | null>(null);
  const busy = runner.status === "running" || runner.status === "paused";
  const last = runner.events[runner.events.length - 1];
  const Q = last?.Q;

  const paint = (i: number) => {
    if (busy) return;
    setGrid((g) => {
      const tiles = g.tiles.slice();
      if (tool === "start" || tool === "goal") tiles.forEach((t, j) => { if (t === tool) tiles[j] = "empty"; });
      tiles[i] = tool;
      return { ...g, tiles };
    });
    setPath(null);
  };

  const start = () => { setPath(null); setAgent(null); runner.setDelay(Number(speed)); runner.start(trainGrid(grid, p, Date.now() % 1e6)); };
  // agent walk animation along the greedy path
  useEffect(() => {
    if (!path) return;
    let i = 0;
    const id = setInterval(() => { i++; if (i >= path.length) clearInterval(id); else setAgent(path[i]); }, 260);
    return () => clearInterval(id);
  }, [path]);

  const maxQ = Q ? Q.map((q) => Math.max(...q)) : [];
  const scale = Math.max(1e-6, ...maxQ.map(Math.abs));
  const rewards = runner.events.map((e) => e.reward);
  const avg = smooth(rewards);
  const hist = runner.events.map((e, i) => ({ step: e.episode, reward: e.reward, avg: avg[i], steps: e.steps, epsilon: e.epsilon }));
  const recent = runner.events.slice(-50);
  const success = recent.length ? recent.filter((e) => e.success).length / recent.length : NaN;

  return (
    <div className="grid grid-cols-1 gap-5 xl:grid-cols-[1fr_340px]">
      <div className="space-y-5">
        <Card title="Gridworld" icon={<Gamepad2 className="size-4" />}
          actions={<>
            {TOOLS.map((t) => <Button key={t.t} size="sm" variant={tool === t.t ? "primary" : "ghost"} onClick={() => setTool(t.t)} disabled={busy} aria-pressed={tool === t.t}>{t.label}</Button>)}
            <Button size="sm" onClick={() => { setGrid(defaultGrid()); setPath(null); }} disabled={busy} aria-label="Reset grid"><RotateCcw className="size-3.5" aria-hidden /></Button>
          </>}>
          <div className="mx-auto grid max-w-2xl gap-1" style={{ gridTemplateColumns: `repeat(${grid.w}, minmax(0, 1fr))` }} role="grid" aria-label="Gridworld. Click cells to paint the selected tile.">
            {grid.tiles.map((t, i) => {
              const v = maxQ[i] ?? 0;
              const showQ = Q && t !== "wall";
              const tint = showQ && t === "empty" ? (v >= 0 ? `rgb(59 130 246 / ${(v / scale) * 0.55})` : `rgb(248 113 113 / ${(-v / scale) * 0.55})`) : undefined;
              const onPath = path?.includes(i);
              return (
                <button key={i} role="gridcell" onClick={() => paint(i)} aria-label={`Cell ${i % grid.w},${Math.floor(i / grid.w)}: ${t}${showQ ? `, value ${fmt(v, 2)}` : ""}`}
                  className={cx("relative aspect-square cursor-pointer rounded-md text-[10px] transition-colors", TILE_STYLE[t], onPath && "outline outline-1 outline-accent/70")}
                  style={tint ? { background: tint } : undefined}>
                  {showQ && t === "empty" && <span className="absolute inset-0 grid place-items-center text-base text-fg/80" aria-hidden>{ARROWS[Q[i].indexOf(Math.max(...Q[i]))]}</span>}
                  {showQ && t === "empty" && <span className="num absolute right-1 bottom-0.5 text-[9px] text-dim" aria-hidden>{v.toFixed(2)}</span>}
                  {t === "goal" && <span className="absolute inset-0 grid place-items-center text-xs font-bold text-good" aria-hidden>+{p.goal}</span>}
                  {t === "trap" && <span className="absolute inset-0 grid place-items-center text-xs font-bold text-bad" aria-hidden>{p.trap}</span>}
                  {agent === i && <motion.span layoutId="agent" className="absolute inset-[18%] rounded-full bg-accent shadow-[0_0_18px_4px_rgb(245_158_11/0.5)]" transition={{ type: "spring", stiffness: 300, damping: 26 }} aria-hidden />}
                </button>
              );
            })}
          </div>
          <div className="mt-4 flex flex-wrap items-center gap-2">
            <Button variant="primary" disabled={!Q || busy} onClick={() => { if (!Q) return; const p = greedyPath(grid, Q); setAgent(p[0]); setPath(p); }}><Bot className="size-4" aria-hidden />Replay learned policy</Button>
            <p className="text-xs text-dim">Arrows = best action per cell. Blue = high expected return, red = expected penalty.</p>
          </div>
        </Card>
        {hist.length > 1 && (
          <div className="grid gap-5 lg:grid-cols-2">
            <Card title="Reward per episode"><Curves data={hist} xLabel="episode" keys={[{ key: "reward", label: "reward", color: "#334155" }, { key: "avg", label: "20-ep average", color: "#f59e0b" }]} /></Card>
            <Card title="Steps & exploration"><Curves data={hist} xLabel="episode" keys={[{ key: "steps", label: "steps to finish", color: "#60a5fa" }, { key: "epsilon", label: "ε (explore rate)", color: "#a78bfa", dashed: true }]} /></Card>
          </div>
        )}
      </div>

      <div className="space-y-5">
        <Card title="Agent">
          <div className="space-y-4">
            <Select label="Algorithm" value={p.algo} onChange={(v) => setP({ ...p, algo: v })} options={[{ value: "q", label: "Q-learning (off-policy)" }, { value: "sarsa", label: "SARSA (on-policy, safer)" }]} />
            <Slider label="Episodes" value={p.episodes} min={50} max={2000} step={50} onChange={(v) => setP({ ...p, episodes: v })} disabled={busy} />
            <Slider label="Learning rate α" value={p.alpha} min={0.05} max={1} step={0.05} onChange={(v) => setP({ ...p, alpha: v })} disabled={busy} />
            <Slider label="Discount γ" value={p.gamma} min={0.5} max={0.99} step={0.01} onChange={(v) => setP({ ...p, gamma: v })} disabled={busy} />
            <Slider label="ε decay / episode" value={p.decay} min={0.9} max={0.999} step={0.001} onChange={(v) => setP({ ...p, decay: v })} disabled={busy} />
            <Slider label="Slippery floor" value={p.slip} min={0} max={0.4} step={0.05} onChange={(v) => setP({ ...p, slip: v })} fmt={pct} disabled={busy} />
          </div>
        </Card>
        <Card title="Reward & punishment">
          <div className="space-y-4">
            <Slider label="Goal reward" value={p.goal} min={0.1} max={5} step={0.1} onChange={(v) => setP({ ...p, goal: v })} disabled={busy} />
            <Slider label="Trap penalty" value={p.trap} min={-5} max={-0.1} step={0.1} onChange={(v) => setP({ ...p, trap: v })} disabled={busy} />
            <Slider label="Cost per step" value={p.stepCost} min={-0.5} max={0} step={0.01} onChange={(v) => setP({ ...p, stepCost: v })} disabled={busy} />
            <Controls runner={runner as never} onStart={start} speed={speed} setSpeed={setSpeed} />
          </div>
        </Card>
        {last && (
          <div className="grid grid-cols-2 gap-3">
            <Stat label="Episode" value={`${last.episode}/${last.total}`} />
            <Stat label="Success (last 50)" value={pct(success)} tone={success > 0.8 ? "good" : "accent"} />
            <Stat label="Last reward" value={fmt(last.reward, 2)} tone={last.reward > 0 ? "good" : "bad"} />
            <Stat label="ε" value={fmt(last.epsilon, 3)} />
          </div>
        )}
      </div>
    </div>
  );
}

// ---------------- CartPole ----------------
function CartCanvas({ state, failed }: { state: CartState; failed: boolean }) {
  const ref = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    const c = ref.current;
    if (!c) return;
    const dpr = window.devicePixelRatio || 1, W = c.clientWidth, H = 240;
    c.width = W * dpr; c.height = H * dpr;
    const g = c.getContext("2d")!;
    g.setTransform(dpr, 0, 0, dpr, 0, 0);
    g.clearRect(0, 0, W, H);
    const sx = W / 6, ground = H - 50, cx = W / 2 + state[0] * sx;
    g.strokeStyle = "#334155"; g.lineWidth = 2; g.beginPath(); g.moveTo(0, ground); g.lineTo(W, ground); g.stroke();
    [-2.4, 2.4].forEach((b) => { g.strokeStyle = "#f8717188"; g.beginPath(); g.moveTo(W / 2 + b * sx, ground - 10); g.lineTo(W / 2 + b * sx, ground + 10); g.stroke(); });
    g.fillStyle = failed ? "#f87171" : "#3b82f6"; g.beginPath(); g.roundRect(cx - 34, ground - 26, 68, 26, 6); g.fill();
    const len = 120, tx = cx + Math.sin(state[2]) * len, ty = ground - 26 - Math.cos(state[2]) * len;
    g.strokeStyle = failed ? "#f87171" : "#f59e0b"; g.lineWidth = 8; g.lineCap = "round"; g.beginPath(); g.moveTo(cx, ground - 26); g.lineTo(tx, ty); g.stroke();
    g.fillStyle = "#f8fafc"; g.beginPath(); g.arc(cx, ground - 26, 5, 0, Math.PI * 2); g.fill();
  }, [state, failed]);
  return <canvas ref={ref} className="h-[240px] w-full rounded-xl border border-border bg-surface/60" role="img" aria-label={`Cart at ${state[0].toFixed(2)}, pole angle ${((state[2] * 180) / Math.PI).toFixed(1)} degrees`} />;
}

function CartPole() {
  const runner = useRunner<CartProgress>();
  const [episodes, setEpisodes] = useState(CART_DEFAULTS.episodes);
  const [gamma, setGamma] = useState(CART_DEFAULTS.gamma);
  const [speed, setSpeed] = useState("0");
  const [frames, setFramesRaw] = useState<CartState[] | null>(null);
  const [fi, setFi] = useState(0);
  const setFrames = (f: CartState[] | null) => { setFi(0); setFramesRaw(f); }; // restart playback from frame 0
  const last = runner.events[runner.events.length - 1];
  const busy = runner.status === "running" || runner.status === "paused";

  useEffect(() => {
    if (!frames) return;
    let i = 0;
    const id = setInterval(() => { i++; if (i >= frames.length) clearInterval(id); else setFi(i); }, 20);
    return () => clearInterval(id);
  }, [frames]);

  const randomRollout = () => {
    const r = mulberry32(Date.now() % 1e6);
    let s: CartState = [0, 0, 0.02, 0];
    const f: CartState[] = [s];
    for (let t = 0; t < 500; t++) { const n = cartStep(s, r() < 0.5 ? 0 : 1); f.push(n.s); s = n.s; if (n.done) break; }
    setFrames(f);
  };

  const rewards = runner.events.map((e) => e.reward);
  const avg = useMemo(() => smooth(rewards, 30), [rewards]);
  const hist = runner.events.map((e, i) => ({ step: e.episode, reward: e.reward, avg: avg[i], epsilon: e.epsilon * 500 }));
  const state = frames?.[fi] ?? [0, 0, 0, 0];
  const failed = !!frames && fi === frames.length - 1 && frames.length < 501;

  return (
    <div className="grid grid-cols-1 gap-5 xl:grid-cols-[1fr_340px]">
      <div className="space-y-5">
        <Card title="CartPole" icon={<Gamepad2 className="size-4" />} actions={frames && <Badge tone={failed ? "red" : "green"}>{fi} / 500 steps{failed ? " · fell" : ""}</Badge>}>
          <CartCanvas state={state} failed={failed} />
          <div className="mt-3 flex flex-wrap gap-2">
            <Button variant="primary" disabled={!last || busy} onClick={() => last && setFrames(cartRollout(last.Q, 500, Date.now() % 1e6))}><Eye className="size-4" aria-hidden />Watch learned policy</Button>
            <Button onClick={randomRollout}><Shuffle className="size-4" aria-hidden />Random policy</Button>
          </div>
          <p className="mt-2 text-xs text-dim">+1 for every step the pole stays up, a penalty when it falls or the cart leaves the track. The agent sees angle and angular velocity, discretised into 72 states.</p>
        </Card>
        {hist.length > 1 && (
          <Card title="Learning curve">
            <Curves data={hist} xLabel="episode" keys={[{ key: "reward", label: "steps balanced", color: "#334155" }, { key: "avg", label: "30-ep average", color: "#f59e0b" }, { key: "epsilon", label: "ε × 500", color: "#a78bfa", dashed: true }]} height={260} />
          </Card>
        )}
      </div>
      <div className="space-y-5">
        <Card title="Agent: tabular Q-learning">
          <div className="space-y-4">
            <Slider label="Episodes" value={episodes} min={100} max={2000} step={50} onChange={setEpisodes} disabled={busy} />
            <Slider label="Discount γ" value={gamma} min={0.8} max={0.999} step={0.001} onChange={setGamma} disabled={busy} />
            <Controls runner={runner as never} speed={speed} setSpeed={setSpeed}
              onStart={() => { setFrames(null); runner.setDelay(Number(speed)); runner.start(trainCart({ ...CART_DEFAULTS, episodes, gamma }, Date.now() % 1e6)); }} />
          </div>
        </Card>
        {last && (
          <div className="grid grid-cols-2 gap-3">
            <Stat label="Episode" value={`${last.episode}/${last.total}`} />
            <Stat label="Avg (last 30)" value={fmt(avg[avg.length - 1], 0)} tone={avg[avg.length - 1] > 400 ? "good" : "accent"} />
            <Stat label="Best" value={Math.max(...rewards)} />
            <Stat label="ε" value={fmt(last.epsilon, 3)} />
          </div>
        )}
      </div>
    </div>
  );
}

export function RLArena() {
  const [tab, setTab] = useState<"grid" | "cart">("grid");
  return (
    <div className="mx-auto max-w-7xl px-4 py-6">
      <div className="mb-5 flex flex-wrap items-center gap-3">
        <h1 className="text-2xl font-semibold tracking-tight">RL Arena</h1>
        <span className="text-sm text-dim">Agents learn from reward and punishment alone</span>
        <div role="tablist" aria-label="Environment" className="ml-auto flex rounded-xl border border-border bg-surface/60 p-1">
          {([["grid", "Gridworld"], ["cart", "CartPole"]] as const).map(([k, l]) => (
            <button key={k} role="tab" aria-selected={tab === k} onClick={() => setTab(k)}
              className={cx("relative cursor-pointer rounded-lg px-4 py-1.5 text-sm transition-colors", tab === k ? "text-black" : "text-sub hover:text-fg")}>
              {tab === k && <motion.span layoutId="rl-tab" className="absolute inset-0 rounded-lg bg-accent" transition={{ type: "spring", stiffness: 400, damping: 32 }} />}
              <span className="relative">{l}</span>
            </button>
          ))}
        </div>
      </div>
      {tab === "grid" ? <GridWorld /> : <CartPole />}
    </div>
  );
}
