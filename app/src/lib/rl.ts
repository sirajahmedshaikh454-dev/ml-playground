// Reinforcement learning: tabular Q-learning / SARSA on an editable gridworld,
// and Q-learning on a discretized CartPole. Rewards and penalties are explicit.
import { mulberry32 } from "./core";

export type Tile = "empty" | "wall" | "trap" | "goal" | "start";
export interface Grid { w: number; h: number; tiles: Tile[] }
export const ACTIONS = [[0, -1], [1, 0], [0, 1], [-1, 0]] as const; // up right down left
export const ARROWS = ["↑", "→", "↓", "←"];

export function defaultGrid(): Grid {
  const w = 8, h = 6;
  const tiles: Tile[] = new Array(w * h).fill("empty");
  const set = (x: number, y: number, t: Tile) => (tiles[y * w + x] = t);
  set(0, 5, "start");
  set(7, 0, "goal");
  [[2, 1], [2, 2], [2, 3], [5, 2], [5, 3], [5, 4], [6, 2]].forEach(([x, y]) => set(x, y, "wall"));
  [[4, 0], [3, 4], [7, 3]].forEach(([x, y]) => set(x, y, "trap"));
  return { w, h, tiles };
}

export interface GridParams {
  algo: "q" | "sarsa";
  episodes: number; alpha: number; gamma: number;
  epsilon: number; decay: number; slip: number;
  stepCost: number; goal: number; trap: number;
}
export const GRID_DEFAULTS: GridParams = { algo: "q", episodes: 400, alpha: 0.5, gamma: 0.95, epsilon: 1, decay: 0.99, slip: 0, stepCost: -0.04, goal: 1, trap: -1 };

export interface GridProgress { episode: number; total: number; reward: number; steps: number; epsilon: number; success: boolean; Q: number[][] }

function gridStep(g: Grid, s: number, a: number, p: GridParams) {
  const x = s % g.w, y = Math.floor(s / g.w);
  const nx = x + ACTIONS[a][0], ny = y + ACTIONS[a][1];
  const blocked = nx < 0 || ny < 0 || nx >= g.w || ny >= g.h || g.tiles[ny * g.w + nx] === "wall";
  const ns = blocked ? s : ny * g.w + nx;
  const t = g.tiles[ns];
  if (t === "goal") return { ns, r: p.goal, done: true };
  if (t === "trap") return { ns, r: p.trap, done: true };
  return { ns, r: p.stepCost, done: false };
}
const argmax = (a: number[]) => a.reduce((b, v, i) => (v > a[b] ? i : b), 0);

export function* trainGrid(g: Grid, p: GridParams, seed = 1): Generator<GridProgress> {
  const r = mulberry32(seed);
  const start = Math.max(0, g.tiles.indexOf("start"));
  const Q = g.tiles.map(() => [0, 0, 0, 0]);
  let eps = p.epsilon;
  const pick = (s: number) => (r() < eps ? Math.floor(r() * 4) : argmax(Q[s]));
  const act = (a: number) => (r() < p.slip ? Math.floor(r() * 4) : a); // slippery floor
  for (let ep = 1; ep <= p.episodes; ep++) {
    let s = start, a = pick(s), total = 0, steps = 0, success = false;
    for (; steps < 200; steps++) {
      const { ns, r: rew, done } = gridStep(g, s, act(a), p);
      total += rew;
      const na = pick(ns);
      const target = done ? rew : rew + p.gamma * (p.algo === "q" ? Math.max(...Q[ns]) : Q[ns][na]);
      Q[s][a] += p.alpha * (target - Q[s][a]);
      s = ns; a = na;
      if (done) { success = g.tiles[ns] === "goal"; steps++; break; }
    }
    eps = Math.max(0.01, eps * p.decay);
    yield { episode: ep, total: p.episodes, reward: total, steps, epsilon: eps, success, Q };
  }
}

export function greedyPath(g: Grid, Q: number[][]) {
  const path = [Math.max(0, g.tiles.indexOf("start"))];
  const seen = new Set(path);
  const p = { ...GRID_DEFAULTS };
  for (let i = 0; i < 100; i++) {
    const s = path[path.length - 1];
    const { ns, done } = gridStep(g, s, argmax(Q[s]), p);
    path.push(ns);
    if (done || seen.has(ns)) break;
    seen.add(ns);
  }
  return path;
}

// ---------------- CartPole ----------------
export type CartState = [number, number, number, number]; // x, x_dot, theta, theta_dot
const TH = (12 * 2 * Math.PI) / 360;
export function cartStep([x, xd, th, thd]: CartState, a: number): { s: CartState; done: boolean } {
  const g = 9.8, mc = 1, mp = 0.1, l = 0.5, f = a ? 10 : -10, tau = 0.02;
  const ct = Math.cos(th), st = Math.sin(th);
  const tmp = (f + mp * l * thd * thd * st) / (mc + mp);
  const thacc = (g * st - ct * tmp) / (l * (4 / 3 - (mp * ct * ct) / (mc + mp)));
  const xacc = tmp - (mp * l * thacc * ct) / (mc + mp);
  const s: CartState = [x + tau * xd, xd + tau * xacc, th + tau * thd, thd + tau * thacc];
  return { s, done: Math.abs(s[0]) > 2.4 || Math.abs(s[2]) > TH };
}
const BINS = [1, 1, 6, 12]; // x and x_dot barely matter; angle + angular velocity do
const LIM = [2.4, 1, TH, 0.87];
function disc(s: CartState) {
  let idx = 0;
  for (let i = 0; i < 4; i++) {
    const b = Math.min(BINS[i] - 1, Math.max(0, Math.floor(((s[i] + LIM[i]) / (2 * LIM[i])) * BINS[i])));
    idx = idx * BINS[i] + b;
  }
  return idx;
}
export interface CartParams { episodes: number; gamma: number; maxSteps: number }
export const CART_DEFAULTS: CartParams = { episodes: 800, gamma: 0.99, maxSteps: 500 };
export interface CartProgress { episode: number; total: number; reward: number; epsilon: number; Q: number[][] }

export function* trainCart(p: CartParams, seed = 3): Generator<CartProgress> {
  const r = mulberry32(seed);
  const Q = Array.from({ length: BINS.reduce((a, b) => a * b, 1) }, () => [0, 0]);
  const init = (): CartState => [0, 0, 0, 0].map(() => (r() - 0.5) * 0.1) as CartState;
  for (let ep = 1; ep <= p.episodes; ep++) {
    // classic decaying schedules for tabular CartPole
    const eps = Math.max(0.01, Math.min(1, 1 - Math.log10(ep / 25)));
    const alpha = Math.max(0.1, Math.min(0.5, 1 - Math.log10(ep / 25)));
    let s = init(), total = 0;
    for (let t = 0; t < p.maxSteps; t++) {
      const d = disc(s);
      const a = r() < eps ? (r() < 0.5 ? 0 : 1) : argmax(Q[d]);
      const { s: ns, done } = cartStep(s, a);
      const rew = done ? -1 : 1; // falling over is punished
      total += 1;
      Q[d][a] += alpha * ((done ? rew : rew + p.gamma * Math.max(...Q[disc(ns)])) - Q[d][a]);
      s = ns;
      if (done) break;
    }
    yield { episode: ep, total: p.episodes, reward: total, epsilon: eps, Q };
  }
}

export function cartRollout(Q: number[][], maxSteps = 500, seed = 9) {
  const r = mulberry32(seed);
  let s: CartState = [0, 0, 0, 0].map(() => (r() - 0.5) * 0.1) as CartState;
  const frames: CartState[] = [s];
  for (let t = 0; t < maxSteps; t++) {
    const { s: ns, done } = cartStep(s, argmax(Q[disc(s)]));
    frames.push(ns);
    s = ns;
    if (done) break;
  }
  return frames;
}
