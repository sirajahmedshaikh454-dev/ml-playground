import { defaultGrid, trainGrid, greedyPath, GRID_DEFAULTS, trainCart, cartRollout, CART_DEFAULTS } from "../src/lib/rl";
const g = defaultGrid();
for (const algo of ["q", "sarsa"] as const) {
  let last; for (const p of trainGrid(g, { ...GRID_DEFAULTS, algo })) last = p;
  const path = greedyPath(g, last!.Q);
  const ok = g.tiles[path[path.length - 1]] === "goal";
  console.log(`${ok ? "ok  " : "FAIL"} grid ${algo}: path ${path.length - 1} steps, last reward ${last!.reward.toFixed(2)}`);
  if (!ok) process.exitCode = 1;
}
let lastC; const rs: number[] = [];
for (const p of trainCart(CART_DEFAULTS)) { lastC = p; rs.push(p.reward); }
const avg = rs.slice(-50).reduce((a, b) => a + b, 0) / 50;
const roll = cartRollout(lastC!.Q).length - 1;
console.log(`${roll >= 150 ? "ok  " : "FAIL"} cartpole: last-50 avg ${avg.toFixed(0)}, greedy rollout ${roll} steps`);
if (roll < 150) process.exitCode = 1;
