"use client";
import { motion } from "framer-motion";
import dynamic from "next/dynamic";
import Link from "next/link";
import { ArrowRight, Boxes, Eraser, FileUp, Gamepad2, LineChart, Lock, Network, ScanSearch, Sigma, Target } from "lucide-react";

const Hero3D = dynamic(() => import("./Hero3D"), { ssr: false });

const PIPE = [
  { icon: FileUp, t: "Upload", d: "CSV, JSON, Excel, text, images" },
  { icon: Eraser, t: "Clean", d: "Missing, outliers, duplicates" },
  { icon: ScanSearch, t: "Explore", d: "Distributions, correlation, 3D" },
  { icon: Sigma, t: "Engineer", d: "Encode, scale, leak-free split" },
  { icon: LineChart, t: "Train", d: "Live curves & boundaries" },
  { icon: Target, t: "Evaluate", d: "Confusion, ROC, residuals" },
];

const fade = { initial: { opacity: 0, y: 16 }, whileInView: { opacity: 1, y: 0 }, viewport: { once: true, margin: "-60px" }, transition: { duration: 0.4, ease: "easeOut" as const } };

export function Landing() {
  // static copy keeps the ML libraries out of the landing bundle
  const sup = ["Linear Regression", "Logistic Regression", "Neural Network (TF.js)", "Decision Tree", "Random Forest", "k-Nearest Neighbors", "Naive Bayes"];
  const uns = ["k-Means", "DBSCAN", "PCA", "Anomaly Detection"];
  return (
    <>
      <section className="relative isolate overflow-hidden">
        <div className="pointer-events-none absolute inset-y-0 right-[-20%] left-[30%] -z-10 opacity-80 sm:left-[45%]"><Hero3D /></div>
        <div className="mx-auto max-w-7xl px-4 pt-20 pb-24 sm:pt-28">
          <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.5 }} className="max-w-2xl">
            <span className="inline-flex items-center gap-2 rounded-full border border-border bg-surface/70 px-3 py-1 text-xs text-sub">
              <Lock className="size-3.5 text-good" aria-hidden />Runs 100% in your browser — your data never leaves it
            </span>
            <h1 className="mt-5 text-4xl leading-[1.08] font-semibold tracking-tight sm:text-6xl">
              Watch machine learning <span className="bg-gradient-to-r from-primary via-sky-400 to-accent bg-clip-text text-transparent">actually learn.</span>
            </h1>
            <p className="mt-5 max-w-xl text-lg text-sub">
              Drop in any dataset, clean it, and train supervised, unsupervised and reinforcement-learning models while every epoch, boundary and reward plays out live.
            </p>
            <div className="mt-8 flex flex-wrap gap-3">
              <Link href="/lab" className="group inline-flex h-12 items-center gap-2 rounded-xl bg-accent px-6 font-medium text-black transition-transform hover:-translate-y-0.5">
                Open the Data Lab<ArrowRight className="size-4 transition-transform group-hover:translate-x-0.5" aria-hidden />
              </Link>
              <Link href="/rl" className="inline-flex h-12 items-center gap-2 rounded-xl border border-border bg-surface/70 px-6 font-medium text-fg transition-colors hover:border-primary/60">
                <Gamepad2 className="size-4" aria-hidden />Try the RL Arena
              </Link>
            </div>
          </motion.div>
        </div>
      </section>

      <section className="mx-auto max-w-7xl px-4 pb-16" aria-labelledby="pipe">
        <motion.h2 {...fade} id="pipe" className="mb-6 text-sm font-semibold tracking-widest text-dim uppercase">The whole data-science pipeline</motion.h2>
        <ol className="grid gap-3 sm:grid-cols-3 lg:grid-cols-6">
          {PIPE.map((s, i) => (
            <motion.li key={s.t} {...fade} transition={{ ...fade.transition, delay: i * 0.06 }} className="card p-4">
              <div className="flex items-center gap-2"><span className="num text-xs text-dim">0{i + 1}</span><s.icon className="size-4 text-accent" aria-hidden /></div>
              <div className="mt-2 font-medium">{s.t}</div>
              <div className="text-xs text-dim">{s.d}</div>
            </motion.li>
          ))}
        </ol>
      </section>

      <section className="mx-auto grid max-w-7xl gap-4 px-4 pb-24 md:grid-cols-3" aria-label="Learning paradigms">
        {[
          { icon: Network, t: "Supervised", d: "Learn from labelled examples to classify or predict numbers.", items: sup, href: "/lab" },
          { icon: Boxes, t: "Unsupervised", d: "Find clusters, structure and anomalies with no labels at all.", items: uns, href: "/lab" },
          { icon: Gamepad2, t: "Reinforcement", d: "Agents learn by trial and error from reward and punishment.", items: ["Q-learning", "SARSA", "Gridworld (editable)", "CartPole"], href: "/rl" },
        ].map((c, i) => (
          <motion.div key={c.t} {...fade} transition={{ ...fade.transition, delay: i * 0.08 }}>
            <Link href={c.href} className="card group block h-full p-6 transition-[border,transform] duration-200 hover:-translate-y-1 hover:border-primary/60">
              <c.icon className="size-7 text-primary" aria-hidden />
              <h3 className="mt-4 text-xl font-semibold">{c.t}</h3>
              <p className="mt-1 text-sm text-sub">{c.d}</p>
              <ul className="mt-4 flex flex-wrap gap-1.5">
                {c.items.map((x) => <li key={x} className="rounded-md bg-muted px-2 py-0.5 text-xs text-sub">{x}</li>)}
              </ul>
              <span className="mt-5 inline-flex items-center gap-1 text-sm text-accent">Start<ArrowRight className="size-3.5 transition-transform group-hover:translate-x-0.5" aria-hidden /></span>
            </Link>
          </motion.div>
        ))}
      </section>
    </>
  );
}
