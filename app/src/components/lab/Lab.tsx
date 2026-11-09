"use client";
import { motion } from "framer-motion";
import { Check, ChevronLeft, ChevronRight } from "lucide-react";
import { useLab } from "@/lib/store";
import { Button, cx } from "../ui";
import { CleanStep } from "./CleanStep";
import { DataStep } from "./DataStep";
import { EvalStep } from "./EvalStep";
import { ExploreStep } from "./ExploreStep";
import { FeatureStep } from "./FeatureStep";
import { TrainStep } from "./TrainStep";

const STEPS = [
  { title: "Data", sub: "Upload or sample", C: DataStep },
  { title: "Clean", sub: "Missing, outliers, dupes", C: CleanStep },
  { title: "Explore", sub: "Distributions + 3D", C: ExploreStep },
  { title: "Features", sub: "Target, encode, split", C: FeatureStep },
  { title: "Train", sub: "Watch it learn", C: TrainStep },
  { title: "Evaluate", sub: "Honest test scores", C: EvalStep },
];

export function Lab() {
  const { raw, step, setStep, evalResult } = useLab();
  const enabled = (i: number) => i === 0 || (!!raw && (i < 5 || !!evalResult));
  const { C } = STEPS[step];

  return (
    <div className="mx-auto grid grid-cols-1 max-w-7xl gap-6 px-4 py-6 lg:grid-cols-[220px_1fr]">
      <nav aria-label="Pipeline steps" className="lg:sticky lg:top-20 lg:self-start">
        <ol className="scroll-thin flex gap-2 overflow-x-auto pb-2 lg:flex-col lg:overflow-visible lg:pb-0">
          {STEPS.map((s, i) => {
            const done = enabled(i) && i < step;
            return (
              <li key={s.title} className="shrink-0">
                <button onClick={() => enabled(i) && setStep(i)} disabled={!enabled(i)} aria-current={i === step ? "step" : undefined}
                  className={cx("group flex w-full cursor-pointer items-center gap-3 rounded-xl border px-3 py-2.5 text-left transition-colors disabled:cursor-not-allowed disabled:opacity-40",
                    i === step ? "border-accent/60 bg-accent/10" : "border-transparent hover:bg-muted/50")}>
                  <span className={cx("num grid size-7 shrink-0 place-items-center rounded-full text-xs font-semibold",
                    i === step ? "bg-accent text-black" : done ? "bg-primary/20 text-primary" : "bg-muted text-dim")}>
                    {done ? <Check className="size-3.5" aria-hidden /> : i + 1}
                  </span>
                  <span>
                    <span className="block text-sm font-medium">{s.title}</span>
                    <span className="hidden text-[11px] text-dim lg:block">{s.sub}</span>
                  </span>
                </button>
              </li>
            );
          })}
        </ol>
      </nav>

      <div className="min-w-0">
        <div className="mb-5 flex flex-wrap items-center gap-3">
          <h1 className="text-2xl font-semibold tracking-tight">{STEPS[step].title}</h1>
          <span className="text-sm text-dim">{STEPS[step].sub}</span>
          <div className="ml-auto flex gap-2">
            <Button size="sm" onClick={() => setStep(step - 1)} disabled={step === 0} aria-label="Previous step"><ChevronLeft className="size-4" aria-hidden /></Button>
            <Button size="sm" variant="primary" onClick={() => setStep(step + 1)} disabled={step === STEPS.length - 1 || !enabled(step + 1)}>Next<ChevronRight className="size-4" aria-hidden /></Button>
          </div>
        </div>
        {/* enter-only transition: switching never waits on an exit animation */}
        <motion.div key={step} initial={{ opacity: 0, x: 16 }} animate={{ opacity: 1, x: 0 }} transition={{ duration: 0.22, ease: "easeOut" }}>
          <C />
        </motion.div>
      </div>
    </div>
  );
}
