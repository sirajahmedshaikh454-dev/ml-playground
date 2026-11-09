"use client";
import { clsx } from "clsx";
import { motion } from "framer-motion";
import type { ButtonHTMLAttributes, ReactNode } from "react";

export const cx = clsx;

export function Card({ title, icon, actions, children, className }: { title?: ReactNode; icon?: ReactNode; actions?: ReactNode; children: ReactNode; className?: string }) {
  return (
    <motion.section
      initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.3, ease: "easeOut" }}
      className={cx("card p-4 sm:p-5", className)}>
      {(title || actions) && (
        <div className="mb-3 flex flex-wrap items-center gap-2">
          {icon && <span className="text-primary" aria-hidden>{icon}</span>}
          {title && <h2 className="text-sm font-semibold tracking-wide text-sub uppercase">{title}</h2>}
          <div className="ml-auto flex flex-wrap items-center gap-2">{actions}</div>
        </div>
      )}
      {children}
    </motion.section>
  );
}

type BtnProps = ButtonHTMLAttributes<HTMLButtonElement> & { variant?: "primary" | "accent" | "ghost" | "danger"; size?: "sm" | "md" };
export function Button({ variant = "ghost", size = "md", className, ...p }: BtnProps) {
  return (
    <button
      {...p}
      className={cx(
        "inline-flex cursor-pointer items-center justify-center gap-2 rounded-lg font-medium transition-[background,color,transform] duration-150 active:scale-[0.97] disabled:cursor-not-allowed disabled:opacity-40",
        size === "sm" ? "h-8 px-3 text-xs" : "h-10 px-4 text-sm",
        variant === "primary" && "bg-primary text-white hover:bg-primary/85",
        variant === "accent" && "bg-accent text-black hover:bg-accent/85",
        variant === "ghost" && "border border-border bg-muted/50 text-sub hover:bg-muted hover:text-fg",
        variant === "danger" && "border border-bad/40 bg-bad/10 text-bad hover:bg-bad/20",
        className,
      )}
    />
  );
}

export function Field({ label, hint, children }: { label: string; hint?: string; children: ReactNode }) {
  return (
    <label className="block text-sm">
      <span className="mb-1 block text-xs font-medium text-dim">{label}</span>
      {children}
      {hint && <span className="mt-1 block text-xs text-dim/80">{hint}</span>}
    </label>
  );
}

export function Select<T extends string>({ value, onChange, options, label, className }: { value: T; onChange: (v: T) => void; options: { value: T; label: string }[]; label: string; className?: string }) {
  return (
    <Field label={label}>
      <select value={value} onChange={(e) => onChange(e.target.value as T)}
        className={cx("h-10 w-full cursor-pointer rounded-lg border border-border bg-surface px-3 text-sm text-fg", className)}>
        {options.map((o) => <option key={o.value} value={o.value}>{o.label}</option>)}
      </select>
    </Field>
  );
}

export function Slider({ label, value, min, max, step, onChange, fmt, disabled }: { label: string; value: number; min: number; max: number; step: number; onChange: (v: number) => void; fmt?: (v: number) => string; disabled?: boolean }) {
  return (
    <label className="block">
      <span className="mb-1 flex items-center justify-between text-xs text-dim">
        <span>{label}</span>
        <span className="num text-sub">{fmt ? fmt(value) : value}</span>
      </span>
      <input type="range" className="w-full cursor-pointer" min={min} max={max} step={step} value={value} disabled={disabled}
        onChange={(e) => onChange(Number(e.target.value))} />
    </label>
  );
}

export function Stat({ label, value, sub, tone }: { label: string; value: ReactNode; sub?: ReactNode; tone?: "good" | "bad" | "accent" }) {
  return (
    <div className="rounded-xl border border-border bg-surface/60 px-3 py-2.5">
      <div className="text-[11px] tracking-wide text-dim uppercase">{label}</div>
      <div className={cx("num mt-0.5 text-xl font-semibold", tone === "good" && "text-good", tone === "bad" && "text-bad", tone === "accent" && "text-accent")}>{value}</div>
      {sub && <div className="mt-0.5 text-xs text-dim">{sub}</div>}
    </div>
  );
}

export function Badge({ children, tone = "default" }: { children: ReactNode; tone?: "default" | "blue" | "amber" | "green" | "red" }) {
  return (
    <span className={cx("inline-flex items-center rounded-md px-1.5 py-0.5 text-[11px] font-medium",
      tone === "default" && "bg-muted text-sub", tone === "blue" && "bg-primary/15 text-primary",
      tone === "amber" && "bg-accent/15 text-accent", tone === "green" && "bg-good/15 text-good", tone === "red" && "bg-bad/15 text-bad")}>
      {children}
    </span>
  );
}

export const fmt = (v: number | undefined, d = 3) =>
  v == null || !Number.isFinite(v) ? "—" : Math.abs(v) >= 1000 ? v.toLocaleString(undefined, { maximumFractionDigits: 0 }) : v.toFixed(d);
export const pct = (v: number | undefined) => (v == null || !Number.isFinite(v) ? "—" : `${(v * 100).toFixed(1)}%`);
