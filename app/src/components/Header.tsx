"use client";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { motion } from "framer-motion";
import { BrainCircuit } from "lucide-react";

const NAV = [
  { href: "/lab", label: "Data Lab" },
  { href: "/rl", label: "RL Arena" },
  { href: "/compare", label: "Experiments" },
];

export function Header() {
  const path = usePathname();
  return (
    <header className="sticky top-0 z-40 border-b border-border/60 bg-bg/75 backdrop-blur-xl">
      <div className="mx-auto flex h-14 max-w-7xl items-center gap-4 px-4">
        <Link href="/" className="flex items-center gap-2 font-semibold tracking-tight" aria-label="ML Playground home">
          <span className="grid size-8 place-items-center rounded-lg bg-gradient-to-br from-primary to-accent text-black">
            <BrainCircuit className="size-5" aria-hidden />
          </span>
          <span className="hidden sm:inline">ML Playground</span>
        </Link>
        <nav className="ml-auto flex items-center gap-1" aria-label="Main">
          {NAV.map((n) => {
            const active = path.startsWith(n.href);
            return (
              <Link key={n.href} href={n.href} aria-current={active ? "page" : undefined}
                className="relative rounded-lg px-3 py-2 text-sm text-sub transition-colors hover:text-fg">
                {active && (
                  <motion.span layoutId="nav-pill" className="absolute inset-0 rounded-lg bg-muted" transition={{ type: "spring", stiffness: 400, damping: 32 }} />
                )}
                <span className="relative">{n.label}</span>
              </Link>
            );
          })}
        </nav>
      </div>
    </header>
  );
}
