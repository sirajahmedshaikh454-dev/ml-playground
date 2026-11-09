import type { Metadata } from "next";
import { RLArena } from "@/components/rl/RLArena";

export const metadata: Metadata = { title: "RL Arena · ML Playground" };

export default function RLPage() {
  return <RLArena />;
}
