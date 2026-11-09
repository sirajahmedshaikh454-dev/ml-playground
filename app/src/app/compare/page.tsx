import type { Metadata } from "next";
import { Compare } from "@/components/Compare";

export const metadata: Metadata = { title: "Experiments · ML Playground" };

export default function ComparePage() {
  return <Compare />;
}
