import type { Metadata } from "next";
import { Lab } from "@/components/lab/Lab";

export const metadata: Metadata = { title: "Data Lab · ML Playground" };

export default function LabPage() {
  return <Lab />;
}
