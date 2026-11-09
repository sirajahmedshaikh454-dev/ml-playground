import type { Metadata, Viewport } from "next";
import { Fira_Code, Fira_Sans } from "next/font/google";
import { Header } from "@/components/Header";
import "./globals.css";

// next/font self-hosts these at build time: no third-party requests at runtime.
const sans = Fira_Sans({ variable: "--font-fira-sans", subsets: ["latin"], weight: ["300", "400", "500", "600", "700"] });
const mono = Fira_Code({ variable: "--font-fira-code", subsets: ["latin"] });

export const metadata: Metadata = {
  title: "ML Playground",
  description: "Upload any data, clean it, and watch supervised, unsupervised and reinforcement learning models train live in your browser.",
};
export const viewport: Viewport = { themeColor: "#0b1120" };

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className={`${sans.variable} ${mono.variable} antialiased`}>
      <body className="flex min-h-dvh flex-col">
        <a href="#main" className="sr-only focus:not-sr-only focus:fixed focus:left-4 focus:top-4 focus:z-50 focus:rounded-lg focus:bg-accent focus:px-3 focus:py-2 focus:text-black">
          Skip to content
        </a>
        <Header />
        <main id="main" className="flex-1">{children}</main>
        <footer className="border-t border-border/60 px-4 py-6 text-center text-xs text-dim">
          Your data never leaves your browser. Built with Next.js, TensorFlow.js and three.js.
        </footer>
      </body>
    </html>
  );
}
