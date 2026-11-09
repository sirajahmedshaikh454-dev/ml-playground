"use client";
import { FileUp, FolderOpen, Loader2, Sparkles, Table2 } from "lucide-react";
import { useRef, useState } from "react";
import { parseFiles } from "@/lib/ingest";
import { SAMPLES } from "@/lib/samples";
import { useLab } from "@/lib/store";
import type { Dataset } from "@/lib/core";
import { Badge, Button, Card, cx } from "../ui";

const TYPE_TONE = { number: "blue", category: "amber", text: "green", boolean: "default" } as const;

export function Preview({ ds, max = 40 }: { ds: Dataset; max?: number }) {
  const cols = ds.columns.slice(0, 30);
  return (
    <div className="scroll-thin max-h-96 overflow-auto rounded-xl border border-border">
      <table className="num w-full text-left text-xs">
        <caption className="sr-only">First {max} rows of {ds.name}</caption>
        <thead className="sticky top-0 bg-surface">
          <tr>
            {cols.map((c) => (
              <th key={c} scope="col" className="border-b border-border px-3 py-2 font-medium whitespace-nowrap text-sub">
                {c} <Badge tone={TYPE_TONE[ds.types[c]]}>{ds.types[c]}</Badge>
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {ds.rows.slice(0, max).map((r, i) => (
            <tr key={i} className="odd:bg-surface/40 hover:bg-muted/60">
              {cols.map((c) => (
                <td key={c} className={cx("max-w-56 truncate px-3 py-1.5", r[c] == null && "text-bad/80 italic")}>
                  {r[c] == null ? "missing" : String(r[c])}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export function DataStep() {
  const { raw, load } = useLab();
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const [drag, setDrag] = useState(false);
  const fileIn = useRef<HTMLInputElement>(null);
  const dirIn = useRef<HTMLInputElement>(null);

  const ingest = async (files: File[]) => {
    setBusy(true); setErr(null);
    try { load(await parseFiles(files)); } catch (e) { setErr((e as Error).message); } finally { setBusy(false); }
  };

  return (
    <div className="space-y-5">
      <Card title="Bring your data" icon={<FileUp className="size-4" />}>
        <div
          onDragOver={(e) => { e.preventDefault(); setDrag(true); }}
          onDragLeave={() => setDrag(false)}
          onDrop={(e) => { e.preventDefault(); setDrag(false); void ingest([...e.dataTransfer.files]); }}
          className={cx("grid-bg flex flex-col items-center justify-center gap-3 rounded-xl border-2 border-dashed px-4 py-10 text-center transition-colors",
            drag ? "border-accent bg-accent/5" : "border-border")}>
          {busy ? <Loader2 className="size-8 animate-spin text-accent" aria-hidden /> : <FileUp className="size-8 text-primary" aria-hidden />}
          <p className="text-base font-medium">Drop a file here, or pick one</p>
          <p className="max-w-lg text-sm text-dim">CSV, TSV, JSON, Excel (.xlsx/.xls), plain text, or a set of images (labels from folder or filename prefix). Files are parsed in your browser and never uploaded.</p>
          <div className="mt-1 flex flex-wrap justify-center gap-2">
            <Button variant="accent" onClick={() => fileIn.current?.click()} disabled={busy}><FileUp className="size-4" aria-hidden />Choose file(s)</Button>
            <Button onClick={() => dirIn.current?.click()} disabled={busy}><FolderOpen className="size-4" aria-hidden />Image folder</Button>
          </div>
          <input ref={fileIn} type="file" multiple hidden accept=".csv,.tsv,.txt,.json,.xlsx,.xls,image/*"
            onChange={(e) => { void ingest([...(e.target.files ?? [])]); e.target.value = ""; }} />
          <input ref={dirIn} type="file" multiple hidden {...({ webkitdirectory: "" } as object)}
            onChange={(e) => { void ingest([...(e.target.files ?? [])].filter((f) => f.type.startsWith("image/"))); e.target.value = ""; }} />
        </div>
        {err && <p role="alert" className="mt-3 rounded-lg border border-bad/40 bg-bad/10 px-3 py-2 text-sm text-bad">Could not load: {err}</p>}
      </Card>

      <Card title="Or start from a sample" icon={<Sparkles className="size-4" />}>
        <div className="grid gap-3 sm:grid-cols-3">
          {SAMPLES.map((s) => (
            <button key={s.id} onClick={() => load(s.make())}
              className={cx("group cursor-pointer rounded-xl border bg-surface/60 p-4 text-left transition-[border,transform] duration-200 hover:-translate-y-0.5 hover:border-primary/60",
                raw?.name === s.make().name ? "border-accent" : "border-border")}>
              <div className="font-medium group-hover:text-primary">{s.title}</div>
              <div className="mt-1 text-xs text-dim">{s.blurb}</div>
            </button>
          ))}
        </div>
      </Card>

      {raw && (
        <Card title={raw.name} icon={<Table2 className="size-4" />}
          actions={<><Badge tone="blue">{raw.rows.length.toLocaleString()} rows</Badge><Badge tone="amber">{raw.columns.length} columns</Badge></>}>
          {raw.note && <p className="mb-3 text-sm text-dim">{raw.note}</p>}
          <Preview ds={raw} />
        </Card>
      )}
    </div>
  );
}
