import Papa from "papaparse";
import { type ColType, type Dataset, type Row, isMissing } from "./core";

const MAX_BYTES = 60 * 1024 * 1024;
const MAX_ROWS = 100_000;
const MAX_IMAGES = 1500;
const IMG = 16; // images are downsampled to 16x16 grayscale -> 256 numeric features

const isNum = (v: unknown) =>
  typeof v === "number" ? Number.isFinite(v) : typeof v === "string" && v.trim() !== "" && Number.isFinite(Number(v));
const isBool = (v: unknown) =>
  typeof v === "boolean" || (typeof v === "string" && ["true", "false"].includes(v.trim().toLowerCase()));

function inferType(vals: unknown[]): ColType {
  const v = vals.filter((x) => !isMissing(x));
  if (!v.length) return "category";
  if (v.every(isBool)) return "boolean";
  if (v.every(isNum)) return "number";
  const strs = v.map(String);
  const uniq = new Set(strs).size;
  const avg = strs.reduce((s, x) => s + x.length, 0) / strs.length;
  return avg > 25 && uniq > strs.length * 0.5 ? "text" : "category";
}

// Builds a clean typed Dataset from loosely-typed parsed rows.
export function buildDataset(name: string, columns: string[], raw: Record<string, unknown>[], note?: string): Dataset {
  // header names become keys: strip prototype-pollution names, dedupe empties
  const safe = columns.map((c, i) => {
    const s = String(c ?? "").trim() || `col_${i + 1}`;
    return ["__proto__", "constructor", "prototype"].includes(s) ? `${s}_` : s;
  });
  const types: Record<string, ColType> = {};
  safe.forEach((c, i) => (types[c] = inferType(raw.map((r) => r[columns[i]]))));
  const rows: Row[] = raw.map((r) => {
    const o: Row = {};
    safe.forEach((c, i) => {
      const v = r[columns[i]];
      if (isMissing(v)) o[c] = null;
      else if (types[c] === "number") o[c] = Number(v);
      else if (types[c] === "boolean") o[c] = String(v).toLowerCase() === "true";
      else o[c] = typeof v === "object" ? JSON.stringify(v) : String(v);
    });
    return o;
  });
  return { name, columns: safe, types, rows, note };
}

function fromObjects(name: string, objs: Record<string, unknown>[], note?: string) {
  const cols: string[] = [];
  const seen = new Set<string>();
  for (const o of objs.slice(0, 500))
    for (const k of Object.keys(o)) if (!seen.has(k)) (seen.add(k), cols.push(k));
  return buildDataset(name, cols, objs, note);
}

function parseJson(name: string, text: string): Dataset {
  let j: unknown = JSON.parse(text);
  if (j && typeof j === "object" && !Array.isArray(j)) {
    const o = j as Record<string, unknown>;
    const arrs = Object.entries(o).filter(([, v]) => Array.isArray(v));
    if (arrs.length === 1 && Object.keys(o).length === 1) j = arrs[0][1];
    else if (arrs.length && arrs.length === Object.keys(o).length) {
      // columnar: {a:[..], b:[..]}
      const n = Math.max(...arrs.map(([, v]) => (v as unknown[]).length));
      j = Array.from({ length: n }, (_, i) => Object.fromEntries(arrs.map(([k, v]) => [k, (v as unknown[])[i]])));
    } else j = [o];
  }
  if (!Array.isArray(j) || !j.length) throw new Error("JSON must contain an array of records");
  if (Array.isArray(j[0])) {
    const [head, ...rest] = j as unknown[][];
    return buildDataset(name, head.map(String), rest.map((r) => Object.fromEntries(head.map((h, i) => [String(h), r[i]]))));
  }
  return fromObjects(name, j.map((x) => (x && typeof x === "object" ? (x as Record<string, unknown>) : { value: x })));
}

async function parseImages(files: File[]): Promise<Dataset> {
  const use = files.slice(0, MAX_IMAGES);
  const canvas = document.createElement("canvas");
  canvas.width = canvas.height = IMG;
  const ctx = canvas.getContext("2d", { willReadFrequently: true })!;
  const cols = ["label", ...Array.from({ length: IMG * IMG }, (_, i) => `p${i}`)];
  const raw: Record<string, unknown>[] = [];
  for (const f of use) {
    const bmp = await createImageBitmap(f);
    ctx.clearRect(0, 0, IMG, IMG);
    ctx.drawImage(bmp, 0, 0, IMG, IMG);
    bmp.close();
    const d = ctx.getImageData(0, 0, IMG, IMG).data;
    const rel = (f as File & { webkitRelativePath?: string }).webkitRelativePath;
    const label = rel && rel.includes("/")
      ? rel.split("/").slice(-2, -1)[0]
      : f.name.replace(/\.[^.]+$/, "").split(/[_\-\s]/)[0].replace(/\d+$/, "") || "image";
    const r: Record<string, unknown> = { label };
    for (let i = 0; i < IMG * IMG; i++) r[`p${i}`] = +(((d[i * 4] + d[i * 4 + 1] + d[i * 4 + 2]) / 3 / 255).toFixed(3));
    raw.push(r);
  }
  const note = `${use.length} images downsampled to ${IMG}x${IMG} grayscale. Label = folder name or filename prefix (cat_1.png -> "cat").` +
    (files.length > use.length ? ` Only the first ${MAX_IMAGES} were used.` : "");
  return buildDataset("images", cols, raw, note);
}

export async function parseFiles(files: File[]): Promise<Dataset> {
  if (!files.length) throw new Error("No file selected");
  if (files.some((f) => f.size > MAX_BYTES)) throw new Error("File too large (limit 60 MB)");
  if (files.every((f) => f.type.startsWith("image/"))) return parseImages(files);

  const f = files[0];
  const ext = f.name.split(".").pop()?.toLowerCase() ?? "";
  const name = f.name;
  let ds: Dataset;
  if (ext === "json") ds = parseJson(name, await f.text());
  else if (ext === "xlsx" || ext === "xls") {
    const XLSX = await import("xlsx");
    const wb = XLSX.read(await f.arrayBuffer(), { type: "array" });
    const rows = XLSX.utils.sheet_to_json<Record<string, unknown>>(wb.Sheets[wb.SheetNames[0]], { defval: null });
    if (!rows.length) throw new Error("Spreadsheet is empty");
    ds = fromObjects(name, rows, `Sheet "${wb.SheetNames[0]}"`);
  } else {
    const text = await f.text();
    const first = text.split(/\r?\n/, 1)[0] ?? "";
    const delimited = /[,;\t|]/.test(first) && ext !== "txt" ? true : /[,;\t|]/.test(first) && text.split(/\r?\n/).length > 2;
    if (!delimited) {
      // plain text: one document per non-empty line; optional "label<TAB>text"
      const lines = text.split(/\r?\n/).map((l) => l.trim()).filter(Boolean);
      const tabbed = lines.every((l) => l.includes("\t"));
      ds = tabbed
        ? buildDataset(name, ["label", "text"], lines.map((l) => { const [a, ...b] = l.split("\t"); return { label: a, text: b.join(" ") }; }), "Text: 'label<TAB>text' per line")
        : buildDataset(name, ["text"], lines.map((l) => ({ text: l })), "Text: one document per line");
    } else {
      const p = Papa.parse<Record<string, unknown>>(text, { header: true, skipEmptyLines: "greedy" });
      if (!p.meta.fields?.length) throw new Error("Could not read columns");
      ds = buildDataset(name, p.meta.fields, p.data);
    }
  }
  if (ds.rows.length > MAX_ROWS) {
    ds = { ...ds, rows: ds.rows.slice(0, MAX_ROWS), note: `Truncated to the first ${MAX_ROWS.toLocaleString()} rows for browser memory.` };
  }
  if (!ds.rows.length) throw new Error("No rows found");
  return ds;
}
