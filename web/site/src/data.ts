// The four JSON documents written by `python -m web.export`, and the only way the page shows a number:
// n("results:experiments.finetune.units.geometry3k_test.geometric.mean") returns a <span class="num">
// whose data-src names the JSON field it came from (WEBSITE_PLAN.md §5: nothing typed by hand).

export const DOC_NAMES = ["results", "scaling", "pseudo_labels", "facts"] as const;
export type DocName = (typeof DOC_NAMES)[number];
// eslint-disable-next-line @typescript-eslint/no-explicit-any
export type Json = any;
export type Docs = Record<DocName, Json>;

let DOCS: Docs | null = null;

export function setDocs(d: Docs): void {
  DOCS = d;
}

export async function loadDocs(base = "data/"): Promise<Docs> {
  const entries = await Promise.all(
    DOC_NAMES.map(async (name) => {
      const r = await fetch(`${base}${name}.json`);
      if (!r.ok) throw new Error(`${name}.json: HTTP ${r.status} (run python -m web.export)`);
      return [name, await r.json()] as const;
    }),
  );
  return Object.fromEntries(entries) as Docs;
}

/** Value at "doc:dotted.path" (keys may contain anything but dots), or undefined. */
export function get(src: string): Json {
  if (!DOCS) throw new Error("data not loaded");
  const [doc, path = ""] = src.split(":", 2) as [DocName, string];
  let v: Json = DOCS[doc];
  for (const k of path ? path.split(".") : []) {
    if (v === null || v === undefined) return undefined;
    v = v[k];
  }
  return v;
}

export type Fmt =
  | "f2" | "f3" | "f1"   // fixed decimals: 0.867
  | "int"                // 1,296
  | "pct"                // fraction -> percent, no decimals: 0.882 -> 88%
  | "pct1"               // fraction -> percent, one decimal: 0.882 -> 88.2%
  | "pctv"               // value already a percent, up to two decimals: 89.0 -> 89%, 7.94 -> 7.94%
  | "sci"                // 2e-5 -> 2 × 10⁻⁵
  | "deg"                // 3.7 -> 3.7°
  | "seeds"              // seed count -> "3 seeds" / "one seed"
  | "raw";

const WORDS = ["no", "one", "two", "three", "four", "five"];

export function format(v: Json, fmt: Fmt): string {
  if (v === null || v === undefined || (typeof v === "number" && !Number.isFinite(v))) return "—";
  switch (fmt) {
    case "f1": return Number(v).toFixed(1);
    case "f2": return Number(v).toFixed(2);
    case "f3": return Number(v).toFixed(3);
    case "int": return Math.round(Number(v)).toLocaleString("en-GB");
    case "pct": return `${Math.round(100 * Number(v))}%`;
    case "pct1": return `${(100 * Number(v)).toFixed(1)}%`;
    case "pctv": return `${Number(v).toLocaleString("en-GB", { maximumFractionDigits: 2 })}%`;
    case "sci": {
      const [m, e] = Number(v).toExponential().split("e");
      const sup = (x: string) => x.replace(/-/g, "⁻").replace(/\d/g, (d) => "⁰¹²³⁴⁵⁶⁷⁸⁹"[Number(d)]);
      return `${Number(m).toLocaleString("en-GB", { maximumFractionDigits: 2 })} × 10${sup(String(Number(e)))}`;
    }
    case "deg": return `${v}°`;
    case "seeds": {
      const k = Number(v);
      return `${k < WORDS.length ? WORDS[k] : k} seed${k === 1 ? "" : "s"}`;
    }
    default: return String(v);
  }
}

const esc = (s: string) => s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/"/g, "&quot;");

/** A traced number: <span class="num" data-src="doc:path">formatted</span>. */
export function n(src: string, fmt: Fmt = "f3"): string {
  const v = get(src);
  const missing = v === null || v === undefined;
  return `<span class="num${missing ? " missing" : ""}" data-src="${esc(src)}" title="${esc(src)}">${esc(format(v, fmt))}</span>`;
}

/** Literal text that contains digits but is not a measurement (a year, a plan section). */
export const lit = (s: string) => `<span class="lit">${esc(s)}</span>`;

// ------------------------------------------------------------------ results helpers

export const UNIT_NAMES: Record<string, string> = {
  val: "validation",
  geometry3k_test: "Geometry3K test",
  pgdp5k_heldout: "PGDP5K held-out source",
  synth_extrap: "synthetic extrapolation",
};
export const TEST_UNITS = ["geometry3k_test", "pgdp5k_heldout", "synth_extrap"];

export const cell = (exp: string, unit: string, ls = "geometric") => `results:experiments.${exp}.units.${unit}.${ls}`;

/** "0.867 ± 0.007" (sd only when there is more than one seed). */
export function ms(exp: string, unit: string, ls = "geometric"): string {
  const base = cell(exp, unit, ls);
  const sd = get(`${base}.sd`);
  return n(`${base}.mean`) + (sd === null || sd === undefined ? "" : ` ± ${n(`${base}.sd`)}`);
}

export const seeds = (exp: string) => n(`results:experiments.${exp}.n_seeds`, "seeds");

export const hasExp = (exp: string) => get(`results:experiments.${exp}`) !== undefined;
