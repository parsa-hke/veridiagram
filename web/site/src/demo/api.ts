// The inference API (web/server/app.py), behind one base URL (WEBSITE_PLAN.md §4.3): facts.meta.api_url,
// or the page's own origin when it is null (the Space serves the site and the API together).

import { get } from "../data";

export type Relation = "parallel" | "perpendicular" | "equal_length" | "on_circle" | "tangent" | "collinear";
export type Ref = [string, ...(string | number)[]];

export interface Figure {
  points: Record<string, [number, number]>;
  segments: [string, string][];
  circles: { center?: string | null; through?: string; center_xy?: [number, number]; radius?: number }[];
}

export interface RulerEntry {
  holds: boolean;
  margin: number | null;
  unit: "deg" | "pct";
  tolerance: number;
  measured: number | null;
}

export interface Result {
  kind: string;
  refs: Ref[];
  applicable: Relation[];
  probs: Partial<Record<Relation, number>>;
  ruler: { basis: "construction" | "clicks" } & Partial<Record<Relation, RulerEntry>>;
  model_input?: string;
}

export interface Statement {
  relation: Relation;
  kind: string;
  refs: Ref[];
  prob?: number;
}

export interface Summary {
  relations: Statement[];
  ruler_relations: Statement[];
  contradictions: { kind: string; fact?: string[]; class?: string[] }[];
  closure_tau: number;
}

export type Body = { construction: Figure } | { image: string; primitives: Figure };

export type Status = "checking" | "warming" | "ready" | "offline";

export class ApiError extends Error {
  constructor(message: string, readonly status: number) {
    super(message);
  }
}

export function apiBase(): string {
  const configured = get("facts:meta.api_url") as string | null | undefined;
  return configured ? configured.replace(/\/$/, "") + "/" : "";
}

async function call<T>(path: string, body?: unknown, timeoutMs = 300_000): Promise<T> {
  const ctl = new AbortController();
  const t = setTimeout(() => ctl.abort(), timeoutMs);
  try {
    const r = await fetch(`${apiBase()}api/${path}`, {
      method: body === undefined ? "GET" : "POST",
      headers: body === undefined ? undefined : { "content-type": "application/json" },
      body: body === undefined ? undefined : JSON.stringify(body),
      signal: ctl.signal,
    });
    const data = await r.json().catch(() => ({}));
    if (!r.ok) throw new ApiError((data as { error?: string }).error ?? `HTTP ${r.status}`, r.status);
    return data as T;
  } catch (e) {
    if (e instanceof ApiError) throw e;
    throw new ApiError(ctl.signal.aborted ? "the instrument took too long to answer" : "the instrument cannot be reached", 0);
  } finally {
    clearTimeout(t);
  }
}

export interface Health {
  ok: boolean;
  ready: boolean;
  model: string | null;
  device: string | null;
}

export const health = () => call<Health>("health", undefined, 8000);

export const ask = (body: Body, queries: { kind: string; refs: Ref[] }[]) =>
  call<{ results: Result[]; model: string; ms: number }>("ask", { ...body, queries });

export const readChunk = (body: Body, offset: number, limit: number) =>
  call<{ results: Result[]; queries_total: number; queries_asked: number; ms: number }>("read", { ...body, offset, limit });

export const closure = (body: Body, answers: { kind: string; refs: Ref[]; probs: Result["probs"] }[]) =>
  call<Summary>("closure", { ...body, answers });

/** Ping /api/health until the model is ready; report every change of status. A sleeping Space answers
 * 502/503 (or not at all) for a minute or two while it wakes: that is "warming". A 404 means there is no
 * API behind this page (a static host): "offline" at once. After two minutes of silence: "offline",
 * still asked every 30 s in case it wakes. */
export function watchHealth(onChange: (s: Status, h?: Health) => void, everyMs = 5000, patienceMs = 120_000): () => void {
  let stopped = false;
  const t0 = Date.now();
  const tick = async () => {
    if (stopped) return;
    let next = everyMs;
    try {
      const h = await health();
      if (h.ready) {
        onChange("ready", h);
        return;                                   // ready: stop polling
      }
      onChange(h.ok ? "warming" : "offline", h);
      if (!h.ok) return;                          // the server says the model cannot be loaded
    } catch (e) {
      if (e instanceof ApiError && e.status === 404) {
        onChange("offline");
        return;
      }
      const patient = Date.now() - t0 < patienceMs;
      onChange(patient ? "warming" : "offline");
      if (!patient) next = 30_000;
    }
    setTimeout(tick, next);
  };
  onChange("checking");
  tick();
  return () => {
    stopped = true;
  };
}
