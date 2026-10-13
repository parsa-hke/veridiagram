// The three witnesses (WEBSITE_PLAN.md §1.2): the Ruler (the coordinate verifier, exact), the Eye (the
// vision transformer, from pixels only) and what the model sees (its exact input). Numbers here come
// from the server's answer at run time, so they carry the class "live".

import type { Ref, Relation, Result, Statement, Summary } from "./api";

export const WORDS: Record<Relation, string> = {
  parallel: "parallel",
  perpendicular: "perpendicular",
  equal_length: "equal in length",
  on_circle: "on the circle",
  tangent: "tangent",
  collinear: "on the line",
};

const SYMBOL: Record<Relation, string> = {
  parallel: "∥",
  perpendicular: "⊥",
  equal_length: "=",
  on_circle: "on",
  tangent: "tangent to",
  collinear: "on",
};

const esc = (s: string) => s.replace(/&/g, "&amp;").replace(/</g, "&lt;");
const live = (s: string) => `<span class="live">${esc(s)}</span>`;

export function refName(r: Ref, circleNames: (string | null)[] = []): string {
  if (r[0] === "seg") return `${r[1]}${r[2]}`;
  if (r[0] === "pt") return String(r[1]);
  const c = circleNames[Number(r[1])];
  return c ? `circle ${c}` : `circle ${Number(r[1]) + 1}`;
}

/** "AB ∥ CD", "P on circle O", "P on AB". */
export function statement(s: { relation: Relation; kind: string; refs: Ref[] }, circles: (string | null)[] = []): string {
  const n = (r: Ref) => refName(r, circles);
  if (s.kind === "pt_line") return `${n(s.refs[0])} ${SYMBOL[s.relation]} ${n(s.refs[1])}${n(s.refs[2])}`;
  return `${n(s.refs[0])} ${SYMBOL[s.relation]} ${n(s.refs[1])}`;
}

function rulerLine(rel: Relation, r: Result): string {
  const e = r.ruler[rel];
  if (!e || e.measured === null) return `<td>—</td>`;
  const m = e.unit === "deg" ? `${e.measured.toFixed(1)}°` : `${e.measured.toFixed(1)}%`;
  const t = e.unit === "deg" ? `${e.tolerance}°` : `${e.tolerance}%`;
  const what = rel === "parallel" ? "from parallel" : rel === "perpendicular" ? "from a right angle"
    : rel === "equal_length" ? "apart" : rel === "collinear" ? "off the line, of the figure's size" : "off the radius";
  return `<td>${live(m)} ${what}; within ${live(t)}: <strong>${e.holds ? "yes" : "no"}</strong></td>`;
}

function eyeLine(rel: Relation, r: Result): string {
  const p = r.probs[rel] ?? 0;
  return `<td><span class="prob-bar" style="--p:${p}"><span></span></span> ${live(p.toFixed(2))} <strong>${p > 0.5 ? "yes" : "no"}</strong></td>`;
}

export function witnesses(r: Result, circles: (string | null)[] = []): string {
  const clicks = r.ruler.basis === "clicks";
  const rows = r.applicable.map((rel) => {
    const disagree = (r.probs[rel] ?? 0) > 0.5 !== Boolean(r.ruler[rel]?.holds);
    return `<tr class="${disagree ? "disagree" : ""}"><th scope="row">${WORDS[rel]}${disagree ? ' <span class="disagree-mark">they disagree</span>' : ""}</th>
      ${rulerLine(rel, r)}${eyeLine(rel, r)}</tr>`;
  }).join("");
  const title = r.refs.length === 3 ? `${refName(r.refs[0])} and the line ${refName(r.refs[1])}${refName(r.refs[2])}`
    : `${refName(r.refs[0], circles)} and ${refName(r.refs[1], circles)}`;
  return `
<div class="witnesses">
  <div class="witness-table">
    <p class="witness-title"><span class="smallcaps">The question</span> ${esc(title)}</p>
    <table class="results verdicts">
      <thead><tr><th></th><th>${clicks ? "The Ruler, on your clicks" : "The Ruler"}</th><th>The Eye</th></tr></thead>
      <tbody>${rows}</tbody>
    </table>
    ${clicks ? `<p class="witness-note">There is no exact Ruler for an uploaded picture: these measurements are
      of where you clicked, not of the drawing.</p>` : ""}
  </div>
  ${r.model_input ? `<figure class="model-view"><img src="data:image/png;base64,${r.model_input}" alt="The image the network received, with the question highlighted in green" width="448" height="448">
    <figcaption><span class="smallcaps">What the model sees</span>: the 448 × 448 picture the network received.</figcaption></figure>` : ""}
</div>`;
}

/** A closure entity as a reader writes it: segments "A-B" or carrier lines "S:A,B" -> "AB". */
export function entity(e: string): string {
  if (e.startsWith("S:")) return e.slice(2).split(",").join("");
  if (/^L\d+$/.test(e)) return "a drawn line";
  return e.split("-").join("");
}

const segKey = (refs: Ref[]) => refs.filter((r) => r[0] === "seg").map((r) => [r[1], r[2]].sort().join("")).sort().join("|");

/** One contradiction in words, with the Ruler's own verdict on the pair when the figure has it. */
export function contradictionText(c: Summary["contradictions"][number], results: Result[] = []): string {
  if (c.kind === "parallel_and_perpendicular") {
    return `it calls ${esc((c.class ?? []).map(entity).join(", "))} both parallel and perpendicular`;
  }
  const [rel, a, b] = (c.fact ?? []) as [Relation, string, string];
  const A = entity(a), B = entity(b);
  const text = `its other answers imply ${esc(A)} ${SYMBOL[rel] ?? rel} ${esc(B)}, which it denies`;
  const r = results.find((x) => segKey(x.refs) === [A, B].sort().join("|"));
  const e = r?.ruler[rel];
  if (!e || e.measured === null) return text;
  const m = e.unit === "deg" ? `${e.measured.toFixed(1)}°` : `${e.measured.toFixed(1)}% apart`;
  return `${text} (the Ruler: ${e.holds ? "it holds" : "it does not hold"}, ${live(m)})`;
}

/** The whole figure read: the Eye's relations next to the Ruler's, and contradictions. */
export function readout(sum: Summary, asked: number, total: number, circles: (string | null)[] = [], results: Result[] = []): string {
  const key = (s: Statement) => `${s.relation}|${JSON.stringify(s.refs)}`;
  const eye = new Map(sum.relations.map((s) => [key(s), s]));
  const ruler = new Map(sum.ruler_relations.map((s) => [key(s), s]));
  const all = [...new Map([...ruler, ...eye]).values()];
  const rows = all.map((s) => {
    const e = eye.get(key(s)), r = ruler.has(key(s));
    const cls = e && r ? "both" : e ? "eye-only" : "ruler-only";
    const note = e && r ? "both" : e ? "the Eye only" : "the Ruler only";
    return `<li class="${cls}" data-refs="${esc(JSON.stringify(s.refs)).replace(/"/g, "&quot;")}" tabindex="0"><span class="stmt">${esc(statement(s, circles))}</span>
      <span class="who">${note}${e?.prob !== undefined ? ` · ${live(e.prob.toFixed(2))}` : ""}</span></li>`;
  }).join("");
  const n = sum.contradictions.length;
  const contra = n
    ? `<div class="contradiction"><p><strong>The Eye's answers do not close</strong> in ${live(String(n))} place${n === 1 ? "" : "s"}:</p>
       <ul>${sum.contradictions.map((c) => `<li>${contradictionText(c, results)}</li>`).join("")}</ul>
       <p class="witness-note">Deduction runs on the Eye's confident answers (probability beyond ${live(String(sum.closure_tau))}).
       “Equal within a tolerance” is not exactly transitive, so a deduced equality can be one the Ruler rejects too.</p></div>`
    : `<p class="consistent">No contradictions in the Eye's confident answers (probability beyond ${live(String(sum.closure_tau))}).</p>`;
  return `
<div class="readout">
  <p><span class="smallcaps">The whole figure</span>: ${live(String(asked))} of ${live(String(total))} questions asked.</p>
  ${all.length ? `<ul class="statements">${rows}</ul>` : "<p>Neither witness finds any relation here.</p>"}
  ${contra}
</div>`;
}
