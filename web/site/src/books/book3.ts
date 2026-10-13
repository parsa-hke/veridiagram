import { TEST_UNITS, UNIT_NAMES, cell, get, hasExp, ms, n, seeds } from "../data";
import { experimentChart, hbarChart, scalingChart } from "../figures/charts";
import { colourVsLetters } from "../figures/props";

const UNITS = TEST_UNITS.map((key) => ({ key, label: UNIT_NAMES[key] }));
const macro = (exp: string, unit: string) => cell(exp, unit);
const numbers = (inner: string) => `<details class="numbers"><summary>The numbers, with seeds</summary>${inner}</details>`;

const mean = (exp: string, unit: string, ls = "geometric") => get(`${cell(exp, unit, ls)}.mean`) as number | undefined;

/** "below" / "above" / "level with", decided by the data, not by the text. */
function cmp(a: number | undefined, b: number | undefined, below = "below", above = "above"): string {
  if (a === undefined || b === undefined) return "compared with";
  return a < b ? below : a > b ? above : "level with";
}

function table(rows: { label: string; exp: string }[], units = TEST_UNITS, ls = "geometric"): string {
  const head = units.map((u) => `<th>${UNIT_NAMES[u]}</th>`).join("");
  const body = rows
    .filter((r) => hasExp(r.exp))
    .map((r) => `<tr><th scope="row">${r.label} <span class="seeds">(${seeds(r.exp)})</span></th>${units
      .map((u) => `<td data-label="${UNIT_NAMES[u]}"><span class="cell">${ms(r.exp, u, ls)}</span></td>`).join("")}</tr>`)
    .join("");
  return `<div class="table-wrap"><table class="results"><thead><tr><th></th>${head}</tr></thead><tbody>${body}</tbody></table></div>`;
}

function prop(no: string, id: string, statement: string, body: string): string {
  return `<article class="prop" id="${id}">
    <h3><span class="prop-no">Proposition ${no}.</span> ${statement}</h3>
    ${body}
    <p class="qed">Q.E.D.</p>
  </article>`;
}

function scalingTable(): string {
  const pts = (get("scaling:points") ?? []) as { unit: string; synthetic: number }[];
  const sizes = (get("scaling:sizes") ?? []) as number[];
  const units = (get("scaling:units") ?? []) as string[];
  const at = (u: string, s: number) => pts.findIndex((p) => p.unit === u && p.synthetic === s);
  const sizeIdx = (s: number) => sizes.indexOf(s);
  const rows = sizes.map((s) => {
    const cells = units.map((u) => {
      const i = at(u, s);
      if (i < 0) return `<td data-label="${UNIT_NAMES[u]}">—</td>`;
      const sd = get(`scaling:points.${i}.sd`);
      return `<td data-label="${UNIT_NAMES[u]}"><span class="cell">${n(`scaling:points.${i}.mean`)}${sd == null ? "" : ` ± ${n(`scaling:points.${i}.sd`)}`}
        <span class="seeds">(${n(`scaling:points.${i}.n_seeds`, "seeds")})</span></span></td>`;
    });
    return `<tr><th scope="row">${n(`scaling:sizes.${sizeIdx(s)}`, "int")}<span class="row-unit"> synthetic diagrams</span></th>${cells.join("")}</tr>`;
  });
  return `<div class="table-wrap"><table class="results"><thead><tr><th>synthetic diagrams</th>${units
    .map((u) => `<th>${UNIT_NAMES[u]}</th>`).join("")}</tr></thead><tbody>${rows.join("")}</tbody></table></div>`;
}

function teacherCell(unit: string): string {
  const teacher = String(get("pseudo_labels:teacher") ?? "");
  const m = teacher.match(/^(.*)_s(\d+)$/);
  if (!m) return "—";
  const [, exp, seed] = m;
  const i = ((get(`${cell(exp, unit)}.seeds`) ?? []) as number[]).indexOf(Number(seed));
  return i < 0 ? "—" : n(`${cell(exp, unit)}.values.${i}`);
}

function perRelationTable(exps: { label: string; exp: string }[], unit: string): string {
  const rels = get("results:relations") as string[];
  const rows = exps
    .filter((e) => hasExp(e.exp))
    .map((e) => `<tr><th scope="row">${e.label} <span class="seeds">(${seeds(e.exp)})</span></th>${rels
      .map((r) => `<td data-label="${r.replace("_", " ")}"><span class="cell">${n(`${cell(e.exp, unit)}.per_relation.${r}.mean`)}</span></td>`).join("")}</tr>`);
  return `<div class="table-wrap wide"><table class="results"><thead><tr><th>${UNIT_NAMES[unit]}</th>${rels
    .map((r) => `<th><code>${r.replace("_", "_<wbr>")}</code></th>`).join("")}</tr></thead><tbody>${rows.join("")}</tbody></table></div>`;
}

function precisionChart(): string {
  if (!get("pseudo_labels:available")) return "";
  const f = "pseudo_labels:filters";
  return hbarChart([{
    label: "precision of the pseudo-positives",
    bars: [
      { cls: "bar-ink", label: "kept by confidence", base: `${f}.confidence.all.pos_precision`, valueOnly: true },
      { cls: "bar-blue", label: "kept by confidence + consistency", base: `${f}.consistency.all.pos_precision`, valueOnly: true },
      { cls: "bar-red", label: "removed by consistency", base: `${f}.removed_by_consistency.all.pos_precision`, valueOnly: true },
    ],
  }], [
    { cls: "bar-ink", html: `kept by confidence (${n(`${f}.confidence.all.pseudo_pos`, "int")})` },
    { cls: "bar-blue", html: `kept by confidence + consistency (${n(`${f}.consistency.all.pseudo_pos`, "int")})` },
    { cls: "bar-red", html: `removed by consistency (${n(`${f}.removed_by_consistency.all.pseudo_pos`, "int")})` },
  ], "Precision of the pseudo-positive labels kept by each filter and removed by the consistency check",
  "Checked after the fact against the coordinates of the unlabelled pool; counts are pseudo-positive labels.");
}

function relationChart(demo: string, unit: string): string {
  const rels = get("results:relations") as string[];
  const rows = [{ exp: "finetune", label: "fine-tune (DINOv3-L)", cls: "bar-ink" }, { exp: demo, label: "DINOv2-L (the demo)", cls: "bar-blue" }]
    .filter((r) => hasExp(r.exp));
  const groups = rels.map((r) => ({
    label: r.replace("_", " "),
    bars: rows.map((x) => ({ cls: x.cls, label: x.label, base: `${cell(x.exp, unit)}.per_relation.${r}` })),
  }));
  return hbarChart(groups, rows.map((x) => ({ cls: x.cls, html: `${x.label} <span class="seeds">(${seeds(x.exp)})</span>` })),
    `Per-relation F1 on ${UNIT_NAMES[unit]}`);
}

function weakest(exp: string, unit: string): string {
  const rels = get("results:relations") as string[];
  const f = (r: string) => get(`${cell(exp, unit)}.per_relation.${r}.mean`) as number;
  return rels.reduce((a, b) => (f(b) < f(a) ? b : a));
}

export function book3(): string {
  const g3 = "geometry3k_test";
  const demo = String(get("facts:meta.demo_run") ?? "").replace(/_s\d+$/, "");
  const weak = weakest(demo, g3);
  const realUnits = ["geometry3k_test", "pgdp5k_heldout"];
  const controlBeatsBoth = realUnits.every(
    (u) => (mean("selftrain_control", u) ?? 0) > Math.max(mean("selftrain_confidence", u) ?? 1, mean("selftrain_consistency", u) ?? 1),
  );
  const consVsConf = realUnits.every((u) => (mean("selftrain_consistency", u) ?? 0) > (mean("selftrain_confidence", u) ?? 1));
  const sd = (exp: string, u: string) => (get(`${cell(exp, u)}.sd`) as number | null) ?? 0;
  const withinTwoSd = realUnits.every((u) => Math.abs((mean("selftrain_consistency", u) ?? 0) - (mean("selftrain_confidence", u) ?? 0))
    < 2 * Math.max(sd("selftrain_consistency", u), sd("selftrain_confidence", u)));

  return `
<section class="book" id="book-3" data-running="Book III · The Propositions">
  <h2><span class="book-no">Book III</span> The Propositions</h2>
  <p class="opening">Each finding is one proposition. Scores are macro-F1 against the geometric labels at a threshold of
  ${n("results:threshold", "f1")}, mean ± standard deviation over seeds; single seeds are labelled as such.</p>

  ${prop("IV", "prop-4", "Fine-tuning reads geometry; a frozen eye does not.", `
    <p>A linear head on frozen features (the probe) against the same model fine-tuned end to end.</p>
    ${experimentChart([{ exp: "finetune", label: "fine-tune", cls: "bar-ink" }, { exp: "probe", label: "linear probe", cls: "bar-red" }],
      UNITS, "Macro-F1 of the linear probe and the fine-tune on three test sets", macro)}
    ${numbers(table([{ label: "linear probe", exp: "probe" }, { label: "fine-tune", exp: "finetune" }]))}`)}

  ${prop("V", "prop-5", "That colour speaketh louder than letters.", `
    <p>Ask the same question with letters instead of colour (“is AB parallel to CD?”, the point names
    given as tokens) and macro-F1 on Geometry3K test falls from ${ms("finetune", g3)} to
    ${ms("ablation_nooverlay", g3)} (${seeds("ablation_nooverlay")}), ${cmp(mean("ablation_nooverlay", g3), mean("probe", g3))}
    the frozen probe that is shown the colour (${ms("probe", g3)}).</p>
    ${colourVsLetters()}
    ${numbers(table([{ label: "colour, fine-tuned", exp: "finetune" }, { label: "letters", exp: "ablation_nooverlay" },
      { label: "colour, frozen probe", exp: "probe" }]))}`)}

  ${prop("VI", "prop-6", "A little synthetic geometry goes a long way, and then no further.", `
    <p>Half of every training batch drawn from verifier-labelled synthetic diagrams, at a fixed budget of
    ${n("scaling:budget.total_images", "int")} images:</p>
    ${scalingChart()}
    ${numbers(scalingTable())}`)}

  ${prop("VII", "prop-7", "The older teacher saw better.", `
    <p>The same recipe with DINOv2-L instead of DINOv3-L. DINOv2 was run with ${seeds("ablation_dinov2")},
    so this is a hint, not a law.</p>
    ${experimentChart([{ exp: "finetune", label: "DINOv3-L", cls: "bar-ink" }, { exp: "ablation_dinov2", label: "DINOv2-L", cls: "bar-blue" }],
      UNITS, "Macro-F1 of the DINOv3-L and DINOv2-L fine-tunes on three test sets", macro)}
    ${numbers(table([{ label: "DINOv3-L", exp: "finetune" }, { label: "DINOv2-L", exp: "ablation_dinov2" }]))}`)}

  ${prop("VIII", "prop-8", "Of resolution and augmentation.", `
    ${experimentChart([{ exp: "finetune", label: "full resolution, augmented", cls: "bar-ink" },
      { exp: "ablation_res224", label: "half resolution", cls: "bar-red" }, { exp: "ablation_noaug", label: "no augmentation", cls: "bar-blue" }],
      UNITS, "Macro-F1 at full and half resolution, and without augmentation, on three test sets", macro)}
    ${numbers(table([{ label: "full resolution, augmented", exp: "finetune" }, { label: "half resolution", exp: "ablation_res224" },
      { label: "no augmentation", exp: "ablation_noaug" }]))}`)}

  ${prop("IX", "prop-9", "Consistency catches wrong labels, but more labels did not help.", `
    <p>The teacher labelled ${n("pseudo_labels:pool_diagrams", "int")} unlabelled diagrams. The consistency check
    flagged ${n("pseudo_labels:contradiction_diagrams", "int")} of them, and it removed the right ones: the
    precision of the pseudo-positives rose from ${n("pseudo_labels:filters.confidence.all.pos_precision")}
    (confidence only) to ${n("pseudo_labels:filters.consistency.all.pos_precision")}, and the entries it removed
    had precision ${n("pseudo_labels:filters.removed_by_consistency.all.pos_precision")}.</p>
    <p>${controlBeatsBoth
      ? "But a control trained the same way without any pseudo-labels beat both filtered students on the real test sets."
      : "Against a control trained the same way without pseudo-labels:"}
    ${consVsConf ? (withinTwoSd
      ? "Consistency edged out confidence on the real test sets, by less than two standard deviations."
      : "Consistency beat confidence on the real test sets.") : ""}
    The control has ${seeds("selftrain_control")} so far.</p>
    ${precisionChart()}
    ${experimentChart([{ exp: "selftrain_control", label: "control, no pseudo-labels", cls: "bar-ink" },
      { exp: "selftrain_confidence", label: "confidence filter", cls: "bar-red" },
      { exp: "selftrain_consistency", label: "confidence + consistency", cls: "bar-blue" }],
      UNITS, "Macro-F1 of the two self-trained students and the control on three test sets", macro)}
    ${numbers(table([{ label: "confidence filter", exp: "selftrain_confidence" },
      { label: "confidence + consistency", exp: "selftrain_consistency" },
      { label: "control, no pseudo-labels", exp: "selftrain_control" }]))}
    <p>The teacher itself (<code>${String(get("pseudo_labels:teacher") ?? "")}</code>): ${TEST_UNITS
      .map((u) => `${UNIT_NAMES[u]} ${teacherCell(u)}`).join("; ")}.</p>`)}

  ${prop("X", "prop-10", "Where it fails.", `
    <p>Per-relation F1 on Geometry3K test. The weak relation is <code>${weak}</code>: telling two lengths apart
    within ${n("facts:tolerances.equal_length_pct", "pctv")} by eye is hard.</p>
    ${relationChart(demo, g3)}
    ${numbers(perRelationTable([{ label: "fine-tune (DINOv3-L)", exp: "finetune" }, { label: "DINOv2-L (the demo)", exp: demo }], g3))}`)}
</section>`;
}
