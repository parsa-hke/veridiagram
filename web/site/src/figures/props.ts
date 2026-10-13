// Book III compositions that are more than one chart.

import { get, n } from "../data";
import { fit, svg, text, type Diagram } from "../svg";
import { hbarChart, type Group } from "./charts";
import { highlight, inkDiagram } from "./geometry";

/** Proposition V: the same question asked in colour and in letters, then the three scores. */
export function colourVsLetters(): string {
  const h = get("figures:hero");
  const g3 = "results:experiments.%s.units.geometry3k_test.geometric";
  const groups: Group[] = [{
    label: "Geometry3K test",
    bars: [
      { cls: "bar-ink", label: "colour, fine-tuned", base: g3.replace("%s", "finetune") },
      { cls: "bar-red", label: "letters, fine-tuned", base: g3.replace("%s", "ablation_nooverlay") },
      { cls: "bar-blue", label: "colour, frozen probe", base: g3.replace("%s", "probe") },
    ].filter((b) => get(`${b.base}.mean`) !== undefined),
  }];
  const legendItems = [
    { cls: "bar-ink", exp: "finetune", label: "colour, fine-tuned" },
    { cls: "bar-red", exp: "ablation_nooverlay", label: "letters, fine-tuned" },
    { cls: "bar-blue", exp: "probe", label: "colour, frozen probe" },
  ].map((i) => ({ cls: i.cls, html: `${i.label} <span class="seeds">(${n(`results:experiments.${i.exp}.n_seeds`, "seeds")})</span>` }));
  const chart = hbarChart(groups, legendItems, "Macro-F1 on Geometry3K test when the question is asked in colour, in letters, and in colour to a frozen probe");
  if (!h) return chart;
  const d: Diagram = h.diagram;
  const [a, b] = h.query.refs as [string, string, string][];
  const f = fit(Object.values(d.points), d.circles, 220, 200, 14);
  const colour = svg(220, 200, inkDiagram(d, f, 0) + highlight(d, f, h.query.refs, 3, 0.8), "The question asked in colour: two segments painted green");
  const letters = svg(220, 230, `<g>${inkDiagram(d, f, 0, { labels: true })}</g>` +
    text([110, 222], `${a[1]}${a[2]} ∥ ${b[1]}${b[2]} ?`, "token-query", 3), "The question asked in letters: the point names as tokens");
  return `
<figure class="construction-figure wide-figure">
  <div class="query-grid two">
    <div class="query-panel">${colour}<p class="query-caption"><span class="smallcaps">in colour</span></p></div>
    <div class="query-panel">${letters}<p class="query-caption"><span class="smallcaps">in letters</span></p></div>
  </div>
</figure>
${chart}`;
}
