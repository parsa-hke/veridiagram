import { get, ms, n } from "../data";

const RELATION_WORDS: Record<string, string> = {
  parallel: "parallel",
  perpendicular: "perpendicular",
  equal_length: "equal in length",
  on_circle: "on a circle",
  tangent: "tangent",
  collinear: "collinear",
};

export function book1(): string {
  const unmarked = ["parallel", "perpendicular", "equal_length", "tangent"]
    .filter((r) => get(`facts:audit.${r}`))
    .map((r) => `<li>${RELATION_WORDS[r]}: ${n(`facts:audit.${r}.pct_geometric_unmarked`, "pctv")} of the true pairs
      are unmarked (${n(`facts:audit.${r}.geometric_not_asserted`, "int")} unmarked,
      ${n(`facts:audit.${r}.asserted_and_geometric`, "int")} marked)</li>`)
    .join("");
  const relations = (get("results:relations") as string[]).map((r) => `<code>${r}</code>`).join(", ");

  return `
<section class="book" id="book-1">
  <h2><span class="book-no">Book I</span> The Problem</h2>

  <article class="prop" id="prop-1">
    <h3><span class="prop-no">Proposition I.</span> A diagram says more than it is labelled to say.</h3>
    <p>Look at a textbook figure and you see that two lines are parallel. The dataset records only what
    the author marked with an arrow, a tick or a right-angle box. Measured with a ruler on the
    annotated coordinates of the Geometry3K figures, most true relations carry no mark at all:</p>
    <ul>${unmarked}</ul>
    <p>So a model graded against the marks is graded against an incomplete answer key. The same
    fine-tuned model scores ${ms("finetune", "geometry3k_test", "asserted")} against the marked
    (<em>asserted</em>) labels and ${ms("finetune", "geometry3k_test", "geometric")} against the measured
    (<em>geometric</em>) ones, on Geometry3K test (${n("results:experiments.finetune.n_seeds", "seeds")}).</p>
    <p class="qed">Q.E.D.</p>
  </article>

  <article class="prop" id="prop-2">
    <h3><span class="prop-no">Proposition II.</span> And sometimes says less.</h3>
    <p>Some figures are drawn out of scale on purpose: the text says “perpendicular”, the picture is
    off by more than ${n("facts:tolerances.not_to_scale_angle_deg", "deg")}, or two “equal” sides differ
    by more than ${n("facts:tolerances.not_to_scale_length_pct", "pctv")}. In the published Geometry3K
    data that is ${n("facts:not_to_scale.geometry3k.not_to_scale", "int")} of
    ${n("facts:not_to_scale.geometry3k.diagrams", "int")} diagrams
    (${n("facts:not_to_scale.geometry3k.pct", "pctv")}); in our figure-level test sets only
    ${n("results:not_to_scale.finetune.geometric.n_diagrams", "int")} diagrams
    (${n("results:not_to_scale.finetune.geometric.n_queries", "int")} queries), where the fine-tune scores
    ${n("results:not_to_scale.finetune.geometric.mean")}. So few diagrams support no claim, and the page says so.</p>
    <p class="qed">Q.E.D.</p>
  </article>

  <article class="prop" id="prop-3">
    <h3><span class="prop-no">Proposition III.</span> The Ruler.</h3>
    <p>Truth here is decided by a coordinate verifier: one exact check per relation, with a tolerance.
    Two lines are parallel or perpendicular within ${n("facts:tolerances.parallel_deg", "deg")}; two
    segments are equal within ${n("facts:tolerances.equal_length_pct", "pctv")}; a point is on a circle
    within ${n("facts:tolerances.on_circle_pct", "pctv")} of the radius; three points are collinear within
    ${n("facts:tolerances.collinear_pct", "pctv")} of the diagram's size.</p>
    <aside class="note">The first tolerances came from valleys in the histograms of measured margins
    (parallel ${n("facts:tolerances.calibration.parallel.valley_or_knee", "deg")}, perpendicular
    ${n("facts:tolerances.calibration.perpendicular.valley_or_knee", "deg")}). A first round of
    ${n("facts:gold.round1.n", "int")} hand-labelled queries showed that the eye accepts more
    (false negatives ${n("facts:gold.round1.fn_rate", "pct1")}), so the tolerances were widened and checked
    on ${n("facts:gold.round2.n", "int")} fresh queries: population-weighted false-positive rate
    ${n("facts:gold.round2_adjudicated_weighted.fp_rate", "pct1")}, false-negative rate
    ${n("facts:gold.round2_adjudicated_weighted.fn_rate", "pct1")} after adjudication.</aside>
    <p class="qed">Q.E.D.</p>
  </article>

  <article class="prop" id="task">
    <h3><span class="prop-no">The task, stated formally.</span></h3>
    <p>Given a diagram and a highlighted query (two segments, a point and a circle, a segment and a
    circle, or a point and a line), predict which of the relations ${relations} hold. <em>None</em> means
    all are false. The model never finds the points itself: primitive detection is out of scope,
    so the query is always supplied.</p>
  </article>
</section>`;
}
