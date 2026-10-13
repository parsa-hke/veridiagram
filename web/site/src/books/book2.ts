import { get, n } from "../data";

function backbone(key: "main" | "demo", label: string): string {
  const b = `facts:backbones.${key}`;
  if (!get(`${b}.available`)) {
    return `<li>${label}: <code>${get(`${b}.name`)}</code> (architecture facts not exported:
      its configuration is not in the local cache; run <code>python -m web.export --online</code> after
      logging in to Hugging Face)</li>`;
  }
  return `<li>${label}: <code>${get(`${b}.name`)}</code>, ${n(`${b}.params_M`, "f1")}M parameters,
    ${n(`${b}.layers`, "int")} transformer layers of width ${n(`${b}.hidden`, "int")}. A square
    ${n("facts:input.long_side", "int")}-px input is cut into a ${n(`${b}.grid`, "int")} × ${n(`${b}.grid`, "int")}
    grid of ${n(`${b}.patch`, "int")}-px patches: ${n(`${b}.patch_tokens`, "int")} patch tokens and one CLS token${
    get(`${b}.register_tokens`) ? `, plus ${n(`${b}.register_tokens`, "int")} register tokens` : ""}.</li>`;
}

export function book2(): string {
  const colour = get("facts:overlay.color") as number[];
  const synth = Object.keys(get("facts:data.synth_train") ?? {});
  const largest = synth[synth.length - 1];
  return `
<section class="book" id="book-2">
  <h2><span class="book-no">Book II</span> The Instrument</h2>

  <article class="prop" id="data">
    <h3><span class="prop-no">The data.</span></h3>
    <ul>
      <li>Geometry3K figures, split by figure (not by problem, so no test figure was seen in training):
        ${n("facts:data.geometry3k_train_figures", "int")} train,
        ${n("facts:data.geometry3k_val_figures", "int")} validation,
        ${n("facts:data.geometry3k_test_figures", "int")} test.</li>
      <li>A held-out source: ${n("facts:data.pgdp5k_heldout_figures", "int")} PGDP5K figures with no
        Geometry3K copy.</li>
      <li>Synthetic diagrams with exact labels, drawn by our own renderer: up to
        ${n(`facts:data.synth_train.${largest}`, "int")} for training
        (${n("facts:data.synthetic_points.0", "int")} to ${n("facts:data.synthetic_points.1", "int")} points), and
        ${n("facts:data.synth_extrap_diagrams", "int")} larger ones for extrapolation
        (${n("facts:data.extrap_points.0", "int")} to ${n("facts:data.extrap_points.1", "int")} points).</li>
      <li>An unlabelled pool for self-training: PGPS9K figures, of which
        ${n("pseudo_labels:pool_diagrams", "int")} were never seen with labels.</li>
    </ul>
    <aside class="note">Each diagram yields at most ${n("facts:data.max_queries_per_diagram", "int")} queries;
    training uses ${n("facts:data.train_queries", "int")} real queries, and the extrapolation set is evaluated
    on ${n("facts:data.eval_caps.synth_extrap", "int")} of its queries.</aside>
  </article>

  <article class="prop" id="query">
    <h3><span class="prop-no">The query.</span></h3>
    <p>The question is asked in colour. The queried shapes are painted onto the image in green,
    <span class="swatch-machine">RGB(${colour.map((_, i) => n(`facts:overlay.color.${i}`, "int")).join(", ")})</span>, after the
    image is resized to ${n("facts:overlay.long_side", "int")} px, so the highlight is the same width in
    every input: a ${n("facts:overlay.width_px", "int")}-px core inside a ${n("facts:overlay.halo_width_px", "int")}-px
    translucent halo.</p>
    <p>Green was chosen because real diagrams do not use it. In ${n("facts:colour_check.near_highlight_r80.images", "int")}
    sampled textbook diagrams, ${n("facts:colour_check.near_highlight_r80.images_with", "int")} contain the highlight's
    colour; green-family ink of any shade appears in ${n("facts:colour_check.hue_green.pct", "pctv")} of them,
    against ${n("facts:colour_check.hue_red.pct", "pctv")} for red and ${n("facts:colour_check.hue_blue.pct", "pctv")}
    for blue.</p>
    <aside class="note">A pixel counts as the highlight's colour when it lies within RGB distance
    ${n("facts:colour_check.near_highlight_r80.radius", "int")} of it; an image counts when at least
    ${n("facts:colour_check.near_highlight_r80.min_pixels", "int")} of its pixels do. Darker and teal greens from coloured textbook figures make up the green-family share.</aside>
  </article>

  <article class="prop" id="model">
    <h3><span class="prop-no">The model.</span></h3>
    <p>A vision transformer pre-trained without labels (DINO), with a small head: the CLS token joined
    to the mean patch token, a LayerNorm, and one sigmoid per relation.</p>
    <ul>
      ${backbone("main", "Main experiments")}
      ${backbone("demo", "Best model and the live demo")}
    </ul>
  </article>

  <article class="prop" id="recipe">
    <h3><span class="prop-no">The recipe.</span></h3>
    <p>First fit the head alone on frozen features of ${n("facts:recipe.lp_queries", "int")} training queries,
    then fine-tune everything (LP-FT) for ${n("facts:recipe.epochs", "int")} epochs of
    ${n("facts:recipe.epoch_images", "int")} images. Augmentation keeps every relation true: rotations up to
    ±${n("facts:recipe.augment.rotation_deg", "deg")}, flips, scaling between
    ${n("facts:recipe.augment.scale.0", "f1")} and ${n("facts:recipe.augment.scale.1", "f1")}, noise. Never shear,
    which would break parallels.</p>
    <aside class="note">Layer-wise learning-rate decay ${n("facts:recipe.llrd", "f1")} from the top block down;
    backbone learning rate ${n("facts:recipe.backbone_lr", "sci")}, head ${n("facts:recipe.head_lr", "sci")};
    ${n("facts:recipe.amp_dtype", "raw")} mixed precision; weight EMA ${n("facts:recipe.ema_decay", "raw")};
    batch ${n("facts:recipe.batch_size", "int")}.</aside>
    <p>The plan as first written needed ${n("facts:compute.planned_gpu_hours", "int")} GPU-hours
    (measured at ${n("facts:compute.train_img_per_s_448", "f1")} images per second per GPU). The project had a
    classroom PC for ${n("facts:meta.lab_visit_hours", "raw")} hours per visit and a Colab account, so every
    fine-tune got the same fixed budget of ${n("facts:recipe.total_images", "int")} images.</p>
  </article>

  <article class="prop" id="extensions">
    <h3><span class="prop-no">Two extensions.</span></h3>
    <p><strong>Part A:</strong> the verifier as an unlimited labeller. Synthetic diagrams come with exact
    labels for free; does adding them help on real diagrams and on larger ones?</p>
    <p><strong>Part B:</strong> self-training on unlabelled real scans. A teacher labels the pool, and a
    deductive consistency check (parallels are transitive; a pair cannot be both parallel and
    perpendicular) throws out diagrams whose answers contradict each other.</p>
  </article>
</section>`;
}
