import { book1 } from "./books/book1";
import { book2 } from "./books/book2";
import { book3 } from "./books/book3";
import { get, lit, n } from "./data";
import { book4 } from "./demo/book4";
import { hero } from "./figures/geometry";
import { resetNotes } from "./ui";

/** The frontispiece figure: a compass circle, two Byrne bars, and the machine's green trace over both.
 * Static here; milestone 3 draws it stroke by stroke. */
function emblem(): string {
  return `
<svg class="emblem" viewBox="0 0 320 240" role="img" aria-labelledby="emblem-title">
  <title id="emblem-title">A circle, a red bar and a yellow bar, and the model's green trace over both bars</title>
  <circle class="ink-line" cx="160" cy="120" r="92" />
  <circle class="ink-dot" cx="160" cy="120" r="3" />
  <line class="bar bar-red" x1="58" y1="88" x2="262" y2="62" />
  <line class="bar bar-yellow" x1="58" y1="178" x2="262" y2="152" />
  <g class="machine-trace">
    <line class="halo" x1="58" y1="88" x2="262" y2="62" />
    <line class="halo" x1="58" y1="178" x2="262" y2="152" />
    <line class="core" x1="58" y1="88" x2="262" y2="62" />
    <line class="core" x1="58" y1="178" x2="262" y2="152" />
  </g>
</svg>`;
}

function frontispiece(): string {
  return `
<header class="frontispiece" id="top">
  <p class="imprint">The Elements, a lost edition · written for a machine</p>
  <h1>${get("facts:meta.title")}</h1>
  ${hero() || `<figure class="emblem-figure">${emblem()}</figure>`}
  <p class="subtitle">${get("facts:meta.subtitle")}</p>
  <p class="course smallcaps">${get("facts:meta.author")} · ${lit(String(get("facts:meta.course")))} ·
    ${get("facts:meta.institution")} · ${n("facts:meta.year", "raw")}</p>
  <div class="actions">
    <a class="button primary" href="#book-1">Read the Elements</a>
    <a class="button" href="#book-4">Construct something</a>
  </div>
</header>`;
}

function contents(): string {
  return `
<nav class="contents" aria-label="Contents">
  <h2 class="smallcaps">Contents</h2>
  <ol>
    <li><a href="#book-1"><span class="smallcaps">Book I</span> The Problem</a></li>
    <li><a href="#book-2"><span class="smallcaps">Book II</span> The Instrument</a></li>
    <li><a href="#book-3"><span class="smallcaps">Book III</span> The Propositions</a></li>
    <li><a href="#book-4"><span class="smallcaps">Book IV</span> Constructions</a></li>
    <li><a href="#colophon"><span class="smallcaps">Colophon</span></a></li>
  </ol>
</nav>`;
}

function deployLine(): string {
  if (!get("facts:deploy")) return "";
  return `<p>Measured on the live instrument (${get("facts:deploy.device")}, ${lit(String(get("facts:deploy.date")))}): one
  question takes ${n("facts:deploy.ask_median_s", "f1")} s (median of ${n("facts:deploy.ask_runs", "int")}); reading a
  figure, ${n("facts:deploy.read_per_query_s", "f1")} s a question. Its answers to
  ${n("facts:deploy.parity_n", "int")} validation questions match the model run locally within
  ${n("facts:deploy.parity_max_dp", "f3")}.</p>`;
}

function colophon(): string {
  const repo = get("facts:meta.repo_public") && get("facts:meta.repo_url")
    ? `<a href="${get("facts:meta.repo_url")}">the code repository</a>`
    : "the code repository, which is private for now";
  const author = String(get("facts:meta.author") ?? "");
  const title = String(get("facts:meta.title") ?? "");
  const cite = `${author.split(" ").reverse().join(", ")} (${get("facts:meta.year")}). ${title}: ${String(get("facts:meta.subtitle") ?? "").replace(/\.$/, "")}. ${get("facts:meta.course")}, ${get("facts:meta.institution")}.`;
  return `
<footer class="colophon" id="colophon" data-running="Colophon">
  <h2>Colophon</h2>
  <p>Set in Libre Caslon and IM Fell English, after Oliver Byrne's <em>The First Six Books of the Elements of
  Euclid</em> (${n("facts:meta.byrne_year", "raw")}), which taught geometry in red, yellow and blue instead of
  letters. Green, which Byrne never used, belongs here to the machine alone.</p>
  <p>Every number on this page is read from JSON written by <code>python -m web.export</code> from the
  project's run records (export of commit <code>${get("results:_export.commit") ?? "unknown"}</code>). See ${repo}.</p>
  ${deployLine()}

  <h3 class="smallcaps">What the instrument cannot do</h3>
  <p>It does not find points, lines or circles: you supply them, and it judges relations between the parts you
  point at. It was trained on clean printed textbook diagrams; photographs and hand-drawn sketches are outside
  what it has seen. Results from a single training seed are labelled as such, and the not-to-scale subset is
  too small to support a claim.</p>

  <h3 class="smallcaps">Privacy</h3>
  <p>Pictures you upload and figures you draw are processed in memory to answer you and are never stored or
  logged. The page keeps two preferences in your browser (the edition and the cursor) and nothing else.</p>

  <h3 class="smallcaps">Licences and sources</h3>
  <ul class="colophon-list">
    <li>The model is fine-tuned from <span class="ref">DINOv2 ViT-L/14</span> (Meta AI), released under the <span class="ref">Apache License 2.0</span>.</li>
    <li>Training and evaluation data: Geometry3K (<span class="ref">Lu et al., ACL 2021</span>; its repository is MIT-licensed, with no
    separate data licence stated), PGDP5K (<span class="ref">Hao et al., ICPR 2022; Zhang et al., IJCAI 2022</span>; MIT per its dataset
    card) and PGPS9K (<span class="ref">Zhang et al., IJCAI 2023</span>; MIT per its dataset card). No image from these datasets is shown
    on this page: every diagram here is drawn by the project's own renderer.</li>
    <li>Fonts: Libre Caslon, IM Fell English and JetBrains Mono, under the SIL Open Font License.</li>
  </ul>

  <h3 class="smallcaps">Cite this</h3>
  <p class="cite"><span class="ref" data-cite>${cite}</span> <button type="button" class="tool plain" data-copy-cite>Copy</button></p>
  <p class="smallcaps">${author} · ${get("facts:meta.institution")} · ${n("facts:meta.year", "raw")}</p>
</footer>`;
}

export function renderPage(root: Element): void {
  resetNotes();
  root.innerHTML = frontispiece() + contents() + `<main>${book1()}${book2()}${book3()}${book4()}</main>` + colophon();
}
