# Data

Three real sources, downloaded 2026-10-10 into `data/raw/<source>/` (gitignored, about 1 GB extracted).
Everything below `data/raw/` is untouched; `python -m data.build` writes `data/processed/<source>.jsonl`
in the unified schema (`data/schema.py`).

## Sources, licences, citations (task 2.1)

| Source | Download used | Archive (sha256 prefix) | Licence | Cite |
|---|---|---|---|---|
| Geometry3K | `https://raw.githubusercontent.com/lupantech/InterGPS/main/data/geometry3k/{train,val,test,logic_forms,symbols}.zip` (Inter-GPS repo; the README also links a Google Drive folder) | train `e12ee4d2`, val `67b1b0a9`, test `a0d5dece`, logic_forms `264cd731`, symbols `90966955` | Repository is MIT; no separate data licence stated | Lu et al., ACL 2021 |
| PGDP5K | `https://huggingface.co/datasets/PeijieWang/PGDP5K/resolve/main/PGDP5K.zip` (HF copy; official page `http://www.nlpr.ia.ac.cn/databases/CASIA-PGDP5K/`) | `2fe9c0c3` | HF dataset card: MIT | Hao et al., ICPR 2022; Zhang et al., IJCAI 2022 |
| PGPS9K | `https://huggingface.co/datasets/PeijieWang/PGPS9K/resolve/main/PGPS9K.zip` (official page `https://nlpr.ia.ac.cn/databases/CASIA-PGPS9K/`) | `a74b2b81` | HF dataset card: MIT | Zhang et al., IJCAI 2023 |

The HF copies of PGDP5K and PGPS9K are not password-protected (the PGPS GitHub README mentions a
password for its own download). Full sha256 sums are in CHANGELOG.md (2026-10-10).

```bibtex
@inproceedings{lu2021intergps,
  title     = {Inter-GPS: Interpretable Geometry Problem Solving with Formal Language and Symbolic Reasoning},
  author    = {Lu, Pan and Gong, Ran and Jiang, Shibiao and Qiu, Liang and Huang, Siyuan and Liang, Xiaodan and Zhu, Song-Chun},
  booktitle = {ACL}, year = {2021}}
@inproceedings{hao2022pgdp5k,
  title     = {PGDP5K: A Diagram Parsing Dataset for Plane Geometry Problems},
  author    = {Hao, Yihan and Zhang, Mingliang and Yin, Fei and Huang, Linlin},
  booktitle = {ICPR}, year = {2022}}
@inproceedings{zhang2022pgdp,
  title     = {Plane Geometry Diagram Parsing},
  author    = {Zhang, Ming-Liang and Yin, Fei and Hao, Yi-Han and Liu, Cheng-Lin},
  booktitle = {IJCAI}, year = {2022}}
@inproceedings{zhang2023pgps,
  title     = {A Multi-Modal Neural Geometric Solver with Textual Clauses Parsed from Diagram},
  author    = {Zhang, Ming-Liang and Yin, Fei and Liu, Cheng-Lin},
  booktitle = {IJCAI}, year = {2023}}
```

## Raw formats, as found (task 2.2)

Inspected by hand: Geometry3K train/0, train/1, train/500, val/2200, test/2500, test/2886;
PGDP5K 4238, 794, 1386, 4867 (+ one record per symbol class: 1009, 1766, 120, 2134, 197);
PGPS9K img_0, img_1, img_1934 and problems prob_0, prob_13 (PGPS9K) and prob_1 (Geometry3K split).

### Geometry3K (`data/raw/geometry3k/`)

```
{train,val,test}/<id>/        2101 / 300 / 601 problems, ids 0-2100 / 2101-2400 / 2401-3001
  img_diagram.png             diagram, RGBA with transparent background
  img_diagram_point.png       same diagram (RGB) + each annotated point as a pure red (255,0,0) dot
                              and its name in pure blue (0,0,255)
  data.json                   problem_text, choices, answer, problem_type_graph, problem_type_goal,
                              source (textbook pdf), comment, id, annot_id, is_standard,
                              compact_choices, precise_value, rough_value, compact_text,
                              annotat_text, unit, img_width, img_height, data_type, date, system
  logic_form.json             text_logic_form, dissolved_text_logic_form, diagram_logic_form,
                              line_instances, point_positions, circle_instances
logic_forms/diagram_logic_forms_annot.json   {id: {annot_id, point_instances, line_instances,
                              circle_instances, diagram_logic_forms, point_positions,
                              redundant_point_positions}}  (same content as the per-problem files)
logic_forms/text_logic_forms_annot{,_dissolved}.json
symbols/<id>.png, <id>.xml    VOC-style boxes (<object><name>text|...</name><bndbox>...) of symbols
                              and text; 3002 png, 2999 xml. Not used.
```

- **Points**: `point_positions` = `{name: [x, y]}`, names are letters (occasionally primed).
  Coordinates are **not image pixels**: they carry a per-diagram translation (+123/+124 px on id 0;
  min x and min y are near 0 and 63 diagrams have negative values). The loader recovers the shift by
  matching the positions to the red dots in `img_diagram_point.png` (all points matched in 2985 of
  3001 diagrams with points; 16 match partially, worst 2 of 5, flagged `offset_partial`). Scale is 1.
- Names are invented by annotators when the figure shows none (e.g. a figure with no letters still
  has A, B, C, D).
- **Segments**: `line_instances` = list of strings `"AB"` (concatenated names; can be `""`). They
  include sub-segments of collinear chains.
- **Circles**: `circle_instances` = list of centre names (`""` when there is none). The radius is
  not stored numerically: predicates refer to it symbolically (`radius_5_0`). The loader uses the
  median distance from the centre to its points on the circle.
- **Relations**: only as logic-form strings in `diagram_logic_form`. Heads over all 3002 problems:
  PointLiesOnLine 11718, Equals 10666, PointLiesOnCircle 2670, Perpendicular 1609, Parallel 384,
  empty string 165, Tangent 1. `Equals` covers `LengthOf(Line)` = value (5048),
  `MeasureOf(Angle)` = value (2691), `MeasureOf` = `MeasureOf` (2331, of which 2084 merely name an
  angle, e.g. `MeasureOf(angle 1)`; 193 are real angle equalities), `LengthOf` = `LengthOf` (529),
  areas and perimeters.
- **Problems**: 3002 problems but only 1851 distinct figures; the published split is by problem.
- **Defects**: 261 diagrams reference points that have no position (e.g. val/2200 uses D);
  id 520 has no positions at all. Against PGDP5K's pixel re-annotation of the same images the
  median point error is 2.4 px but 4.5% of points are >10 px off (e.g. id 0: X and Y about 40 px off).

### PGDP5K (`data/raw/pgdp5k/PGDP5K/`)

```
annotations/{train,val,test}.json   {id: record}, 3500 / 500 / 1000 (readme.txt swaps val and test)
{train,val,test}/<id>.png           images
our_diagram_logic_forms_annot.json  {id: {point_instances, line_instances, circle_instances,
                                    diagram_logic_forms, point_positions}} for all 5000, derived by
                                    the authors from the primitive annotation, pixel coordinates
readme.txt
```

Record (PGDP primitive format, also used by PGPS9K):

```
file_name, width, height
geos.points   [{id: "p0", loc: [[x, y]]}]                      pixels, origin top-left
geos.lines    [{id: "l0", loc: [[x1, y1], [x2, y2]]}]          only the longest segment of a line
geos.circles  [{id: "c0", loc: [[cx, cy], r, "1111"]}]         r in pixels; "1111" = quadrant flags
symbols       [{id: "s0", sym_class, text_class, text_content, bbox: [x, y, w, h]}]
              sym_class: text, perpendicular, bar, double bar, triple bar, quad bar, parallel,
              double parallel, triple parallel, angle, double angle, triple angle, quad angle,
              penta angle, arrow, head, head_len
              text_class (text only): point, line, len, angle, degree, area
relations.geo2geo  [p, l|c, "endpoint" | "online" | "oncircle" | "center"]
relations.sym2geo  [s, [targets]]  point label [p]; perpendicular [p, l, l]; bar* [p, p, l];
                                   parallel* [l]; angle* [p, l, l]
relations.sym2sym  [...]           arrows/heads to text
```

- **Point names** come from `text` symbols with `text_class: point` linked to a point by `sym2geo`.
  14607 points in 3089 diagrams have no label (loader names them `_p<k>`); labels can be primed
  (`B'`) or digits.
- **Relations** are drawn marks, not text: tick classes (equal lengths), parallel arrows, angle
  arcs (equal angles), right-angle squares, plus incidences. The loader's parse agrees with the
  authors' logic forms: parallel 520/520, equal-length pairs 2863/2863, perpendicular 2452/2467
  (the rest are marks linked to fewer than two lines), equal-angle 1420 common.
- 1809 of its images are Geometry3K images (paper: 1813).

### PGPS9K (`data/raw/pgps9k/PGPS9K/`)

```
diagram_annotation.json   {"img_<n>": record} for 4000 diagrams, PGDP primitive format (above)
Diagram/img_<n>.png       clean diagrams
Diagram_Visual/img_<n>.png  diagrams with the annotation drawn on
PGPS9K/{train,test}.json  8021 / 1000 problems {"prob_<n>": {diagram, text, parsing_stru_seqs,
                          parsing_sem_seqs, expression, answer, type, book, page, choices}}
Geometry3K/{train,test}.json  8432 / 589 problems, the Geometry3K problems in the same format
knowledge_point.txt       Chinese/English names of the 30 problem types
```

- `parsing_stru_seqs`: `line A B C` (collinear points), `\odot O lieson A B C`, `line m lieson ...`.
- `parsing_sem_seqs`: `AC \perp EB on B`, `AB \parallel CD`, `XY = YZ`, `m \angle ABC = m \angle DEF`,
  values (`AB = 7`, `m \angle URS = 49`, `m \widehat AE = 72`). The loader does not use clauses:
  relations come from the diagram annotation, as for PGDP5K, so both sources share one definition.
- Problems are split, diagrams are not: 701 diagrams serve both train and test problems. Diagrams get
  split `all`; problem splits and books are in `meta`.
- 1734 of its images are Geometry3K images (paper: 1738) and 2816 are PGDP5K images.

## Unified schema and derived relations

See the `data/schema.py` docstring. Asserted relations per source:

| Relation | Geometry3K | PGDP5K / PGPS9K |
|---|---|---|
| parallel | `Parallel(Line, Line)` | same parallel-mark class |
| perpendicular | `Perpendicular(Line, Line)` | right-angle mark (the two marked lines) |
| equal_length | `Equals(LengthOf(Line), LengthOf(Line))` | same tick-mark class |
| on_circle | `PointLiesOnCircle` | `oncircle` incidence |
| tangent | `Tangent(Line, Circle)` | none in the annotation |
| collinear | `PointLiesOnLine(P, Line(A, B))` | `online` incidence |
| equal_angle | `Equals(MeasureOf(Angle(3 pts)), MeasureOf(Angle(3 pts)))` | same angle-arc class |
| midpoint, perp_bisector | derived from collinear + equal_length (+ perpendicular) | same |

parallel, equal_length and equal_angle are closed under transitivity within a diagram. Equal text
values (two sides both labelled 5) are not turned into relations.

## Scripts

```bash
python -m data.build      # raw -> data/processed/<source>.jsonl, prints per-source counts
python -m data.overlap    # duplicates -> data/processed/{overlap,figure_clusters}.csv, reports/tables/source_overlap.csv
python -m data.census     # reports/tables/relation_census.csv
python -m data.sanity     # reports/figures/{image_sizes,g3k_vs_pgdp5k_coords}.png, reports/tables/coordinate_sanity.csv
```
