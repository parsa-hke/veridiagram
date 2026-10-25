# Report figures (plan §8.6)

Every figure is written by a script; nothing is hand-edited. Regenerate with the command in the last column (from the repository root, inside the virtualenv).

| figure | file(s) | present | regenerate |
|---|---|---|---|
| loader check | `loader_check_geometry3k.png`, `loader_check_pgdp5k.png`, `loader_check_pgps9k.png` | yes, yes, yes | `python -m data.sanity` |
| tolerance histograms (combined) | `tolerance_all.png` | yes | `python -m verifier.tolerance` |
| dataset examples with overlays | `dataset_examples.png` | yes | `python -m data.dataset --config configs/smoke.yaml --examples` |
| synthetic examples | `synthetic_examples.png` | yes | `python -m data.synthetic.check` |
| scaling curve (Part A) | `scaling_square.png`, `scaling.png` | yes, yes | `python -m analysis.phase7` |
| headline table | `headline_table.png` | yes | `python -m analysis.figures (data: python -m analysis.headline_table)` |
| per-relation F1 bars per test set | `per_relation_f1.png` | yes | `python -m analysis.figures (data: python -m analysis.phase8)` |
| attention maps | `attention_grid.png` | yes | `python -m analysis.attention_maps` |
| audit examples: asserted-but-false, true-but-unmarked (datasets) | `audit_geometry3k_asserted_not_geometric.png`, `audit_geometry3k_geometric_not_asserted.png` | yes, yes | `python -m verifier.audit` |
| audit examples with the model's answer | `audit_examples_model.png` | yes | `python -m analysis.asserted_vs_geometric` |
| error analysis: accuracy vs verifier margin; F1 by points and segment length | `error_margin.png`, `error_breakdowns.png` | yes, yes | `python -m analysis.error_analysis` |
