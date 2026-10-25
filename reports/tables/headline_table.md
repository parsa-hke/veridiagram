### Queries solved by the probe (sq_probe) and the fine-tune (sq_finetune_colab); geometric labels, val-tuned thresholds; mean over seeds [0, 1, 2]

| test set | queries | both | fine-tune only | probe only | neither |
|---|---|---|---|---|---|
| geometry3k_test | 11319 | 7612 (67.2%) | 2554 (22.6%) | 396 (3.5%) | 756 (6.7%) |
| pgdp5k_heldout | 25272 | 14080 (55.7%) | 8336 (33.0%) | 742 (2.9%) | 2114 (8.4%) |
| synth_extrap | 10000 | 7011 (70.1%) | 1941 (19.4%) | 374 (3.7%) | 675 (6.8%) |

### The same per (query, relation) decision

| test set | relation | decisions | both | fine-tune only | probe only | neither |
|---|---|---|---|---|---|---|
| geometry3k_test | all | 23009 | 81.5% | 13.4% | 2.1% | 3.1% |
| geometry3k_test | parallel | 5845 | 92.1% | 7.3% | 0.3% | 0.3% |
| geometry3k_test | perpendicular | 5845 | 85.2% | 12.3% | 1.0% | 1.5% |
| geometry3k_test | equal_length | 6348 | 62.4% | 23.2% | 5.8% | 8.7% |
| geometry3k_test | on_circle | 457 | 85.7% | 13.2% | 0.9% | 0.2% |
| geometry3k_test | tangent | 616 | 83.2% | 10.7% | 2.2% | 3.9% |
| geometry3k_test | collinear | 3898 | 90.5% | 8.5% | 0.2% | 0.7% |
| pgdp5k_heldout | all | 53478 | 74.3% | 20.2% | 1.9% | 3.6% |
| pgdp5k_heldout | parallel | 14103 | 85.9% | 13.7% | 0.2% | 0.2% |
| pgdp5k_heldout | perpendicular | 14103 | 83.4% | 13.8% | 1.1% | 1.7% |
| pgdp5k_heldout | equal_length | 15301 | 45.8% | 38.8% | 5.3% | 10.1% |
| pgdp5k_heldout | on_circle | 723 | 81.7% | 16.8% | 0.9% | 0.6% |
| pgdp5k_heldout | tangent | 999 | 81.2% | 12.7% | 2.1% | 4.0% |
| pgdp5k_heldout | collinear | 8249 | 90.1% | 9.2% | 0.2% | 0.5% |
| synth_extrap | all | 17380 | 79.9% | 13.9% | 2.8% | 3.4% |
| synth_extrap | parallel | 3690 | 82.3% | 15.4% | 0.9% | 1.3% |
| synth_extrap | perpendicular | 3690 | 89.2% | 6.1% | 2.7% | 2.0% |
| synth_extrap | equal_length | 3756 | 47.9% | 34.9% | 7.6% | 9.5% |
| synth_extrap | on_circle | 520 | 84.6% | 13.1% | 0.4% | 1.9% |
| synth_extrap | tangent | 471 | 83.2% | 7.4% | 4.7% | 4.7% |
| synth_extrap | collinear | 5253 | 93.6% | 4.0% | 0.9% | 1.5% |
