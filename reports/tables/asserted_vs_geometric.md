## sq_finetune_colab: geometric vs asserted labels (seeds pooled, val-tuned thresholds)

### Macro-F1 and per-relation F1 against each label set

| unit | macro geometric | macro asserted | parallel g / a | perpendicular g / a | equal_length g / a | on_circle g / a | tangent g / a | collinear g / a |
|---|---|---|---|---|---|---|---|---|
| geometry3k_test | 0.886 | 0.570 | 0.966 / 0.163 | 0.928 / 0.645 | 0.668 / 0.083 | 0.993 / 0.993 | 0.810 / 0.000 | 0.953 / 0.968 |
| pgdp5k_heldout | 0.894 | 0.576 | 0.978 / 0.217 | 0.903 / 0.561 | 0.672 / 0.141 | 0.988 / 0.988 | 0.858 / 0.000 | 0.965 / 0.974 |
| synth_extrap | 0.665 | 0.466 | 0.789 / 0.531 | 0.659 / 0.359 | 0.499 / 0.057 | 0.907 / 0.746 | 0.614 / 0.526 | 0.520 / 0.579 |

### Label cells and the model's positive rate in each (asserted-only = marked but drawn out of tolerance; geometric-only = true in the drawing, unmarked)

geometry3k_test:

| relation | n | agree + (model +) | agree - (model +) | asserted-only (model +) | geometric-only (model +) | errors vs asserted: label's | errors vs geometric: followed the mark |
|---|---|---|---|---|---|---|---|
| parallel | 17535 | 138 (0.99) | 16029 (0.00) | 0 (-) | 1368 (0.97) | 1330 of 1395 (95%) | 38 of 103 (37%) |
| perpendicular | 17535 | 1482 (0.97) | 14517 (0.01) | 3 (0.67) | 1533 (0.88) | 1349 of 1593 (85%) | 187 of 431 (43%) |
| equal_length | 19044 | 198 (0.91) | 14994 (0.10) | 15 (0.40) | 3837 (0.68) | 2606 of 4119 (63%) | 1246 of 2759 (45%) |
| on_circle | 1371 | 1005 (0.99) | 366 (0.00) | 0 (-) | 0 (-) | 0 of 15 (0%) | 0 of 15 (0%) |
| tangent | 1848 | 0 (-) | 1572 (0.05) | 0 (-) | 276 (0.87) | 239 of 314 (76%) | 37 of 112 (33%) |
| collinear | 11694 | 1131 (1.00) | 10458 (0.00) | 12 (0.25) | 93 (0.26) | 33 of 76 (43%) | 72 of 115 (63%) |
| all | 69027 | 3954 (0.98) | 57936 (0.03) | 30 (0.37) | 7107 (0.78) | 5557 of 7512 (74%) | 1580 of 3535 (45%) |

pgdp5k_heldout:

| relation | n | agree + (model +) | agree - (model +) | asserted-only (model +) | geometric-only (model +) | errors vs asserted: label's | errors vs geometric: followed the mark |
|---|---|---|---|---|---|---|---|
| parallel | 42309 | 438 (0.99) | 38781 (0.00) | 0 (-) | 3090 (0.98) | 3025 of 3117 (97%) | 65 of 157 (41%) |
| perpendicular | 42309 | 2469 (0.95) | 35976 (0.01) | 75 (0.44) | 3789 (0.83) | 3206 of 3726 (86%) | 658 of 1178 (56%) |
| equal_length | 45903 | 1032 (0.88) | 36249 (0.13) | 24 (0.38) | 8598 (0.74) | 6362 of 11175 (57%) | 2260 of 7073 (32%) |
| on_circle | 2169 | 1416 (0.98) | 753 (0.01) | 0 (-) | 0 (-) | 0 of 33 (0%) | 0 of 33 (0%) |
| tangent | 2997 | 0 (-) | 2355 (0.04) | 0 (-) | 642 (0.86) | 553 of 647 (85%) | 89 of 183 (49%) |
| collinear | 24747 | 2367 (1.00) | 22248 (0.00) | 0 (-) | 132 (0.32) | 42 of 125 (34%) | 90 of 173 (52%) |
| all | 160434 | 7722 (0.96) | 136362 (0.04) | 99 (0.42) | 16251 (0.81) | 13188 of 18823 (70%) | 3162 of 8797 (36%) |

synth_extrap:

| relation | n | agree + (model +) | agree - (model +) | asserted-only (model +) | geometric-only (model +) | errors vs asserted: label's | errors vs geometric: followed the mark |
|---|---|---|---|---|---|---|---|
| parallel | 11070 | 246 (0.94) | 10524 (0.02) | 0 (-) | 300 (0.77) | 230 of 406 (57%) | 70 of 246 (28%) |
| perpendicular | 11070 | 222 (0.85) | 10368 (0.03) | 0 (-) | 480 (0.66) | 315 of 670 (47%) | 165 of 520 (32%) |
| equal_length | 11268 | 114 (0.66) | 9924 (0.16) | 0 (-) | 1230 (0.72) | 886 of 2473 (36%) | 344 of 1931 (18%) |
| on_circle | 1560 | 108 (0.99) | 1353 (0.00) | 0 (-) | 99 (0.69) | 68 of 73 (93%) | 31 of 36 (86%) |
| tangent | 1413 | 129 (0.63) | 1230 (0.05) | 12 (0.08) | 42 (0.57) | 35 of 148 (24%) | 19 of 132 (14%) |
| collinear | 15759 | 153 (0.99) | 15336 (0.01) | 0 (-) | 270 (0.20) | 55 of 220 (25%) | 215 of 380 (57%) |
| all | 52140 | 972 (0.86) | 48735 (0.05) | 12 (0.08) | 2421 (0.65) | 1589 of 3990 (40%) | 844 of 3245 (26%) |
