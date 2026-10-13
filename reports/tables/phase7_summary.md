### Macro-F1 against geometric labels (mean ± sd over seeds; threshold 0.5)

| experiment | plan | seeds | val | geometry3k_test | pgdp5k_heldout | synth_extrap |
|---|---|---|---|---|---|---|
| probe | 7.1 linear probe | 3 | 0.600 ± 0.002 | 0.595 ± 0.013 | 0.517 ± 0.003 | 0.185 ± 0.013 |
| finetune | 7.2 fine-tune (Part A: 0 synthetic) | 3 | 0.876 ± 0.008 | 0.867 ± 0.007 | 0.869 ± 0.008 | 0.631 ± 0.017 |
| scaling_1k | 7.9 Part A: 1k synthetic | 1 | 0.871 | 0.876 | 0.885 | 0.718 |
| finetune_synth10k | 7.3 fine-tune + 10k synthetic (Part A: 10k) | 3 | 0.872 ± 0.009 | 0.871 ± 0.010 | 0.885 ± 0.004 | 0.713 ± 0.009 |
| scaling_50k | 7.9 Part A: 50k synthetic | 3 | 0.875 ± 0.004 | 0.872 ± 0.004 | 0.882 ± 0.005 | 0.707 ± 0.008 |
| ablation_dinov2 | 7.5 DINOv2-L backbone | 1 | 0.923 | 0.913 | 0.920 | 0.754 |
| ablation_res224 | 7.6 224 px | 1 | 0.865 | 0.855 | 0.861 | 0.609 |
| ablation_noaug | 7.7 no augmentation | 1 | 0.890 | 0.868 | 0.870 | 0.632 |
| ablation_nooverlay | 7.8 no overlay (query as tokens) | 1 | 0.450 | 0.431 | 0.429 | 0.137 |
| selftrain_confidence | 7.11 (a) confidence filter | 3 | 0.903 ± 0.006 | 0.886 ± 0.003 | 0.896 ± 0.003 | 0.737 ± 0.009 |
| selftrain_consistency | 7.11 (b) confidence + consistency filter | 3 | 0.907 ± 0.005 | 0.895 ± 0.010 | 0.903 ± 0.006 | 0.724 ± 0.010 |
| selftrain_control | 7.11 control: no pseudo-labels | 1 | 0.929 | 0.909 | 0.919 | 0.741 |

### Macro-F1 against asserted labels (mean ± sd over seeds; threshold 0.5)

| experiment | plan | seeds | val | geometry3k_test | pgdp5k_heldout | synth_extrap |
|---|---|---|---|---|---|---|
| probe | 7.1 linear probe | 3 | 0.472 ± 0.013 | 0.411 ± 0.012 | 0.359 ± 0.009 | 0.116 ± 0.010 |
| finetune | 7.2 fine-tune (Part A: 0 synthetic) | 3 | 0.629 ± 0.005 | 0.543 ± 0.007 | 0.546 ± 0.010 | 0.385 ± 0.015 |
| scaling_1k | 7.9 Part A: 1k synthetic | 1 | 0.629 | 0.544 | 0.552 | 0.433 |
| finetune_synth10k | 7.3 fine-tune + 10k synthetic (Part A: 10k) | 3 | 0.626 ± 0.002 | 0.542 ± 0.003 | 0.547 ± 0.003 | 0.403 ± 0.005 |
| scaling_50k | 7.9 Part A: 50k synthetic | 3 | 0.628 ± 0.005 | 0.538 ± 0.006 | 0.541 ± 0.006 | 0.402 ± 0.009 |
| ablation_dinov2 | 7.5 DINOv2-L backbone | 1 | 0.643 | 0.554 | 0.567 | 0.441 |
| ablation_res224 | 7.6 224 px | 1 | 0.624 | 0.527 | 0.535 | 0.382 |
| ablation_noaug | 7.7 no augmentation | 1 | 0.635 | 0.550 | 0.558 | 0.441 |
| ablation_nooverlay | 7.8 no overlay (query as tokens) | 1 | 0.342 | 0.281 | 0.263 | 0.072 |
| selftrain_confidence | 7.11 (a) confidence filter | 3 | 0.637 ± 0.001 | 0.546 ± 0.004 | 0.552 ± 0.002 | 0.423 ± 0.008 |
| selftrain_consistency | 7.11 (b) confidence + consistency filter | 3 | 0.638 ± 0.003 | 0.549 ± 0.003 | 0.555 ± 0.002 | 0.410 ± 0.008 |
| selftrain_control | 7.11 control: no pseudo-labels | 1 | 0.646 | 0.558 | 0.567 | 0.418 |
