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
| selftrain_control | 7.11 control: no pseudo-labels | 3 | 0.922 ± 0.006 | 0.903 ± 0.005 | 0.914 ± 0.005 | 0.734 ± 0.007 |
| finetune_colab | 7.2 fine-tune, retrained on Colab (same config) | 3 | 0.878 ± 0.007 | 0.868 ± 0.007 | 0.869 ± 0.008 | 0.629 ± 0.018 |
| sq_probe | 7.1 linear probe, square padding | 3 | 0.588 ± 0.004 | 0.577 ± 0.012 | 0.517 ± 0.004 | 0.183 ± 0.010 |
| sq_scaling_1k | 7.9 Part A: 1k synthetic, square padding | 1 | 0.873 | 0.879 | 0.890 | 0.718 |
| sq_finetune_synth10k | 7.3 fine-tune + 10k synthetic (Part A: 10k), square padding | 3 | 0.880 ± 0.008 | 0.879 ± 0.008 | 0.892 ± 0.004 | 0.713 ± 0.009 |
| sq_scaling_50k | 7.9 Part A: 50k synthetic, square padding | 3 | 0.880 ± 0.008 | 0.877 ± 0.009 | 0.889 ± 0.008 | 0.707 ± 0.008 |
| sq_ablation_dinov2 | 7.5 DINOv2-L backbone, square padding | 1 | 0.940 | 0.930 | 0.940 | 0.754 |
| sq_ablation_nooverlay | 7.8 no overlay (query as tokens), square padding | 1 | 0.458 | 0.426 | 0.430 | 0.137 |
| sq_selftrain_confidence | 7.11 (a) confidence filter, square padding | 3 | 0.909 ± 0.007 | 0.898 ± 0.002 | 0.906 ± 0.003 | 0.737 ± 0.009 |
| sq_selftrain_consistency | 7.11 (b) confidence + consistency filter, square padding | 3 | 0.915 ± 0.004 | 0.907 ± 0.006 | 0.912 ± 0.005 | 0.724 ± 0.010 |
| sq_selftrain_control | 7.11 control: no pseudo-labels, square padding | 3 | 0.927 ± 0.007 | 0.915 ± 0.005 | 0.924 ± 0.003 | 0.734 ± 0.007 |
| sq_finetune_colab | 7.2 fine-tune, retrained on Colab (same config), square padding | 3 | 0.882 ± 0.004 | 0.872 ± 0.008 | 0.874 ± 0.008 | 0.629 ± 0.018 |

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
| selftrain_control | 7.11 control: no pseudo-labels | 3 | 0.646 ± 0.001 | 0.561 ± 0.003 | 0.568 ± 0.001 | 0.425 ± 0.009 |
| finetune_colab | 7.2 fine-tune, retrained on Colab (same config) | 3 | 0.630 ± 0.004 | 0.543 ± 0.007 | 0.547 ± 0.009 | 0.382 ± 0.018 |
| sq_probe | 7.1 linear probe, square padding | 3 | 0.469 ± 0.007 | 0.397 ± 0.012 | 0.355 ± 0.010 | 0.116 ± 0.011 |
| sq_scaling_1k | 7.9 Part A: 1k synthetic, square padding | 1 | 0.630 | 0.540 | 0.552 | 0.433 |
| sq_finetune_synth10k | 7.3 fine-tune + 10k synthetic (Part A: 10k), square padding | 3 | 0.628 ± 0.003 | 0.540 ± 0.003 | 0.547 ± 0.003 | 0.403 ± 0.006 |
| sq_scaling_50k | 7.9 Part A: 50k synthetic, square padding | 3 | 0.629 ± 0.008 | 0.533 ± 0.010 | 0.541 ± 0.009 | 0.402 ± 0.009 |
| sq_ablation_dinov2 | 7.5 DINOv2-L backbone, square padding | 1 | 0.649 | 0.552 | 0.568 | 0.441 |
| sq_ablation_nooverlay | 7.8 no overlay (query as tokens), square padding | 1 | 0.340 | 0.281 | 0.261 | 0.072 |
| sq_selftrain_confidence | 7.11 (a) confidence filter, square padding | 3 | 0.637 ± 0.001 | 0.545 ± 0.003 | 0.553 ± 0.002 | 0.423 ± 0.008 |
| sq_selftrain_consistency | 7.11 (b) confidence + consistency filter, square padding | 3 | 0.639 ± 0.003 | 0.548 ± 0.004 | 0.556 ± 0.002 | 0.410 ± 0.008 |
| sq_selftrain_control | 7.11 control: no pseudo-labels, square padding | 3 | 0.648 ± 0.001 | 0.561 ± 0.003 | 0.569 ± 0.002 | 0.425 ± 0.009 |
| sq_finetune_colab | 7.2 fine-tune, retrained on Colab (same config), square padding | 3 | 0.634 ± 0.003 | 0.543 ± 0.005 | 0.548 ± 0.008 | 0.382 ± 0.018 |

### Square vs batch padding (§0.2 #10 note): geometric macro-F1 change, mean over seeds (share of decisions changed)

| experiment | seeds | val | geometry3k_test | pgdp5k_heldout | synth_extrap |
|---|---|---|---|---|---|
| probe | 3 | -0.012 (3.7%) | -0.018 (5.0%) | -0.000 (4.2%) | -0.002 (0.3%) |
| scaling_1k | 1 | +0.003 (1.4%) | +0.003 (2.2%) | +0.005 (1.7%) | +0.000 (0.0%) |
| finetune_synth10k | 3 | +0.008 (1.6%) | +0.007 (2.5%) | +0.007 (1.8%) | +0.000 (0.1%) |
| scaling_50k | 3 | +0.006 (1.6%) | +0.005 (2.5%) | +0.007 (1.9%) | +0.000 (0.0%) |
| ablation_dinov2 | 1 | +0.017 (2.0%) | +0.017 (2.8%) | +0.019 (2.5%) | +0.000 (0.0%) |
| ablation_nooverlay | 1 | +0.008 (2.2%) | -0.005 (2.9%) | +0.000 (2.3%) | +0.000 (0.0%) |
| selftrain_confidence | 3 | +0.006 (1.4%) | +0.012 (2.2%) | +0.010 (1.8%) | +0.000 (0.0%) |
| selftrain_consistency | 3 | +0.007 (1.5%) | +0.011 (2.1%) | +0.009 (1.8%) | +0.000 (0.0%) |
| selftrain_control | 3 | +0.005 (1.4%) | +0.012 (2.1%) | +0.010 (1.8%) | +0.000 (0.0%) |
| finetune_colab | 3 | +0.004 (1.3%) | +0.005 (1.9%) | +0.005 (1.6%) | +0.000 (0.0%) |
