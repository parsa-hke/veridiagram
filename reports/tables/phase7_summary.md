### Macro-F1 against geometric labels (mean ± sd over seeds; threshold 0.5)

| experiment | plan | seeds | val | geometry3k_test | pgdp5k_heldout | synth_extrap |
|---|---|---|---|---|---|---|
| probe | 7.1 linear probe | 3 | 0.600 ± 0.002 | 0.595 ± 0.013 | 0.517 ± 0.003 | 0.185 ± 0.013 |
| finetune | 7.2 fine-tune (Part A: 0 synthetic) | 3 | 0.876 ± 0.008 | 0.867 ± 0.007 | 0.869 ± 0.008 | 0.631 ± 0.017 |
| finetune_synth10k | 7.3 fine-tune + 10k synthetic (Part A: 10k) | 3 | 0.872 ± 0.009 | 0.871 ± 0.010 | 0.885 ± 0.004 | 0.713 ± 0.009 |
| ablation_res224 | 7.6 224 px | 1 | 0.865 | 0.855 | 0.861 | 0.609 |
| ablation_noaug | 7.7 no augmentation | 1 | 0.890 | 0.868 | 0.870 | 0.632 |

### Macro-F1 against asserted labels (mean ± sd over seeds; threshold 0.5)

| experiment | plan | seeds | val | geometry3k_test | pgdp5k_heldout | synth_extrap |
|---|---|---|---|---|---|---|
| probe | 7.1 linear probe | 3 | 0.472 ± 0.013 | 0.411 ± 0.012 | 0.359 ± 0.009 | 0.116 ± 0.010 |
| finetune | 7.2 fine-tune (Part A: 0 synthetic) | 3 | 0.629 ± 0.005 | 0.543 ± 0.007 | 0.546 ± 0.010 | 0.385 ± 0.015 |
| finetune_synth10k | 7.3 fine-tune + 10k synthetic (Part A: 10k) | 3 | 0.626 ± 0.002 | 0.542 ± 0.003 | 0.547 ± 0.003 | 0.403 ± 0.005 |
| ablation_res224 | 7.6 224 px | 1 | 0.624 | 0.527 | 0.535 | 0.382 |
| ablation_noaug | 7.7 no augmentation | 1 | 0.635 | 0.550 | 0.558 | 0.441 |
