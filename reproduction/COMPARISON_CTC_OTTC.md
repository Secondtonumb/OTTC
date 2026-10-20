# CTC / OTTC + CE / OTTC + JSD

All models use the official LV60 base, TIMIT fused training data, seed 42,
40 epochs and final step 2920. OTTC learning rate is 2e-4, CTC 5e-5.
The upstream `jsd` is reverse-KL-to-the-mixture, not standard JSD.
Standard JSD was run separately as `jsd_standard`.
All metrics below are percentages. Inference receives audio only.
Boundary F1 uses 20ms one-to-one matching, argmax transitions including blank,
and no fitted shift. Overlap includes all reference duration in its denominator.
These boundary/overlap metrics do not reproduce the paper's F1/IDR definitions.
Single-seed results; cross-domain no-sil PER uses the shared normalization.

| Model | Test set | raw PER ↓ | no-sil PER ↓ | Boundary F1@20ms ↑ | Interval overlap ↑ |
|---|---|---:|---:|---:|---:|
| CTC | readme_kaldi_test_240 | 8.01 | 8.16 | 53.93 | 25.95 |
| CTC | timit_core_test_192 | 11.13 | 8.67 | 54.31 | 25.94 |
| CTC | buckeye_test | 52.16 | 26.98 | 47.01 | 17.44 |
| CTC | l2arctic_test | 40.85 | 23.38 | 47.55 | 18.55 |
| OTTC + CE | readme_kaldi_test_240 | 8.61 | 8.94 | 62.35 | 71.28 |
| OTTC + CE | timit_core_test_192 | 12.07 | 9.68 | 62.79 | 70.33 |
| OTTC + CE | buckeye_test | 55.42 | 28.66 | 51.96 | 45.19 |
| OTTC + CE | l2arctic_test | 41.97 | 24.31 | 61.38 | 54.17 |
| OTTC + upstream jsd | readme_kaldi_test_240 | 8.61 | 8.87 | 65.29 | 73.68 |
| OTTC + upstream jsd | timit_core_test_192 | 12.07 | 9.65 | 65.36 | 72.60 |
| OTTC + upstream jsd | buckeye_test | 55.66 | 28.60 | 54.40 | 47.01 |
| OTTC + upstream jsd | l2arctic_test | 41.67 | 23.69 | 64.57 | 56.48 |
| OTTC + standard JSD | readme_kaldi_test_240 | 15.53 | 17.20 | 61.53 | 70.14 |
| OTTC + standard JSD | timit_core_test_192 | 19.46 | 18.49 | 61.84 | 68.99 |
| OTTC + standard JSD | buckeye_test | 56.59 | 34.01 | 52.52 | 44.43 |
| OTTC + standard JSD | l2arctic_test | 43.11 | 29.82 | 62.20 | 52.85 |
