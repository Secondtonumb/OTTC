# Libri100：checkpoint 平均、PER / PanPhon PFER / K2 FA

PFER 是音频自由解码结果的 PanPhon 0.22.2 非加权特征编辑距离 / 参考 IPA 段数；展开双元音，移除 sil/q，dx→d。不使用 supplied-phone FA 的标签计算 PFER。PER 与原报告逐项一致。

K2 FA 提供参考音素序列，不提供边界时间；F1/R 是全量集合，MAE 仅计成功对齐，直接差异另附共同成功集合。Libri 边界来自已有自动对齐且身份未核验；Buckeye 为原人工 partial19 子集、zero-shot。

比较 Last、最后3个平均、dev FA最好3个平均、dev PER最好3个平均。dev只用 Libri dev-clean+dev-other、池化计数。可选权重仅为保留的5个（epoch39–40），不代表全程最优；完全相同的选中集合复用权重及推理。

完成全部平均策略的目标：CE, 官方 JSD, standard JSD
各策略完成数：last3 3/3, dev_fa_best3 3/3, dev_per_best3 3/3。

| 目标 | last3 steps | dev FA best3 steps | dev PER best3 steps |
|---|---|---|---|
| CE | 17600, 17800, 17840 | 17394, 17400, 17600 | 17394, 17400, 17600 |
| 官方 JSD | 17600, 17800, 17840 | 17394, 17400, 17600 | 17394, 17400, 17800 |
| standard JSD | 17600, 17800, 17840 | 17394, 17400, 17600 | 17394, 17400, 17800 |

同一组steps只推理一次；多个选择策略可能复用同一结果。

| 目标 | 保留的 step | pooled dev PER ↓ | pooled dev K2 FA F1@20 ↑ | dev FA MAE ms ↓ |
|---|---:|---:|---:|---:|
| CE | 17394 | 4.7331 | 60.2158 | 44.0207 |
| CE | 17400 | 4.7226 | 60.1676 | 43.5975 |
| CE | 17600 | 4.7301 | 60.1297 | 43.4571 |
| CE | 17800 | 4.7363 | 60.0573 | 43.7359 |
| CE | 17840 | 4.7372 | 60.0531 | 44.1025 |
| 官方 JSD | 17394 | 4.9709 | 62.4351 | 51.8639 |
| 官方 JSD | 17400 | 4.9617 | 62.4777 | 54.2241 |
| 官方 JSD | 17600 | 4.9997 | 62.3622 | 52.3249 |
| 官方 JSD | 17800 | 4.9701 | 62.1454 | 52.1887 |
| 官方 JSD | 17840 | 4.9731 | 62.1560 | 51.7797 |
| standard JSD | 17394 | 4.7749 | 61.1072 | 79.0648 |
| standard JSD | 17400 | 4.7787 | 61.1461 | 78.4924 |
| standard JSD | 17600 | 4.7814 | 61.2315 | 76.6936 |
| standard JSD | 17800 | 4.7771 | 61.0613 | 75.6187 |
| standard JSD | 17840 | 4.7846 | 61.0740 | 75.7807 |

| 目标 | 权重 | split | PER ↓ | PFER ↓ | FA F1@20 ↑ | FA R ↑ | FA MAE ms ↓ |
|---|---|---|---:|---:|---:|---:|---:|
| CE | last | dev_clean | 3.1072 | 1.8227 | 61.7381 | 67.6521 | 34.6818 |
| CE | last | dev_other | 6.5176 | 3.5608 | 58.2182 | 64.7842 | 54.3930 |
| CE | last | test_clean | 3.1559 | 1.9385 | 60.9509 | 67.0082 | 36.9528 |
| CE | last | test_other | 5.9864 | 3.2309 | 57.9459 | 64.5601 | 47.9843 |
| CE | last | buckeye_dev | 27.3416 | 11.5029 | 52.5662 | 59.7470 | 32.8781 |
| CE | last | buckeye_test | 31.4729 | 13.2497 | 47.9604 | 55.8947 | 37.1405 |
| CE | last3 | dev_clean | 3.1098 | 1.8251 | 61.7615 | 67.6714 | 34.7056 |
| CE | last3 | dev_other | 6.4989 | 3.5492 | 58.2088 | 64.7765 | 54.6655 |
| CE | last3 | test_clean | 3.1617 | 1.9443 | 60.9618 | 67.0171 | 37.4191 |
| CE | last3 | test_other | 5.9705 | 3.2236 | 57.9643 | 64.5751 | 47.8631 |
| CE | last3 | buckeye_dev | 27.3329 | 11.5027 | 52.6063 | 59.7809 | 32.8283 |
| CE | last3 | buckeye_test | 31.4425 | 13.2615 | 47.9954 | 55.9243 | 37.1025 |
| CE | dev_fa_best3 | dev_clean | 3.1217 | 1.8377 | 61.8744 | 67.7645 | 33.7126 |
| CE | dev_fa_best3 | dev_other | 6.4758 | 3.5320 | 58.3096 | 64.8591 | 54.2926 |
| CE | dev_fa_best3 | test_clean | 3.1670 | 1.9381 | 61.0782 | 67.1130 | 37.4026 |
| CE | dev_fa_best3 | test_other | 5.9611 | 3.2280 | 58.0752 | 64.6660 | 47.7194 |
| CE | dev_fa_best3 | buckeye_dev | 27.3242 | 11.4747 | 52.6197 | 59.7922 | 32.8334 |
| CE | dev_fa_best3 | buckeye_test | 31.5397 | 13.2643 | 48.0495 | 55.9699 | 37.1286 |
| CE | dev_per_best3 | dev_clean | 3.1217 | 1.8377 | 61.8744 | 67.7645 | 33.7126 |
| CE | dev_per_best3 | dev_other | 6.4758 | 3.5320 | 58.3096 | 64.8591 | 54.2926 |
| CE | dev_per_best3 | test_clean | 3.1670 | 1.9381 | 61.0782 | 67.1130 | 37.4026 |
| CE | dev_per_best3 | test_other | 5.9611 | 3.2280 | 58.0752 | 64.6660 | 47.7194 |
| CE | dev_per_best3 | buckeye_dev | 27.3242 | 11.4747 | 52.6197 | 59.7922 | 32.8334 |
| CE | dev_per_best3 | buckeye_test | 31.5397 | 13.2643 | 48.0495 | 55.9699 | 37.1286 |
| 官方 JSD | last | dev_clean | 3.2271 | 1.8935 | 63.8706 | 69.4100 | 42.4309 |
| 官方 JSD | last | dev_other | 6.8803 | 3.6929 | 60.2889 | 66.4810 | 61.9917 |
| 官方 JSD | last | test_clean | 3.3529 | 2.0265 | 62.9698 | 68.6703 | 49.8889 |
| 官方 JSD | last | test_other | 6.2227 | 3.3241 | 60.1273 | 66.3481 | 66.3141 |
| 官方 JSD | last | buckeye_dev | 26.7886 | 11.4472 | 54.4123 | 61.3064 | 30.0578 |
| 官方 JSD | last | buckeye_test | 31.1476 | 13.0220 | 49.6507 | 57.3203 | 35.9200 |
| 官方 JSD | last3 | dev_clean | 3.2121 | 1.8820 | 63.9449 | 69.4712 | 43.3299 |
| 官方 JSD | last3 | dev_other | 6.8966 | 3.7053 | 60.3553 | 66.5355 | 61.4444 |
| 官方 JSD | last3 | test_clean | 3.3466 | 2.0222 | 63.0399 | 68.7280 | 48.6500 |
| 官方 JSD | last3 | test_other | 6.2415 | 3.3330 | 60.2241 | 66.4274 | 65.4505 |
| 官方 JSD | last3 | buckeye_dev | 26.8082 | 11.4484 | 54.5082 | 61.3874 | 30.0460 |
| 官方 JSD | last3 | buckeye_test | 31.1506 | 13.0174 | 49.6857 | 57.3498 | 36.0737 |
| 官方 JSD | dev_fa_best3 | dev_clean | 3.2240 | 1.8916 | 64.2082 | 69.6882 | 43.8225 |
| 官方 JSD | dev_fa_best3 | dev_other | 6.8955 | 3.7036 | 60.4843 | 66.6411 | 65.3265 |
| 官方 JSD | dev_fa_best3 | test_clean | 3.3492 | 2.0230 | 63.2612 | 68.9101 | 47.7095 |
| 官方 JSD | dev_fa_best3 | test_other | 6.2475 | 3.3352 | 60.3522 | 66.5324 | 67.0363 |
| 官方 JSD | dev_fa_best3 | buckeye_dev | 26.7929 | 11.4308 | 54.5773 | 61.4457 | 30.1087 |
| 官方 JSD | dev_fa_best3 | buckeye_test | 31.2722 | 13.1199 | 49.7461 | 57.4008 | 35.8813 |
| 官方 JSD | dev_per_best3 | dev_clean | 3.2219 | 1.8886 | 64.1081 | 69.6057 | 43.8300 |
| 官方 JSD | dev_per_best3 | dev_other | 6.8780 | 3.6941 | 60.4135 | 66.5831 | 65.7865 |
| 官方 JSD | dev_per_best3 | test_clean | 3.3492 | 2.0263 | 63.1640 | 68.8302 | 47.7527 |
| 官方 JSD | dev_per_best3 | test_other | 6.2376 | 3.3258 | 60.2609 | 66.4575 | 67.0107 |
| 官方 JSD | dev_per_best3 | buckeye_dev | 26.7734 | 11.4034 | 54.5773 | 61.4457 | 30.0913 |
| 官方 JSD | dev_per_best3 | buckeye_test | 31.2449 | 13.0528 | 49.7366 | 57.3928 | 35.8442 |
| standard JSD | last | dev_clean | 3.1496 | 1.8532 | 62.9249 | 68.6305 | 69.7834 |
| standard JSD | last | dev_other | 6.5706 | 3.5148 | 59.0587 | 65.4731 | 82.3318 |
| standard JSD | last | test_clean | 3.2579 | 1.9754 | 61.8858 | 67.7780 | 84.2589 |
| standard JSD | last | test_other | 6.0509 | 3.2573 | 58.9404 | 65.3753 | 88.3524 |
| standard JSD | last | buckeye_dev | 26.4968 | 11.0976 | 54.3097 | 61.2197 | 30.0092 |
| standard JSD | last | buckeye_test | 29.9316 | 12.3571 | 49.6061 | 57.2827 | 35.4355 |
| standard JSD | last3 | dev_clean | 3.1573 | 1.8575 | 62.9706 | 68.6683 | 69.8023 |
| standard JSD | last3 | dev_other | 6.5683 | 3.5140 | 59.1024 | 65.5089 | 83.8309 |
| standard JSD | last3 | test_clean | 3.2552 | 1.9781 | 61.9336 | 67.8173 | 83.1918 |
| standard JSD | last3 | test_other | 6.0321 | 3.2444 | 58.9831 | 65.4103 | 88.7140 |
| standard JSD | last3 | buckeye_dev | 26.4990 | 11.0904 | 54.3432 | 61.2480 | 30.0027 |
| standard JSD | last3 | buckeye_test | 29.9681 | 12.3937 | 49.6252 | 57.2988 | 35.5503 |
| standard JSD | dev_per_best3 | dev_clean | 3.1568 | 1.8605 | 62.9615 | 68.6607 | 70.0536 |
| standard JSD | dev_per_best3 | dev_other | 6.5537 | 3.5186 | 59.0980 | 65.5053 | 86.0625 |
| standard JSD | dev_per_best3 | test_clean | 3.2483 | 1.9711 | 61.9253 | 67.8105 | 82.6507 |
| standard JSD | dev_per_best3 | test_other | 6.0481 | 3.2572 | 58.9701 | 65.3997 | 91.3630 |
| standard JSD | dev_per_best3 | buckeye_dev | 26.5230 | 11.1188 | 54.3454 | 61.2499 | 30.0031 |
| standard JSD | dev_per_best3 | buckeye_test | 29.9863 | 12.3551 | 49.5233 | 57.2129 | 38.2667 |
| standard JSD | dev_fa_best3 | dev_clean | 3.1553 | 1.8582 | 63.0169 | 68.7064 | 69.8536 |
| standard JSD | dev_fa_best3 | dev_other | 6.5345 | 3.5087 | 59.1816 | 65.5738 | 83.9929 |
| standard JSD | dev_fa_best3 | test_clean | 3.2478 | 1.9722 | 61.9674 | 67.8451 | 85.3269 |
| standard JSD | dev_fa_best3 | test_other | 6.0233 | 3.2405 | 59.0193 | 65.4400 | 91.1073 |
| standard JSD | dev_fa_best3 | buckeye_dev | 26.5251 | 11.1232 | 54.3677 | 61.2687 | 29.9599 |
| standard JSD | dev_fa_best3 | buckeye_test | 30.0106 | 12.3792 | 49.5774 | 57.2585 | 36.8649 |

PFER差异：负值较好。相对变化和配对置信区间分别列出；1000次utterance bootstrap，未处理同说话人相关性。

| 目标 / 权重 | 基线 | split | ΔPFER pp ↓ | 相对降低 % ↑ | 配对95% CI pp |
|---|---|---|---:|---:|---|
| CE / last3 | CE / last | dev_clean | +0.0024 | -0.13 | [-0.0078, +0.0124] |
| CE / last3 | CE / last | dev_other | -0.0116 | +0.33 | [-0.0301, +0.0059] |
| CE / last3 | CE / last | test_clean | +0.0057 | -0.29 | [-0.0065, +0.0182] |
| CE / last3 | CE / last | test_other | -0.0073 | +0.23 | [-0.0276, +0.0133] |
| CE / last3 | CE / last | buckeye_dev | -0.0002 | +0.00 | [-0.0292, +0.0275] |
| CE / last3 | CE / last | buckeye_test | +0.0118 | -0.09 | [-0.0379, +0.0591] |
| CE / dev_fa_best3 | CE / last | dev_clean | +0.0150 | -0.82 | [-0.0022, +0.0337] |
| CE / dev_fa_best3 | CE / last | dev_other | -0.0288 | +0.81 | [-0.0595, +0.0002] |
| CE / dev_fa_best3 | CE / last | test_clean | -0.0005 | +0.03 | [-0.0219, +0.0204] |
| CE / dev_fa_best3 | CE / last | test_other | -0.0029 | +0.09 | [-0.0330, +0.0261] |
| CE / dev_fa_best3 | CE / last | buckeye_dev | -0.0282 | +0.25 | [-0.0857, +0.0331] |
| CE / dev_fa_best3 | CE / last | buckeye_test | +0.0146 | -0.11 | [-0.0818, +0.0992] |
| CE / dev_fa_best3 | CE / last3 | dev_clean | +0.0125 | -0.69 | [-0.0029, +0.0294] |
| CE / dev_fa_best3 | CE / last3 | dev_other | -0.0172 | +0.48 | [-0.0465, +0.0103] |
| CE / dev_fa_best3 | CE / last3 | test_clean | -0.0062 | +0.32 | [-0.0252, +0.0133] |
| CE / dev_fa_best3 | CE / last3 | test_other | +0.0044 | -0.14 | [-0.0233, +0.0312] |
| CE / dev_fa_best3 | CE / last3 | buckeye_dev | -0.0280 | +0.24 | [-0.0899, +0.0309] |
| CE / dev_fa_best3 | CE / last3 | buckeye_test | +0.0028 | -0.02 | [-0.0886, +0.0896] |
| CE / dev_per_best3 | CE / last | dev_clean | +0.0150 | -0.82 | [-0.0022, +0.0337] |
| CE / dev_per_best3 | CE / last | dev_other | -0.0288 | +0.81 | [-0.0595, +0.0002] |
| CE / dev_per_best3 | CE / last | test_clean | -0.0005 | +0.03 | [-0.0219, +0.0204] |
| CE / dev_per_best3 | CE / last | test_other | -0.0029 | +0.09 | [-0.0330, +0.0261] |
| CE / dev_per_best3 | CE / last | buckeye_dev | -0.0282 | +0.25 | [-0.0857, +0.0331] |
| CE / dev_per_best3 | CE / last | buckeye_test | +0.0146 | -0.11 | [-0.0818, +0.0992] |
| CE / dev_per_best3 | CE / last3 | dev_clean | +0.0125 | -0.69 | [-0.0029, +0.0294] |
| CE / dev_per_best3 | CE / last3 | dev_other | -0.0172 | +0.48 | [-0.0465, +0.0103] |
| CE / dev_per_best3 | CE / last3 | test_clean | -0.0062 | +0.32 | [-0.0252, +0.0133] |
| CE / dev_per_best3 | CE / last3 | test_other | +0.0044 | -0.14 | [-0.0233, +0.0312] |
| CE / dev_per_best3 | CE / last3 | buckeye_dev | -0.0280 | +0.24 | [-0.0899, +0.0309] |
| CE / dev_per_best3 | CE / last3 | buckeye_test | +0.0028 | -0.02 | [-0.0886, +0.0896] |
| 官方 JSD / last | CE / last | dev_clean | +0.0708 | -3.88 | [+0.0073, +0.1295] |
| 官方 JSD / last | CE / last | dev_other | +0.1321 | -3.71 | [+0.0281, +0.2389] |
| 官方 JSD / last | CE / last | test_clean | +0.0880 | -4.54 | [+0.0256, +0.1554] |
| 官方 JSD / last | CE / last | test_other | +0.0933 | -2.89 | [-0.0151, +0.1885] |
| 官方 JSD / last | CE / last | buckeye_dev | -0.0557 | +0.48 | [-0.2483, +0.1543] |
| 官方 JSD / last | CE / last | buckeye_test | -0.2277 | +1.72 | [-0.5742, +0.0923] |
| 官方 JSD / last3 | 官方 JSD / last | dev_clean | -0.0115 | +0.61 | [-0.0270, +0.0029] |
| 官方 JSD / last3 | 官方 JSD / last | dev_other | +0.0124 | -0.34 | [-0.0094, +0.0321] |
| 官方 JSD / last3 | 官方 JSD / last | test_clean | -0.0043 | +0.21 | [-0.0184, +0.0097] |
| 官方 JSD / last3 | 官方 JSD / last | test_other | +0.0088 | -0.27 | [-0.0086, +0.0277] |
| 官方 JSD / last3 | 官方 JSD / last | buckeye_dev | +0.0013 | -0.01 | [-0.0243, +0.0305] |
| 官方 JSD / last3 | 官方 JSD / last | buckeye_test | -0.0046 | +0.04 | [-0.0640, +0.0558] |
| 官方 JSD / last3 | CE / last3 | dev_clean | +0.0569 | -3.12 | [-0.0049, +0.1136] |
| 官方 JSD / last3 | CE / last3 | dev_other | +0.1561 | -4.40 | [+0.0523, +0.2640] |
| 官方 JSD / last3 | CE / last3 | test_clean | +0.0780 | -4.01 | [+0.0147, +0.1450] |
| 官方 JSD / last3 | CE / last3 | test_other | +0.1094 | -3.39 | [+0.0050, +0.2060] |
| 官方 JSD / last3 | CE / last3 | buckeye_dev | -0.0543 | +0.47 | [-0.2484, +0.1599] |
| 官方 JSD / last3 | CE / last3 | buckeye_test | -0.2441 | +1.84 | [-0.5905, +0.0880] |
| 官方 JSD / dev_fa_best3 | 官方 JSD / last | dev_clean | -0.0019 | +0.10 | [-0.0250, +0.0192] |
| 官方 JSD / dev_fa_best3 | 官方 JSD / last | dev_other | +0.0108 | -0.29 | [-0.0216, +0.0411] |
| 官方 JSD / dev_fa_best3 | 官方 JSD / last | test_clean | -0.0035 | +0.17 | [-0.0250, +0.0178] |
| 官方 JSD / dev_fa_best3 | 官方 JSD / last | test_other | +0.0111 | -0.33 | [-0.0167, +0.0366] |
| 官方 JSD / dev_fa_best3 | 官方 JSD / last | buckeye_dev | -0.0163 | +0.14 | [-0.0680, +0.0345] |
| 官方 JSD / dev_fa_best3 | 官方 JSD / last | buckeye_test | +0.0979 | -0.75 | [+0.0118, +0.1871] |
| 官方 JSD / dev_fa_best3 | 官方 JSD / last3 | dev_clean | +0.0096 | -0.51 | [-0.0082, +0.0265] |
| 官方 JSD / dev_fa_best3 | 官方 JSD / last3 | dev_other | -0.0016 | +0.04 | [-0.0309, +0.0270] |
| 官方 JSD / dev_fa_best3 | 官方 JSD / last3 | test_clean | +0.0008 | -0.04 | [-0.0183, +0.0177] |
| 官方 JSD / dev_fa_best3 | 官方 JSD / last3 | test_other | +0.0022 | -0.07 | [-0.0267, +0.0287] |
| 官方 JSD / dev_fa_best3 | 官方 JSD / last3 | buckeye_dev | -0.0176 | +0.15 | [-0.0709, +0.0316] |
| 官方 JSD / dev_fa_best3 | 官方 JSD / last3 | buckeye_test | +0.1025 | -0.79 | [+0.0188, +0.1886] |
| 官方 JSD / dev_fa_best3 | CE / dev_fa_best3 | dev_clean | +0.0539 | -2.94 | [-0.0068, +0.1075] |
| 官方 JSD / dev_fa_best3 | CE / dev_fa_best3 | dev_other | +0.1717 | -4.86 | [+0.0655, +0.2810] |
| 官方 JSD / dev_fa_best3 | CE / dev_fa_best3 | test_clean | +0.0849 | -4.38 | [+0.0209, +0.1513] |
| 官方 JSD / dev_fa_best3 | CE / dev_fa_best3 | test_other | +0.1072 | -3.32 | [+0.0007, +0.2036] |
| 官方 JSD / dev_fa_best3 | CE / dev_fa_best3 | buckeye_dev | -0.0439 | +0.38 | [-0.2423, +0.1712] |
| 官方 JSD / dev_fa_best3 | CE / dev_fa_best3 | buckeye_test | -0.1444 | +1.09 | [-0.4954, +0.1668] |
| 官方 JSD / dev_per_best3 | 官方 JSD / last | dev_clean | -0.0049 | +0.26 | [-0.0238, +0.0138] |
| 官方 JSD / dev_per_best3 | 官方 JSD / last | dev_other | +0.0013 | -0.03 | [-0.0290, +0.0318] |
| 官方 JSD / dev_per_best3 | 官方 JSD / last | test_clean | -0.0002 | +0.01 | [-0.0216, +0.0195] |
| 官方 JSD / dev_per_best3 | 官方 JSD / last | test_other | +0.0017 | -0.05 | [-0.0244, +0.0270] |
| 官方 JSD / dev_per_best3 | 官方 JSD / last | buckeye_dev | -0.0437 | +0.38 | [-0.0976, +0.0049] |
| 官方 JSD / dev_per_best3 | 官方 JSD / last | buckeye_test | +0.0308 | -0.24 | [-0.0586, +0.1202] |
| 官方 JSD / dev_per_best3 | 官方 JSD / last3 | dev_clean | +0.0066 | -0.35 | [-0.0099, +0.0235] |
| 官方 JSD / dev_per_best3 | 官方 JSD / last3 | dev_other | -0.0111 | +0.30 | [-0.0393, +0.0183] |
| 官方 JSD / dev_per_best3 | 官方 JSD / last3 | test_clean | +0.0041 | -0.20 | [-0.0152, +0.0223] |
| 官方 JSD / dev_per_best3 | 官方 JSD / last3 | test_other | -0.0071 | +0.21 | [-0.0318, +0.0206] |
| 官方 JSD / dev_per_best3 | 官方 JSD / last3 | buckeye_dev | -0.0450 | +0.39 | [-0.0999, +0.0046] |
| 官方 JSD / dev_per_best3 | 官方 JSD / last3 | buckeye_test | +0.0354 | -0.27 | [-0.0539, +0.1251] |
| 官方 JSD / dev_per_best3 | CE / dev_per_best3 | dev_clean | +0.0509 | -2.77 | [-0.0100, +0.1074] |
| 官方 JSD / dev_per_best3 | CE / dev_per_best3 | dev_other | +0.1622 | -4.59 | [+0.0598, +0.2685] |
| 官方 JSD / dev_per_best3 | CE / dev_per_best3 | test_clean | +0.0882 | -4.55 | [+0.0264, +0.1522] |
| 官方 JSD / dev_per_best3 | CE / dev_per_best3 | test_other | +0.0978 | -3.03 | [-0.0107, +0.1948] |
| 官方 JSD / dev_per_best3 | CE / dev_per_best3 | buckeye_dev | -0.0713 | +0.62 | [-0.2658, +0.1439] |
| 官方 JSD / dev_per_best3 | CE / dev_per_best3 | buckeye_test | -0.2115 | +1.59 | [-0.5523, +0.1088] |
| standard JSD / last | CE / last | dev_clean | +0.0305 | -1.67 | [-0.0239, +0.0876] |
| standard JSD / last | CE / last | dev_other | -0.0459 | +1.29 | [-0.1456, +0.0490] |
| standard JSD / last | CE / last | test_clean | +0.0369 | -1.90 | [-0.0169, +0.0902] |
| standard JSD / last | CE / last | test_other | +0.0265 | -0.82 | [-0.0828, +0.1286] |
| standard JSD / last | CE / last | buckeye_dev | -0.4053 | +3.52 | [-0.5989, -0.1964] |
| standard JSD / last | CE / last | buckeye_test | -0.8926 | +6.74 | [-1.1851, -0.6034] |
| standard JSD / last3 | standard JSD / last | dev_clean | +0.0043 | -0.23 | [-0.0107, +0.0178] |
| standard JSD / last3 | standard JSD / last | dev_other | -0.0008 | +0.02 | [-0.0161, +0.0157] |
| standard JSD / last3 | standard JSD / last | test_clean | +0.0027 | -0.13 | [-0.0082, +0.0145] |
| standard JSD / last3 | standard JSD / last | test_other | -0.0130 | +0.40 | [-0.0279, +0.0009] |
| standard JSD / last3 | standard JSD / last | buckeye_dev | -0.0072 | +0.07 | [-0.0328, +0.0166] |
| standard JSD / last3 | standard JSD / last | buckeye_test | +0.0366 | -0.30 | [-0.0031, +0.0830] |
| standard JSD / last3 | CE / last3 | dev_clean | +0.0323 | -1.77 | [-0.0246, +0.0892] |
| standard JSD / last3 | CE / last3 | dev_other | -0.0351 | +0.99 | [-0.1409, +0.0583] |
| standard JSD / last3 | CE / last3 | test_clean | +0.0338 | -1.74 | [-0.0210, +0.0875] |
| standard JSD / last3 | CE / last3 | test_other | +0.0208 | -0.65 | [-0.0841, +0.1187] |
| standard JSD / last3 | CE / last3 | buckeye_dev | -0.4124 | +3.58 | [-0.5950, -0.2071] |
| standard JSD / last3 | CE / last3 | buckeye_test | -0.8678 | +6.54 | [-1.1597, -0.5805] |
| standard JSD / dev_per_best3 | standard JSD / last | dev_clean | +0.0073 | -0.39 | [-0.0126, +0.0284] |
| standard JSD / dev_per_best3 | standard JSD / last | dev_other | +0.0038 | -0.11 | [-0.0240, +0.0326] |
| standard JSD / dev_per_best3 | standard JSD / last | test_clean | -0.0043 | +0.22 | [-0.0257, +0.0168] |
| standard JSD / dev_per_best3 | standard JSD / last | test_other | -0.0001 | +0.00 | [-0.0273, +0.0261] |
| standard JSD / dev_per_best3 | standard JSD / last | buckeye_dev | +0.0212 | -0.19 | [-0.0259, +0.0680] |
| standard JSD / dev_per_best3 | standard JSD / last | buckeye_test | -0.0019 | +0.02 | [-0.0628, +0.0561] |
| standard JSD / dev_per_best3 | standard JSD / last3 | dev_clean | +0.0031 | -0.16 | [-0.0180, +0.0246] |
| standard JSD / dev_per_best3 | standard JSD / last3 | dev_other | +0.0046 | -0.13 | [-0.0219, +0.0333] |
| standard JSD / dev_per_best3 | standard JSD / last3 | test_clean | -0.0069 | +0.35 | [-0.0293, +0.0153] |
| standard JSD / dev_per_best3 | standard JSD / last3 | test_other | +0.0129 | -0.40 | [-0.0127, +0.0379] |
| standard JSD / dev_per_best3 | standard JSD / last3 | buckeye_dev | +0.0284 | -0.26 | [-0.0167, +0.0718] |
| standard JSD / dev_per_best3 | standard JSD / last3 | buckeye_test | -0.0386 | +0.31 | [-0.1001, +0.0188] |
| standard JSD / dev_per_best3 | CE / dev_per_best3 | dev_clean | +0.0229 | -1.24 | [-0.0351, +0.0802] |
| standard JSD / dev_per_best3 | CE / dev_per_best3 | dev_other | -0.0133 | +0.38 | [-0.1149, +0.0807] |
| standard JSD / dev_per_best3 | CE / dev_per_best3 | test_clean | +0.0331 | -1.71 | [-0.0246, +0.0887] |
| standard JSD / dev_per_best3 | CE / dev_per_best3 | test_other | +0.0292 | -0.91 | [-0.0769, +0.1261] |
| standard JSD / dev_per_best3 | CE / dev_per_best3 | buckeye_dev | -0.3559 | +3.10 | [-0.5443, -0.1516] |
| standard JSD / dev_per_best3 | CE / dev_per_best3 | buckeye_test | -0.9092 | +6.85 | [-1.2179, -0.6199] |
| standard JSD / dev_fa_best3 | standard JSD / last | dev_clean | +0.0050 | -0.27 | [-0.0165, +0.0269] |
| standard JSD / dev_fa_best3 | standard JSD / last | dev_other | -0.0062 | +0.18 | [-0.0343, +0.0248] |
| standard JSD / dev_fa_best3 | standard JSD / last | test_clean | -0.0032 | +0.16 | [-0.0248, +0.0196] |
| standard JSD / dev_fa_best3 | standard JSD / last | test_other | -0.0169 | +0.52 | [-0.0451, +0.0095] |
| standard JSD / dev_fa_best3 | standard JSD / last | buckeye_dev | +0.0256 | -0.23 | [-0.0248, +0.0748] |
| standard JSD / dev_fa_best3 | standard JSD / last | buckeye_test | +0.0222 | -0.18 | [-0.0434, +0.0913] |
| standard JSD / dev_fa_best3 | standard JSD / last3 | dev_clean | +0.0007 | -0.04 | [-0.0211, +0.0226] |
| standard JSD / dev_fa_best3 | standard JSD / last3 | dev_other | -0.0054 | +0.15 | [-0.0314, +0.0220] |
| standard JSD / dev_fa_best3 | standard JSD / last3 | test_clean | -0.0059 | +0.30 | [-0.0287, +0.0168] |
| standard JSD / dev_fa_best3 | standard JSD / last3 | test_other | -0.0039 | +0.12 | [-0.0311, +0.0219] |
| standard JSD / dev_fa_best3 | standard JSD / last3 | buckeye_dev | +0.0329 | -0.30 | [-0.0140, +0.0809] |
| standard JSD / dev_fa_best3 | standard JSD / last3 | buckeye_test | -0.0145 | +0.12 | [-0.0808, +0.0481] |
| standard JSD / dev_fa_best3 | CE / dev_fa_best3 | dev_clean | +0.0205 | -1.12 | [-0.0378, +0.0778] |
| standard JSD / dev_fa_best3 | CE / dev_fa_best3 | dev_other | -0.0233 | +0.66 | [-0.1208, +0.0688] |
| standard JSD / dev_fa_best3 | CE / dev_fa_best3 | test_clean | +0.0341 | -1.76 | [-0.0235, +0.0913] |
| standard JSD / dev_fa_best3 | CE / dev_fa_best3 | test_other | +0.0125 | -0.39 | [-0.0922, +0.1072] |
| standard JSD / dev_fa_best3 | CE / dev_fa_best3 | buckeye_dev | -0.3515 | +3.06 | [-0.5535, -0.1480] |
| standard JSD / dev_fa_best3 | CE / dev_fa_best3 | buckeye_test | -0.8851 | +6.67 | [-1.1945, -0.5917] |

Last结果：官方JSD在Libri test-clean / other 的PFER略高于CE；Buckeye降低0.228pp（相对1.72%），配对CI跨0，目前不能支持“大幅提升”。standard JSD在Buckeye降低0.893pp（相对6.74%），CI不跨0；Libri两个test仍未优于CE。

所有三个目标的替换错误多数属于特征接近的音素。特征距离、S/I/D和预测/参考IPA段数比保存于 pfer/{objective}/{artifact}/metrics.json，逐句成本保存于 *_per_utterance.jsonl。PFER改善不能仅由它数值小于PER来判断。
