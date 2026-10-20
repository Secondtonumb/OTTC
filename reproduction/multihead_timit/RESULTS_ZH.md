# TIMIT 四头结果

完成正式训练的配置：3/3。

| Objective | Head | 原生识别错误率 % | 边界 F1@20ms % |
|---|---|---:|---:|
| ctc | OTTC ARPAbet | 10.26 | 52.92 |
| ctc | IF-MDD ARPAbet | 9.00 | 53.05 |
| ctc | MFA IPA | 19.55 | 无参考边界 |
| ctc | ZIPA IPA127 | 9.49 | 无参考边界 |
| ce | OTTC ARPAbet | 9.91 | 63.72 |
| ce | IF-MDD ARPAbet | 9.83 | 63.26 |
| ce | MFA IPA | 11.25 | 无参考边界 |
| ce | ZIPA IPA127 | 12.97 | 无参考边界 |
| jsd_upstream | OTTC ARPAbet | 9.68 | 67.62 |
| jsd_upstream | IF-MDD ARPAbet | 9.54 | 67.08 |
| jsd_upstream | MFA IPA | 11.21 | 无参考边界 |
| jsd_upstream | ZIPA IPA127 | 19.91 | 无参考边界 |

ZIPA 的识别错误率按 token 计算。MFA 使用词典发音目标；两 IPA 头没有真实时间参考。
各头原生识别错误率的标签粒度不同，不能直接比较。测试集为 TIMIT core 192。

- ctc: https://wandb.ai/kevingenghaopeng-/ottc-english-multiphoneset/runs/7tp3slt5
- ce: https://wandb.ai/kevingenghaopeng-/ottc-english-multiphoneset/runs/ql9zhyav
- jsd_upstream: https://wandb.ai/kevingenghaopeng-/ottc-english-multiphoneset/runs/q0zlpgjh
