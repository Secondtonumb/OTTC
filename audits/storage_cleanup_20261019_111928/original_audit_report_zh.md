# OTTC checkpoint 只读清理审计

审计时间：2026-10-19T11:12:51.616275+09:00
范围：/work/gm64/m64000/OTTC
仅检查本仓库；没有删除、移动、覆盖或修改实验文件；没有扫描其他项目。

## 可供批准的精确文件组

| 组 | 内容 | 文件数 | 可释放 GiB | 风险 |
|---|---|---:|---:|---|
| A1 | 已完成 WavLM 12组的早期 epoch 权重 | 262 | 615.91 | 中 |
| A2 | Zapformer 六步验证 checkpoint（含960秒batch试验） | 60 | 204.81 | 低 |
| A3 | 本次已完成 TIMIT10ms 四组的早期 epoch 权重 | 77 | 90.51 | 中 |
| A4 | Libri960 三组四步 smoke 的模型/optimizer/scheduler | 48 | 84.25 | 低 |
| A5 | 本次六组 TIMIT10ms 的 profile_weights.pt | 6 | 7.05 | 低 |
| B1 | 旧 NCCL core dump（额外选项） | 1 | 4.85 | 低 |

合计 1007.37 GiB（0.984 TiB）。
建议第一批只选 A2+A4+A5，共 296.11 GiB：全部是验证/性能测试产物，保留所有正式模型。
全部候选实际 GID 均为22753/gm64；组配额当前超限约 0.097 GiB。按此快照，第一批清理后约 14.711 TiB / 15 TiB；其他作业写入会影响实际值。

## 保留规则

- 所有原始SSL权重、数据集、旧正式模型和源代码快照。
- 已完成40轮实验的epoch31–40全部独立权重、best、final、final5/final10平均模型及完整最新resume。
- 额外保留epoch1/5/10/15/20/25/30，以及最佳dev PER、dev loss、F1、R、FA MAE对应的原始epoch。
- 已完成10轮迁移实验的全部10轮权重，因为它们都是final10成员。
- 两组仅完成epoch1的S3T2实验全部权重和残留checkpoint，恢复方案未决。
- 全部配置、loss/metric日志、预测、图表、报告、W&B记录和验证PASS/RESULT证据。

## 风险与证据

- 没有核实任何候选的独立备份；若批准，删除将绕过回收站。验证模型可按原配方重做；被删的早期正式epoch不能从平均权重精确恢复。
- 队列查询未见本用户未完成PBS作业；可检查的本地进程未引用候选。4个凭据/SSH会话进程受内核保护，无法查看其文件描述符；执行前必须再次核对。
- 339个早期正式epoch候选已作仓库文本引用扫描，0命中；模型目录内部metadata所引用成员已加入保留集。
- A2依据ALL_ARMS_PASS.json及validation_960/RESULT.json：四组各连续/恢复6步验证，960试验未正式训练。
- A4依据每组VALIDATION_COMPLETE.json：仅4次optimizer更新，optimizer/RNG恢复验证已通过。
- A5依据profile_recipe.py：profile_weights.pt明确标记profile_only，正式训练重新从原始SSL初始化。

完整逐文件清单（真实GID、大小、inode、mtime及原因）：/tmp/ottc_checkpoint_audit_20261019/MANIFEST.json
英文详细路径/保留轮次：/tmp/ottc_checkpoint_audit_20261019/REPORT.md

## A1 / A3 每组保留与候选轮次

### A1: wavlm__timit__10ms__mono__jsd_standard
路径：/work/gm64/m64000/OTTC/reproduction/native_stride_ottc_20261018/models/wavlm__timit__10ms__mono__jsd_standard/epoch_weights/
保留：[1, 5, 10, 15, 20, 25, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40]
候选：[2, 3, 4, 6, 7, 8, 9, 11, 12, 13, 14, 16, 17, 18, 19, 21, 22, 23, 24, 26, 27, 28, 29]

### A1: wavlm__buckeye__10ms__mono__ce
路径：/work/gm64/m64000/OTTC/reproduction/native_stride_ottc_20261018/models/wavlm__buckeye__10ms__mono__ce/epoch_weights/
保留：[1, 5, 10, 12, 15, 20, 25, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40]
候选：[2, 3, 4, 6, 7, 8, 9, 11, 13, 14, 16, 17, 18, 19, 21, 22, 23, 24, 26, 27, 28, 29]

### A1: wavlm__buckeye__10ms__mono__jsd_upstream
路径：/work/gm64/m64000/OTTC/reproduction/native_stride_ottc_20261018/models/wavlm__buckeye__10ms__mono__jsd_upstream/epoch_weights/
保留：[1, 5, 10, 15, 20, 22, 25, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40]
候选：[2, 3, 4, 6, 7, 8, 9, 11, 12, 13, 14, 16, 17, 18, 19, 21, 23, 24, 26, 27, 28, 29]

### A1: wavlm__buckeye__20ms__mono__jsd_standard
路径：/work/gm64/m64000/OTTC/reproduction/native_stride_ottc_20261018/models/wavlm__buckeye__20ms__mono__jsd_standard/epoch_weights/
保留：[1, 5, 10, 15, 20, 21, 25, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40]
候选：[2, 3, 4, 6, 7, 8, 9, 11, 12, 13, 14, 16, 17, 18, 19, 22, 23, 24, 26, 27, 28, 29]

### A1: wavlm__timit__20ms__mono__ce
路径：/work/gm64/m64000/OTTC/reproduction/native_stride_ottc_20261018/models/wavlm__timit__20ms__mono__ce/epoch_weights/
保留：[1, 5, 10, 11, 15, 20, 25, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40]
候选：[2, 3, 4, 6, 7, 8, 9, 12, 13, 14, 16, 17, 18, 19, 21, 22, 23, 24, 26, 27, 28, 29]

### A1: wavlm__timit__20ms__mono__jsd_upstream
路径：/work/gm64/m64000/OTTC/reproduction/native_stride_ottc_20261018/models/wavlm__timit__20ms__mono__jsd_upstream/epoch_weights/
保留：[1, 5, 10, 15, 20, 21, 25, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40]
候选：[2, 3, 4, 6, 7, 8, 9, 11, 12, 13, 14, 16, 17, 18, 19, 22, 23, 24, 26, 27, 28, 29]

### A1: wavlm__buckeye__10ms__mono__jsd_standard
路径：/work/gm64/m64000/OTTC/reproduction/native_stride_ottc_20261018/models/wavlm__buckeye__10ms__mono__jsd_standard/epoch_weights/
保留：[1, 5, 10, 15, 20, 25, 26, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40]
候选：[2, 3, 4, 6, 7, 8, 9, 11, 12, 13, 14, 16, 17, 18, 19, 21, 22, 23, 24, 27, 28]

### A1: wavlm__buckeye__20ms__mono__jsd_upstream
路径：/work/gm64/m64000/OTTC/reproduction/native_stride_ottc_20261018/models/wavlm__buckeye__20ms__mono__jsd_upstream/epoch_weights/
保留：[1, 5, 10, 15, 20, 25, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40]
候选：[2, 3, 4, 6, 7, 8, 9, 11, 12, 13, 14, 16, 17, 18, 19, 21, 22, 23, 24, 26, 27]

### A1: wavlm__timit__20ms__mono__jsd_standard
路径：/work/gm64/m64000/OTTC/reproduction/native_stride_ottc_20261018/models/wavlm__timit__20ms__mono__jsd_standard/epoch_weights/
保留：[1, 5, 10, 15, 20, 25, 28, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40]
候选：[2, 3, 4, 6, 7, 8, 9, 11, 12, 13, 14, 16, 17, 18, 19, 21, 22, 23, 24, 26, 27, 29]

### A1: wavlm__timit__10ms__mono__ce
路径：/work/gm64/m64000/OTTC/reproduction/native_stride_ottc_20261018/models/wavlm__timit__10ms__mono__ce/epoch_weights/
保留：[1, 5, 10, 11, 15, 20, 25, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40]
候选：[2, 3, 4, 6, 7, 8, 9, 12, 13, 14, 16, 17, 18, 19, 21, 22, 23, 24, 26, 27, 28, 29]

### A1: wavlm__timit__10ms__mono__jsd_upstream
路径：/work/gm64/m64000/OTTC/reproduction/native_stride_ottc_20261018/models/wavlm__timit__10ms__mono__jsd_upstream/epoch_weights/
保留：[1, 5, 10, 15, 20, 25, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40]
候选：[2, 3, 4, 6, 7, 8, 9, 11, 12, 13, 14, 16, 17, 18, 19, 21, 22, 23, 24, 26, 27, 28, 29]

### A1: wavlm__buckeye__20ms__mono__ce
路径：/work/gm64/m64000/OTTC/reproduction/native_stride_ottc_20261018/models/wavlm__buckeye__20ms__mono__ce/epoch_weights/
保留：[1, 5, 6, 10, 11, 12, 15, 20, 25, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40]
候选：[2, 3, 4, 7, 8, 9, 13, 14, 16, 17, 18, 19, 21, 22, 23, 24, 26, 27, 28, 29]

### A3: wav2vec2__timit__10ms__mono__jsd_upstream
路径：/work/gm64/m64000/OTTC/reproduction/timit_ssl_native10_20261019/models/wav2vec2__timit__10ms__mono__jsd_upstream/epoch_weights/
保留：[1, 5, 10, 11, 12, 15, 20, 25, 28, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40]
候选：[2, 3, 4, 6, 7, 8, 9, 13, 14, 16, 17, 18, 19, 21, 22, 23, 24, 26, 27, 29]

### A3: wav2vec2__timit__10ms__mono__jsd_standard
路径：/work/gm64/m64000/OTTC/reproduction/timit_ssl_native10_20261019/models/wav2vec2__timit__10ms__mono__jsd_standard/epoch_weights/
保留：[1, 5, 10, 15, 16, 20, 23, 24, 25, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40]
候选：[2, 3, 4, 6, 7, 8, 9, 11, 12, 13, 14, 17, 18, 19, 21, 22, 26, 27, 28, 29]

### A3: wav2vec2__timit__10ms__mono__ce
路径：/work/gm64/m64000/OTTC/reproduction/timit_ssl_native10_20261019/models/wav2vec2__timit__10ms__mono__ce/epoch_weights/
保留：[1, 5, 7, 8, 10, 15, 19, 20, 21, 25, 28, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40]
候选：[2, 3, 4, 6, 9, 11, 12, 13, 14, 16, 17, 18, 22, 23, 24, 26, 27, 29]

### A3: wav2vec2__timit__10ms__s3t2ss__jsd_upstream
路径：/work/gm64/m64000/OTTC/reproduction/timit_ssl_native10_20261019/models/wav2vec2__timit__10ms__s3t2ss__jsd_upstream/epoch_weights/
保留：[1, 5, 6, 7, 10, 13, 15, 16, 20, 25, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40]
候选：[2, 3, 4, 8, 9, 11, 12, 14, 17, 18, 19, 21, 22, 23, 24, 26, 27, 28, 29]

删除尚未获批。需要用户明确指定组号；获批后重新生成新鲜清单并核对进程、队列、保留集和文件身份，随后才执行。
