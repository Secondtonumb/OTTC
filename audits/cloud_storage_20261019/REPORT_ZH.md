# OTTC 云端评估 artifact 清理审计（2026-10-19）

状态：用户明确批准后，73 个版本全部删除成功；已通过云端逐版本验收，并对保留的 546 个本地文件再次校验 MD5。其余 188 个原有版本均核实仍在。

范围仅限 kevingenghaopeng- 的 ottc-timit-reproduction 和 ottc-english-multiphoneset 项目中的 evaluation/evaluation-summary 类型；不代表整个组织全部存储。

容量是 artifact 版本逻辑大小之和，实际配额释放受云端内容去重和刷新延迟影响。

| 项目 | 审计版本数 | 逻辑容量 GB | 候选删除版本数 | 候选逻辑容量 GB | 保留逻辑容量 GB |
|---|---:|---:|---:|---:|---:|
| ottc-english-multiphoneset | 32 | 1.741309 | 9 | 1.330784 | 0.410525 |
| ottc-timit-reproduction | 229 | 2.945688 | 64 | 2.806770 | 0.138918 |

## 校验与保留

候选文件均逐文件通过大小和 base64 MD5 与云端 manifest 对比；没有引用文件，没有 W&B 下游消费者，来源运行均为 finished。候选仅含 JSON、JSONL、Python、Markdown 和 CSV，无模型权重。删除前会再次核实版本 ID、运行状态、消费者和所有本地文件哈希。

本地模型、预测、指标、评估代码、配置、训练恢复文件，以及 W&B runs/history/summary/report 均不在删除范围。云端 artifact 删除不可逆，恢复须从保留的本地文件重新上传；原云端版本 ID 不保证可恢复。

保留所有小于 4 MB 的版本，以及无法找到完整一致本地备份的较大版本。

## 已删除的精确清单（73 个版本）

| 项目 | 精确版本 | MB | 本地备份目录 |
|---|---|---:|---|
| ottc-timit-reproduction | libri960-continue5-jsd_upstream-K2FA-average_last5:v0 | 98.162 | /work/gm64/m64000/OTTC/reproduction/libri960_continue5_20261017/evaluation/jsd_upstream/average_last5 |
| ottc-timit-reproduction | libri960-continue5-jsd_standard-K2FA-average_last5:v0 | 98.176 | /work/gm64/m64000/OTTC/reproduction/libri960_continue5_20261017/evaluation/jsd_standard/average_last5 |
| ottc-timit-reproduction | libri960-continue5-ce-K2FA-average_last5:v0 | 98.180 | /work/gm64/m64000/OTTC/reproduction/libri960_continue5_20261017/evaluation/ce/average_last5 |
| ottc-timit-reproduction | libri960-continue5-jsd_upstream-K2FA-best_dev:v0 | 98.164 | /work/gm64/m64000/OTTC/reproduction/libri960_continue5_20261017/evaluation/jsd_upstream/best_dev |
| ottc-timit-reproduction | libri960-continue5-jsd_standard-K2FA-best_dev:v0 | 98.175 | /work/gm64/m64000/OTTC/reproduction/libri960_continue5_20261017/evaluation/jsd_standard/best_dev |
| ottc-timit-reproduction | libri960-continue5-ce-K2FA-best_dev:v0 | 98.205 | /work/gm64/m64000/OTTC/reproduction/libri960_continue5_20261017/evaluation/ce/best_dev |
| ottc-timit-reproduction | libri960-continue5-jsd_upstream-K2FA-final:v0 | 98.167 | /work/gm64/m64000/OTTC/reproduction/libri960_continue5_20261017/evaluation/jsd_upstream/final |
| ottc-timit-reproduction | libri960-continue5-jsd_standard-K2FA-final:v0 | 98.180 | /work/gm64/m64000/OTTC/reproduction/libri960_continue5_20261017/evaluation/jsd_standard/final |
| ottc-timit-reproduction | libri960-continue5-ce-K2FA-final:v0 | 98.161 | /work/gm64/m64000/OTTC/reproduction/libri960_continue5_20261017/evaluation/ce/final |
| ottc-timit-reproduction | libri100_ctc_0p5_q1_seed42-summaries:v0 | 13.188 | /work/gm64/m64000/OTTC/reproduction/cif_quantity_20261016/runs/libri100_ctc_0p5_q1_seed42 |
| ottc-timit-reproduction | libri100_ce_q1_seed42-summaries:v0 | 12.970 | /work/gm64/m64000/OTTC/reproduction/cif_quantity_20261016/runs/libri100_ce_q1_seed42 |
| ottc-timit-reproduction | buckeye_ctc_0p5_q1_seed42-summaries:v0 | 5.154 | /work/gm64/m64000/OTTC/reproduction/cif_quantity_20261016/runs/buckeye_ctc_0p5_q1_seed42 |
| ottc-timit-reproduction | buckeye_ottc_jsd_0p5_q1_seed42-summaries:v0 | 5.275 | /work/gm64/m64000/OTTC/reproduction/cif_quantity_20261016/runs/buckeye_ottc_jsd_0p5_q1_seed42 |
| ottc-timit-reproduction | buckeye_ce_q1_seed42-summaries:v0 | 5.075 | /work/gm64/m64000/OTTC/reproduction/cif_quantity_20261016/runs/buckeye_ce_q1_seed42 |
| ottc-timit-reproduction | libri100_ottc_jsd_0p5_q1_seed42-summaries:v1 | 13.481 | /work/gm64/m64000/OTTC/reproduction/cif_quantity_20261016/runs/libri100_ottc_jsd_0p5_q1_seed42 |
| ottc-timit-reproduction | libri100-jsd_standard-avg-17394_17400_17600-PFER:v0 | 4.536 | /work/gm64/m64000/OTTC/reproduction/libri100_singlehead_average_20261016/pfer/jsd_standard/17394_17400_17600 |
| ottc-timit-reproduction | libri100-jsd_standard-avg-17394_17400_17800-PFER:v0 | 4.536 | /work/gm64/m64000/OTTC/reproduction/libri100_singlehead_average_20261016/pfer/jsd_standard/17394_17400_17800 |
| ottc-timit-reproduction | libri100-singlehead-jsd_standard-avg-17394_17400_17600-K2FA:v0 | 98.172 | /work/gm64/m64000/OTTC/reproduction/libri100_singlehead_average_20261016/results/jsd_standard/evaluation/17394_17400_17600 |
| ottc-timit-reproduction | libri100-singlehead-jsd_standard-avg-17394_17400_17800-K2FA:v0 | 98.171 | /work/gm64/m64000/OTTC/reproduction/libri100_singlehead_average_20261016/results/jsd_standard/evaluation/17394_17400_17800 |
| ottc-timit-reproduction | libri100-singlehead-jsd_standard-avg-17394_17400_17600-resume-buckeye_test-K2FA:v0 | 4.637 | /work/gm64/m64000/OTTC/reproduction/libri100_singlehead_average_20261016/results/jsd_standard/evaluation/17394_17400_17600/resume_round_buckeye_test |
| ottc-timit-reproduction | libri100-singlehead-jsd_standard-avg-17394_17400_17800-resume-buckeye_test-K2FA:v0 | 4.637 | /work/gm64/m64000/OTTC/reproduction/libri100_singlehead_average_20261016/results/jsd_standard/evaluation/17394_17400_17800/resume_round_buckeye_test |
| ottc-timit-reproduction | libri100-jsd_upstream-avg-17394_17400_17600-PFER:v0 | 4.535 | /work/gm64/m64000/OTTC/reproduction/libri100_singlehead_average_20261016/pfer/jsd_upstream/17394_17400_17600 |
| ottc-timit-reproduction | libri100-ce-avg-17394_17400_17600-PFER:v0 | 4.534 | /work/gm64/m64000/OTTC/reproduction/libri100_singlehead_average_20261016/pfer/ce/17394_17400_17600 |
| ottc-timit-reproduction | libri100-jsd_upstream-avg-17394_17400_17800-PFER:v0 | 4.535 | /work/gm64/m64000/OTTC/reproduction/libri100_singlehead_average_20261016/pfer/jsd_upstream/17394_17400_17800 |
| ottc-timit-reproduction | libri100-singlehead-jsd_upstream-avg-17394_17400_17600-K2FA:v0 | 98.153 | /work/gm64/m64000/OTTC/reproduction/libri100_singlehead_average_20261016/results/jsd_upstream/evaluation/17394_17400_17600 |
| ottc-timit-reproduction | libri100-singlehead-ce-avg-17394_17400_17600-K2FA:v0 | 98.213 | /work/gm64/m64000/OTTC/reproduction/libri100_singlehead_average_20261016/results/ce/evaluation/17394_17400_17600 |
| ottc-timit-reproduction | libri100-singlehead-jsd_upstream-avg-17394_17400_17800-K2FA:v0 | 98.155 | /work/gm64/m64000/OTTC/reproduction/libri100_singlehead_average_20261016/results/jsd_upstream/evaluation/17394_17400_17800 |
| ottc-timit-reproduction | libri100-singlehead-ce-avg-17394_17400_17600-resume-buckeye_test-K2FA:v0 | 4.640 | /work/gm64/m64000/OTTC/reproduction/libri100_singlehead_average_20261016/results/ce/evaluation/17394_17400_17600/resume_round_buckeye_test |
| ottc-timit-reproduction | libri100-singlehead-jsd_upstream-avg-17394_17400_17600-resume-buckeye_test-K2FA:v0 | 4.635 | /work/gm64/m64000/OTTC/reproduction/libri100_singlehead_average_20261016/results/jsd_upstream/evaluation/17394_17400_17600/resume_round_buckeye_test |
| ottc-timit-reproduction | libri100-singlehead-jsd_upstream-avg-17394_17400_17800-resume-buckeye_test-K2FA:v0 | 4.635 | /work/gm64/m64000/OTTC/reproduction/libri100_singlehead_average_20261016/results/jsd_upstream/evaluation/17394_17400_17800/resume_round_buckeye_test |
| ottc-timit-reproduction | libri100-singlehead-jsd_standard-dev-candidate-17800-K2FA:v0 | 43.712 | /work/gm64/m64000/OTTC/reproduction/libri100_singlehead_average_20261016/candidates/jsd_standard/step_17800 |
| ottc-timit-reproduction | libri100-singlehead-jsd_standard-dev-candidate-17800-resume-dev_other-K2FA:v0 | 21.105 | /work/gm64/m64000/OTTC/reproduction/libri100_singlehead_average_20261016/candidates/jsd_standard/step_17800/resume_round_dev_other |
| ottc-timit-reproduction | libri100-singlehead-jsd_standard-dev-candidate-17600-K2FA:v0 | 43.713 | /work/gm64/m64000/OTTC/reproduction/libri100_singlehead_average_20261016/candidates/jsd_standard/step_17600 |
| ottc-timit-reproduction | libri100-singlehead-jsd_standard-dev-candidate-17400-K2FA:v0 | 43.713 | /work/gm64/m64000/OTTC/reproduction/libri100_singlehead_average_20261016/candidates/jsd_standard/step_17400 |
| ottc-timit-reproduction | libri100-singlehead-ce-dev-candidate-17400-K2FA:v0 | 43.727 | /work/gm64/m64000/OTTC/reproduction/libri100_singlehead_average_20261016/candidates/ce/step_17400 |
| ottc-timit-reproduction | libri100-singlehead-jsd_upstream-dev-candidate-17800-K2FA:v0 | 43.702 | /work/gm64/m64000/OTTC/reproduction/libri100_singlehead_average_20261016/candidates/jsd_upstream/step_17800 |
| ottc-timit-reproduction | libri100-singlehead-jsd_standard-dev-candidate-17400-resume-dev_other-K2FA:v0 | 21.105 | /work/gm64/m64000/OTTC/reproduction/libri100_singlehead_average_20261016/candidates/jsd_standard/step_17400/resume_round_dev_other |
| ottc-timit-reproduction | libri100-singlehead-ce-dev-candidate-17400-resume-dev_other-K2FA:v0 | 21.113 | /work/gm64/m64000/OTTC/reproduction/libri100_singlehead_average_20261016/candidates/ce/step_17400/resume_round_dev_other |
| ottc-timit-reproduction | libri100-singlehead-jsd_upstream-dev-candidate-17800-resume-dev_other-K2FA:v0 | 21.097 | /work/gm64/m64000/OTTC/reproduction/libri100_singlehead_average_20261016/candidates/jsd_upstream/step_17800/resume_round_dev_other |
| ottc-timit-reproduction | libri100-singlehead-jsd_upstream-dev-candidate-17400-K2FA:v0 | 43.702 | /work/gm64/m64000/OTTC/reproduction/libri100_singlehead_average_20261016/candidates/jsd_upstream/step_17400 |
| ottc-timit-reproduction | libri100-singlehead-ce-dev-candidate-17800-K2FA:v0 | 43.728 | /work/gm64/m64000/OTTC/reproduction/libri100_singlehead_average_20261016/candidates/ce/step_17800 |
| ottc-timit-reproduction | libri100-ce-avg-17600_17800_17840-PFER:v0 | 4.528 | /work/gm64/m64000/OTTC/reproduction/libri100_singlehead_average_20261016/pfer/ce/17600_17800_17840 |
| ottc-timit-reproduction | libri100-jsd_standard-avg-17600_17800_17840-PFER:v0 | 4.530 | /work/gm64/m64000/OTTC/reproduction/libri100_singlehead_average_20261016/pfer/jsd_standard/17600_17800_17840 |
| ottc-timit-reproduction | libri100-singlehead-ce-avg-17600_17800_17840-K2FA:v0 | 98.208 | /work/gm64/m64000/OTTC/reproduction/libri100_singlehead_average_20261016/results/ce/evaluation/17600_17800_17840 |
| ottc-timit-reproduction | libri100-singlehead-ce-avg-17600_17800_17840-resume-test_other_buckeye_dev_buckeye_test-K2FA:v0 | 32.195 | /work/gm64/m64000/OTTC/reproduction/libri100_singlehead_average_20261016/results/ce/evaluation/17600_17800_17840/resume_round_test_other_buckeye_dev_buckeye_test |
| ottc-timit-reproduction | libri100-singlehead-jsd_standard-avg-17600_17800_17840-K2FA:v0 | 98.167 | /work/gm64/m64000/OTTC/reproduction/libri100_singlehead_average_20261016/results/jsd_standard/evaluation/17600_17800_17840 |
| ottc-timit-reproduction | libri100-singlehead-ce-dev-candidate-17394-K2FA:v0 | 43.726 | /work/gm64/m64000/OTTC/reproduction/libri100_singlehead_average_20261016/candidates/ce/step_17394 |
| ottc-timit-reproduction | libri100-singlehead-jsd_standard-dev-candidate-17394-K2FA:v0 | 43.712 | /work/gm64/m64000/OTTC/reproduction/libri100_singlehead_average_20261016/candidates/jsd_standard/step_17394 |
| ottc-timit-reproduction | libri100-singlehead-jsd_upstream-dev-candidate-17600-K2FA:v0 | 43.701 | /work/gm64/m64000/OTTC/reproduction/libri100_singlehead_average_20261016/candidates/jsd_upstream/step_17600 |
| ottc-timit-reproduction | libri100-singlehead-jsd_standard-avg-17600_17800_17840-resume-test_other_buckeye_dev_buckeye_test-K2FA:v0 | 32.178 | /work/gm64/m64000/OTTC/reproduction/libri100_singlehead_average_20261016/results/jsd_standard/evaluation/17600_17800_17840/resume_round_test_other_buckeye_dev_buckeye_test |
| ottc-timit-reproduction | libri100-singlehead-jsd_upstream-dev-candidate-17394-K2FA:v0 | 43.703 | /work/gm64/m64000/OTTC/reproduction/libri100_singlehead_average_20261016/candidates/jsd_upstream/step_17394 |
| ottc-timit-reproduction | libri100-singlehead-ce-dev-candidate-17600-K2FA:v0 | 43.728 | /work/gm64/m64000/OTTC/reproduction/libri100_singlehead_average_20261016/candidates/ce/step_17600 |
| ottc-timit-reproduction | libri100-jsd_upstream-avg-17600_17800_17840-PFER:v0 | 4.530 | /work/gm64/m64000/OTTC/reproduction/libri100_singlehead_average_20261016/pfer/jsd_upstream/17600_17800_17840 |
| ottc-timit-reproduction | libri100-singlehead-jsd_upstream-avg-17600_17800_17840-K2FA:v0 | 98.148 | /work/gm64/m64000/OTTC/reproduction/libri100_singlehead_average_20261016/results/jsd_upstream/evaluation/17600_17800_17840 |
| ottc-timit-reproduction | libri100-singlehead-jsd_upstream-avg-17600_17800_17840-resume-buckeye_dev_buckeye_test-K2FA:v0 | 10.443 | /work/gm64/m64000/OTTC/reproduction/libri100_singlehead_average_20261016/results/jsd_upstream/evaluation/17600_17800_17840/resume_round_buckeye_dev_buckeye_test |
| ottc-timit-reproduction | libri100-jsd_standard-last-PFER:v0 | 4.527 | /work/gm64/m64000/OTTC/reproduction/libri100_singlehead_average_20261016/pfer/jsd_standard/last |
| ottc-timit-reproduction | libri100-ce-last-PFER:v0 | 4.526 | /work/gm64/m64000/OTTC/reproduction/libri100_singlehead_average_20261016/pfer/ce/last |
| ottc-timit-reproduction | libri100-jsd_upstream-last-PFER:v0 | 4.526 | /work/gm64/m64000/OTTC/reproduction/libri100_singlehead_average_20261016/pfer/jsd_upstream/last |
| ottc-timit-reproduction | libri100-singlehead-jsd_upstream-K2FA:v0 | 98.144 | /work/gm64/m64000/OTTC/reproduction/libri100_singlehead_20261014/evaluation/jsd_upstream |
| ottc-timit-reproduction | libri100-singlehead-ce-K2FA:v0 | 98.204 | /work/gm64/m64000/OTTC/reproduction/libri100_singlehead_20261014/evaluation/ce |
| ottc-timit-reproduction | libri100-singlehead-jsd_standard-K2FA:v0 | 98.161 | /work/gm64/m64000/OTTC/reproduction/libri100_singlehead_20261014/evaluation/jsd_standard |
| ottc-timit-reproduction | buckeye-singlehead-jsd_standard-K2FA:v0 | 10.446 | /work/gm64/m64000/OTTC/reproduction/buckeye_singlehead_20261014/evaluation/jsd_standard |
| ottc-timit-reproduction | buckeye-singlehead-ce-K2FA:v0 | 10.447 | /work/gm64/m64000/OTTC/reproduction/buckeye_singlehead_20261014/evaluation/ce |
| ottc-timit-reproduction | buckeye-singlehead-jsd_upstream-K2FA:v0 | 10.444 | /work/gm64/m64000/OTTC/reproduction/buckeye_singlehead_20261014/evaluation/jsd_upstream |
| ottc-english-multiphoneset | libri100-jsd_upstream-K2FA-vs-text-free-FA-text-free:v0 | 445.038 | /work/gm64/m64000/OTTC/reproduction/multihead_libri100/evaluation_modes/jsd_upstream |
| ottc-english-multiphoneset | libri100-ce-K2FA-vs-text-free-FA-text-free:v0 | 445.075 | /work/gm64/m64000/OTTC/reproduction/multihead_libri100/evaluation_modes/ce |
| ottc-english-multiphoneset | libri100-ctc-K2FA-vs-text-free-FA-text-free:v0 | 271.708 | /work/gm64/m64000/OTTC/reproduction/multihead_libri100/evaluation_modes/ctc |
| ottc-english-multiphoneset | libri100-jsd-FAbench-buckeye_processed:v0 | 124.315 | /work/gm64/m64000/OTTC/reproduction/fabench_jsd_20261009/buckeye_processed/full |
| ottc-english-multiphoneset | libri100-jsd-FAbench-timit:v0 | 7.820 | /work/gm64/m64000/OTTC/reproduction/fabench_jsd_20261009/timit/full |
| ottc-english-multiphoneset | jsd_upstream-FA-text-free:v0 | 12.452 | /work/gm64/m64000/OTTC/reproduction/multihead_timit/evaluation_modes/jsd_upstream |
| ottc-english-multiphoneset | ce-FA-text-free:v0 | 12.445 | /work/gm64/m64000/OTTC/reproduction/multihead_timit/evaluation_modes/ce |
| ottc-english-multiphoneset | ctc-FA-text-free:v1 | 7.489 | /work/gm64/m64000/OTTC/reproduction/multihead_timit/evaluation_modes/ctc |
| ottc-english-multiphoneset | ctc-FA-text-free:v0 | 4.442 | /work/gm64/m64000/OTTC/reproduction/multihead_timit/superseded_evaluations/numpy_viterbi_20261008_1409/ctc |

## 保护项与未通过原因

| 项目 | 版本 | MB | 原因 |
|---|---|---:|---|
| ottc-timit-reproduction | libri960-timit-jsd_standard:v0 | 10.993 | no identical local copy: COMPARE_COMPLETE.json |
| ottc-timit-reproduction | libri960-timit-ce:v0 | 10.991 | no identical local copy: COMPARE_COMPLETE.json |
| ottc-timit-reproduction | libri960-timit-jsd_upstream:v0 | 10.991 | no identical local copy: COMPARE_COMPLETE.json |
| ottc-timit-reproduction | jsd_fs-metrics:v0 | 6.882 | no identical local copy: evaluation_loss.jsonl |
| ottc-timit-reproduction | ce_only-s3t2ss-topology-review:v0 | 10.110 | no identical local copy: COMPLETE.json |
| ottc-timit-reproduction | ce_fs-s3t2ss-topology-review:v0 | 10.110 | no identical local copy: COMPLETE.json |
| ottc-timit-reproduction | jsd_only-s3t2ss-topology-review:v0 | 10.110 | no identical local copy: COMPLETE.json |
| ottc-timit-reproduction | jsd_fs-s3t2ss-topology-review:v0 | 10.110 | no identical local copy: COMPLETE.json |
| ottc-timit-reproduction | ce_fs-metrics:v1 | 6.816 | no identical local copy: evaluation_loss.jsonl |
| ottc-timit-reproduction | ce_only-metrics:v1 | 6.589 | no identical local copy: evaluation_loss.jsonl |
| ottc-english-multiphoneset | mfa-mfa-protocol-rescore:v0 | 27.873 | local producer directory unavailable |
| ottc-english-multiphoneset | ctc-mfa-protocol-rescore:v0 | 27.873 | local producer directory unavailable |
| ottc-english-multiphoneset | ce-mfa-protocol-rescore:v0 | 27.873 | local producer directory unavailable |
| ottc-english-multiphoneset | jsd_upstream-mfa-protocol-rescore:v0 | 27.873 | local producer directory unavailable |
| ottc-english-multiphoneset | ce-final-benchmark:v2 | 99.384 | local producer directory unavailable |
| ottc-english-multiphoneset | jsd_upstream-final-benchmark:v2 | 99.373 | local producer directory unavailable |
| ottc-english-multiphoneset | ctc-final-benchmark:v2 | 89.744 | local producer directory unavailable |

## 机器可读证据

- /work/gm64/m64000/OTTC/audits/cloud_storage_20261019/ottc_cloud_artifacts_inventory.json：完整云端版本清单
- /work/gm64/m64000/OTTC/audits/cloud_storage_20261019/ottc_cloud_copy_audit.json：逐文件备份路径和 MD5、来源运行与消费者检查
- /work/gm64/m64000/OTTC/audits/cloud_storage_20261019/ottc_cloud_deletion_manifest.json：73 个具体待删除版本及本地文件证据

执行脚本：/tmp/delete_verified_ottc_cloud_copies.py；已执行并写入逐项删除日志。实际计费/配额释放量以云端去重后统计为准，本报告的 4.137554 GB 是删除版本逻辑大小之和。

## 后续测试策略

默认 evaluation_uploads=false。测试/评估 W&B 初始化使用 disabled 模式；常用纯发布函数默认直接返回；训练末尾评估 artifact 上传受独立开关控制。训练标量记录继续使用原 online 默认。原始完成结果、模型、源码快照不修改。61 个 Python 文件编译检查通过；12 个纯发布入口的提前退出、真实 W&B disabled 模式及无网络日志/Artifact 调用已检查。
