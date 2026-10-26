# Buckeye full40 Frame-wise：四任务 × 三 seed

按用户新要求替换原单组 AG / seed42（3715358.opbs）：CE、标准 JSD、AG、J-div 各一个 PBS 任务，
每任务 `select=3:ncpus=32:mpiprocs=1`，共三节点/三张 GH200。MPI rank0/1/2 分别派发 seed42/43/44，
进入 Trainer 前清理 MPI/DDP 环境；每个 seed 断言 world_size=1，有独立 optimizer、checkpoint、日志和 W&B run。
四个任务可独立排队和并行，总计12组训练；没有跨 seed 或跨损失梯度同步。

原单组配置、源码及提交凭证保留；新配方独立位于本目录。替换前后证据见 `REPLACED_SINGLE_JOB.json`。
新任务的幂等提交记录为 `SUBMISSIONS_GROUPED.json`，实时核验状态为 `STATUS.json`。

通过 `reproduction.buckeye_dataset.load_buckeye_rows` 和已校验 registry 使用新版 full40，
train18221/dev6417/test6839、固定24/8/8 speaker split、原42-ID词表及其映射/排除规则。
使用 `reference` 的原始时间区间，以16kHz感受野中心 `199.5+160*i` 投影逐帧标签。
未覆盖帧用-100忽略，不补造边缘sil、不拆复合标签、不合并金标区间、不改时间戳；未知或重叠标签报错。
完整帧覆盖和逐句忽略清单见 CPU_PASS / DATA_AUDIT。

CE 复用原 FrameCE；三种散度直接复用现有 FrameDivergence 和仓库 helper，定义不变：

| cost | 定义与 zero-target 处理 |
|---|---|
| ce | hard gold-frame CE |
| jsd_standard | 0.5 KL(p∥m)+0.5 KL(q∥m)，exact one-hot，无 smoothing |
| ag | 原 upstream legacy-jsd：0.5 KL(m∥p)+0.5 KL(m∥q_perturbed)，沿用原 epsilon、不归一化 |
| jdiv | KL(p∥q_eps)+KL(q_eps∥p)，q_eps=normalize(clamp(q,min=1e-12))，不乘0.5 |

每句有效帧平均，再按实际逻辑batch句数平均；无 OT/CTC/quantity/smoothing 等辅助目标。
不按 raw loss 高低比较损失。gold 时间监督强于 OT-only，结果须分开解释。
原始 wav2vec2-large-lv60 SSL 初始化、随机 phone head，native10ms；前六 CNN 冻结，
最后 CNN、Transformer、phone head训练，mass head冻结不用。
40 epochs、逻辑batch64（尾45）、LR2e-4/warmup500/weight decay0.005/clip1；模型/保存/平均FP32，encoder BF16。

CPU gate 检查12个配置、全量时间线、四损失独立公式/梯度/屏蔽、实际dev选模回调、精确resume和FP32平均。
launcher gate 实际运行12个 worker dry-run，验证 rank→seed 以及被污染的MPI环境清理。
每个 seed 在所在GPU上分别测 physical batch64/48，包含最长64、代表64、尾45、最长验证、验证后反传、
SSL初始化、有效梯度和FP32保存恢复。选择通过所有gate且代表逻辑batch端到端最快的配置；均失败则不训练。
profile权重不进入正式训练，正式训练重新初始化。prepared/submitted/running/profile/训练完成/评估完成分开报告。

每完整epoch原子保存独立FP32权重并保留全部40份，另存best-dev链接和最新完整Trainer状态。
每轮仅用Buckeye dev PER选择；结束后独立FP32评估best_dev/final5/final10，记录平均成员、epoch、step和hash。
先Buckeye dev选候选，Buckeye test和TIMIT dev/test从不选择；最终PER/PFER和text-free/FA的P/R/F1/R-value、FA MAE分开。
FA沿用无剪枝精确CTC搜索，gold时间仅用于训练、loss诊断和评分，不用于搜索；保留全部长尾样本。
单best-dev作为主表，final5/final10和dev_selected另列；三seed未齐保持pending，不计算不完整均值。
只向W&B记录训练标量和配置，所有profile/验证/测试/预测/报告仅本地，OTTC_EVAL_UPLOAD=0。

每任务申请8小时，预留1小时用于最终评估；达到训练预算会保存后以75退出，保留真实worker退出码，不能当作完成。

```bash
source reproduction/libri960_continue5_20261017/environment.sh
"$LIBRI_PYTHON" -B -m reproduction.buckeye_frame_sweep_20261027.validate
"$LIBRI_PYTHON" -B -m reproduction.buckeye_frame_sweep_20261027.validate_launcher
# 原单组取消及其凭证完成后：
"$LIBRI_PYTHON" -B -m reproduction.buckeye_frame_sweep_20261027.submit
```

运行日志：`jobs/<jobid>/frame_<cost>__seed<seed>/worker.log`；每seed进度/权重：`models/<recipe>/`。
GPU gate、训练完成、评估完成分别以 `profiles/<recipe>/PASS.json`、`TRAIN_COMPLETE.json`、`BENCHMARK_COMPLETE.json` 为证。
汇总输出为 `REPORT_ZH.md` / `RESULTS.json` / `RESULTS.csv`；保留原 OT-only 报告并在其中链接本组补充。

## 本次提交记录

原单组作业3715358已在排队、未开始训练时取消。新作业如下；每任务均已核验为3节点/3GPU。状态以 [STATUS.json](STATUS.json) 的核验时间为准。

| 目标 | Job ID | seeds | 节点 / GPU |
|---|---|---|---|
| ce | 3715421.opbs | 42 / 43 / 44 | 3 / 3 |
| jsd_standard | 3715422.opbs | 42 / 43 / 44 | 3 / 3 |
| ag | 3715423.opbs | 42 / 43 / 44 | 3 / 3 |
| jdiv | 3715424.opbs | 42 / 43 / 44 | 3 / 3 |
