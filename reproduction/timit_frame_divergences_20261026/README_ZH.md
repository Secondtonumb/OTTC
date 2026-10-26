# TIMIT Frame-wise JSD / J-DIV / AG

用户要求：在 Frame-wise CE 基础上增加三种逐帧损失，每种3卡并行。
执行方式为每种损失1个PBS作业，每个作业申请3节点/3卡，内部通过MPI并行派发3个独立seed（42/43/44）。
每个seed独占1节点/1GPU，world_size=1，共3个PBS作业、9个独立训练进程。
batch64是每个seed的真实batch；不同seed和损失不共享梯度。
现有CE任务及其源码/续训契约保持原样。

完全复用 `../timit_frame_ce_20261026/` 的数据加载、PHN逐帧投影、冻结规则和评分协议。
数据为TIMIT train4620/dev400/test192，原42-ID词表，保留sil/q/dx；
以CNN感受野中心 `199.5+160*i` samples（16kHz）取人工PHN标签，未标注帧显式忽略。
原始SSL初始化和随机phone head；native10ms；前六层CNN冻结，最后CNN/Transformer/phone head训练。
mass/duration head冻结且不参与损失。无OT/CTC/forward-sum/quantity/smoothing辅助目标。

三种损失直接调用 `ottc/ops_loss/ot_loss.py` 中已有helper，不另写替代公式。
使用自然对数，逐类求和、每句话有效监督帧平均，再对实际逻辑batch取平均。
FP32计算散度，BF16用于encoder计算，参数/保存/平均始终FP32。

| recipe cost | 定义 | one-hot零概率处理 |
|---|---|---|
| `jsd_standard` | 0.5 KL(p∥m)+0.5 KL(q∥m)，m=(p+q)/2 | 标准零项约定，无target smoothing |
| `jdiv` | KL(p∥q_eps)+KL(q_eps∥p)，不乘0.5 | q_eps=normalize(clamp(q,min=1e-12)) |
| `ag` | 原上游legacy `jsd`：0.5 KL(m∥p)+0.5 KL(m∥q_perturbed) | 原q+(1-q)*1e-12-q*1e-12，不归一化 |

AG与legacy上游在相同输入上逐位相同。若出现原helper的softmax下溢造成非有限值，
立即失败并保留日志，不静默修改公式或给prediction加epsilon。
各目标loss数值尺度不同，不能按raw loss高低排列方法性能。

40 epochs/2920 steps，LR2e-4、warmup500、weight decay0.005、clip1、batch64。
两小时walltime上限。每个作业先做实际模型/真实数据GPU profile：SSL全部encoder权重一致、
最长64/代表64/尾12、梯度、吞吐、显存、FP32保存及4句评分链路；通过后重新初始化正式训练。
CPU gate已覆盖3种目标的独立数值与梯度对照、mask/reduction、BF16→FP32、
冻结层、audio-only推理、逐epoch保存、FP32平均和精确断点续训。

每个完整epoch独立原子保存权重，保留全部40份；每轮保留最新完整Trainer续训状态。
dev PER选择single best；结束后另外评估final5/final10 FP32平均，记录全部成员/epoch/step/hash。
所有候选先dev后test；主比较是同一single best-dev模型，平均结果另列。
PER/PanPhon PFER、text-free及supplied-phone FA的P/R/F1/R-value（10/20/30ms）、FA MAE沿用CE评分器。
原CTC图FA在frame-supervised后验上作为匹配诊断，保留其无blank监督的协议说明。

仅训练标量/配置发W&B；profile及dev/test预测/指标/报告均本地保存，`OTTC_EVAL_UPLOAD=0`。
`REPORT.md` / `COMPARISON.json` / `RESULTS.csv`联合列出Frame-wise CE及本组三种目标，
缺失seed保留pending，不把queued/running或只完成训练视为完成benchmark。

```bash
source reproduction/libri960_continue5_20261017/environment.sh
"$LIBRI_PYTHON" -B -m reproduction.timit_frame_divergences_20261026.validate_launcher
"$LIBRI_PYTHON" -B -m reproduction.timit_frame_divergences_20261026.validate
"$LIBRI_PYTHON" -B -m reproduction.timit_frame_divergences_20261026.submit
```

当前3个分组任务的提交凭证在`SUBMISSIONS_GROUPED.json`；实际队列和运行状态以PBS以及本地profile/训练/benchmark完成标记为准。
`run.pbs`请求`select=3:ncpus=32:mpiprocs=1`，通过`mpirun -np 3 --map-by ppr:1:node`派发。
worker清理MPI rank环境后进入独立Trainer，并在运行时断言world_size=1。每个seed单独记录日志、W&B ID、权重和续训状态；作业头检查全部3个seed的实际退出码与benchmark完成标记。
日志位于`jobs/<jobid>/<recipe>/worker.log`，真实并行状态另存各模型的`RUNTIME_PARALLELISM.json`。

历史单卡提交（已全部取消、未开始训练；原凭证`SUBMISSIONS.json`及`launch_archive/single_jobs/`保留）：

- `frame_jsd_standard__seed42`：`3709587.opbs`
- `frame_jsd_standard__seed43`：`3709588.opbs`
- `frame_jsd_standard__seed44`：`3709589.opbs`
- `frame_jdiv__seed42`：`3709590.opbs`
- `frame_jdiv__seed43`：`3709591.opbs`
- `frame_jdiv__seed44`：`3709592.opbs`
- `frame_ag__seed42`：`3709593.opbs`
- `frame_ag__seed43`：`3709594.opbs`
- `frame_ag__seed44`：`3709595.opbs`

当前分组提交（每个job3节点/3GPU，内部并行seeds42/43/44；已核验均排队）：

- `jsd_standard`：`3709816.opbs`
- `jdiv`：`3709817.opbs`
- `ag`：`3709818.opbs`
