# 普通 S3T2：Zapformer 原生训练实现迁移

**最新用户修正：** 跳过普通 `S3T2` 正式训练，直接做 `S3T2**`（seed42、10轮）。普通任务3536313及旧控制任务3536361均在执行前取消；直接S3T2**任务为3536393。普通S3T2验证及吞吐记录保留为迁移证据，下面此前提交普通正式训练的描述属于历史准备。当前状态见 `debug_status.json` 和 `../zapformer_libri100_s3t2ss_native_20261012/README.md`。

**最新调试状态（2026-10-12）：** 用户批准的 k2 确定性累加适配已默认接入训练。新版普通S3T2的单卡完整验证（3536077）、四节点DDP（3536124）及跨轮恢复/完成/最终平均评测（3536162）全部通过，JSD与CE模型、累计平均及优化器差异均为0；两轮58步的全部保存状态和逐步原始loss逐位一致。完整模型1/2/4节点吞吐测试也通过。已提交独立seed42、10轮Libri100 JSD（3536313）及正常限时续训/后续独立S3T2**的控制任务（3536361），正式10轮尚未完成。详见 `debug_status.json` 和 `REPRODUCIBILITY.md`。

**最新实验顺序：** 先完成新版全部验证，再测GPU吞吐，然后从Libri100 upstream SP-OTTC JSD开始，接着S3T2**，之后按用户计划扩展960和MFA全量。下方MFA全量/CE正式脚本是此前准备的入口，目前不会自动提交。

当前验收入口覆盖从头 SP-OTTC upstream JSD 和 JSD checkpoint 后的 framewise CE，均使用普通 `s3_t2`。后续 S3T2** 为单独实验。入口直接调用冻结的 `vendor/zapformer/train.py::train_one_epoch`；135份原生 Python 源文件逐文件 SHA256 校验，训练循环没有复制重写。

JSD直接运行的默认数据为已核验的原生IPA Libri100清单 `../zapformer_libri100_data_20261012/training.json`（28,538句、100.586496小时）。`--jsd-manifest`显式选择其他数据，路径和哈希进入断点合约。下表为**历史准备入口**，其worker显式选择MFA全量，尚未提交。

| 阶段 | 数据与初始化 | 正式入口 | 计划 |
|---|---|---|---|
| 从头 JSD | 原英语 MFA 混合语料：1,798,637句 / 2,888.025h；seed42随机初始化 | `run_jsd.pbs` | 8节点×1GPU，10轮 |
| Framewise CE | 之前 Libri100 时间轴：28,538句 / 100.5865h；普通S3T2-JSD checkpoint | `run_framece.pbs` | 4节点×1GPU，3轮 |

两阶段均为单配置 true DDP，regular-g，4小时/段。每段预留10分钟保存退出；再次提交同一脚本从最新checkpoint继续，同一W&B run ID。异常退出不会自动重复提交。默认CE初始化仍为原JSD epoch10 / step68868；可以通过 `S3T2_INIT_CHECKPOINT` 显式改为本次从头JSD的新checkpoint。两个阶段为独立实验，不能互换其optimizer/scheduler状态。

## 直接复用的实现

- 原生 BatchedRubik、参数学习率分组、beta1=.99、矩阵与标量更新、尺度约束和mixed-dtype状态；没有额外gradient clipping或microbatch切分。
- 原生 InterpCosineLRScheduler，base LR=.02、min_factor=.025、linear_scale=.5；原生每epoch步数偏差调整。
- 原生 BF16 autocast / GradScaler(init_scale=1)、低scale处理与25/100/400 batch增长规则。
- 原生encoder辅助梯度，base=.05，乘当前scaler scale；global batch>2000再乘.25。断点读回后先实例化scaler的CUDA scale，避免恢复后的首次forward错误地使用1。
- 完整原生 AlternatingSpecAugment：时间/频率mask、time warp，保留私有CPU/CUDA Generator。CE标签通过同一次时间变换移动；padding/gap保持ignore。
- 原生batch-sum loss、DDP梯度平均、batch_count调度、ParamPlotter与projection overlap诊断。
- 原生float64累计model_avg，每200 batches更新，完整保存；完成后分别评估当前权重和累计平均。min/max copies保留原生默认1。

## Phone任务适配边界

用户批准的k2适配仅将CUDA重复索引梯度累加改为PyTorch确定性`index_add_`，并在原生DenseFsaVec构建前显式展开终止行赋值，兼容PyTorch 2.6的确定性索引写入。原图、前向分数、后验、JSD公式及原生网络/优化器均复用。全部新训练启用确定性算法、cuDNN确定性及单线程autograd；依赖版本、k2源码/二进制哈希、CUBLAS配置进入合约。保证范围为相同软件/硬件和world size下的重复及恢复，不承诺跨world size或跨版本逐位一致。

这里完整复用适用于该任务的Zapformer训练实现。任务仍是10ms IPA phone训练：80维缓存Fbank，逐帧ScaledLinear+LayerNorm投影，去掉ASR前端的时间下采样，105个phone ID / 313个三状态输出。CE整个声学骨干和分类头可训练，未使用的mass head冻结。ASR的SentencePiece、RNN-T、attention loss被phone目标替换；没有重建3倍speed perturbation或改变现有时间轴。不能称为原ASR数据与任务的完整复现。

JSD直接复用原归档 `archived_ot_loss.py` 的 `batched_ottc_loss_bucketized(..., jsd=True)`，并加原S3T2 forward-sum，权重1；没有重写散度。对照旧实现保留控制列1..6剔除、库存ID不变。CE三状态后验logsumexp汇总成phone，保留blank归一化，历史floor/ceil时间标签，忽略gap/padding；CE目标权重1。原ASR的CTC多目标权重.2不应用于此phone目标。CE无JSD或forward-sum项。

动态组batch直接使用原Lhotse DynamicBucketingSampler：quadratic_duration=30，30 buckets、buffer20000、shuffle/drop_last、rank同步、每epoch seed。缓存输入适配器保存完整batch计划哈希和已消费cursor；恢复enumerate offset保留原生按batch调度。默认max_duration=100秒/rank，是当前full batch预算配置，与历史320秒/rank不同。正式单卡吞吐和收敛须以实际运行评估。

Checkpoint保存模型、累计平均、优化器、scheduler、每rank scaler/RNG/增强私有Generator、已消费batch、epoch和源码/数据合约。每50个native训练batch及轮末/限时退出保存，默认保留最近8个完整checkpoint；累计平均状态随checkpoint保存。world size、LR、epoch预算、数据/源码合约变化拒绝续训。验收标记绑定当前源码，过期标记不能启动正式入口。Native ParamPlotter和轮内平滑报告窗口在进程重启时重新创建；训练状态、优化器及数据cursor单独验证，不把报告窗口当连续曲线。

## TIMIT与W&B

固定official dev400 / core-test192。每50个轮内batch按原生触发规则测dev loss及recognition/segmentation（第1个batch也触发），微调前和每轮结束测dev/test；完成后测累计平均。TIMIT不参与梯度训练或test选模。

原生普通S3T2 completion的音频识别记录IPA PER和映射ARPAbet/TIMIT39的S/D/I/PER。自由分段和supplied-phone原生S3T2 FA分别记录P/R/F1/R-value@10/20/30ms，复用冻结IF-MDD scorer并记hash。FA不冒称word-transcript MFA同输入对比；不会用换decoder的分数宣称训练提升。

W&B默认开启，project/entity读取 `reproduction/wandb_config.json`，支持环境覆盖。每步记录原始loss/分量、frames、有效CE帧数、native batch step、实际optimizer update数、epoch、LR、aux系数及GradScaler。Loss raw sum的不同目标不可直接比较。各阶段独立run，恢复复用run ID；网络失败转offline，完成上传评测artifact。Smoke为独立offline验证run，不是正式训练结果。

## 运行

```bash
cd /work/gm64/m64000/OTTC
# 两个验证完成标记通过后，正式脚本才允许启动。
/opt/pbs/bin/qsub reproduction/zapformer_s3t2_native_20261012/run_jsd.pbs
/opt/pbs/bin/qsub reproduction/zapformer_s3t2_native_20261012/run_framece.pbs

# CE也可从新JSD模型开始；替换实际checkpoint路径。
/opt/pbs/bin/qsub -v S3T2_INIT_CHECKPOINT=/absolute/path/to/new-jsd-checkpoint.pt reproduction/zapformer_s3t2_native_20261012/run_framece.pbs
```

历史正式入口输出为 `training_jsd/` 与 `training_framece/`，不会改写已有实验。验证状态以 `validation/SMOKE_PASS.json`、`validation_ddp/DDP_PASS.json` 为准，另有 `../zapformer_native_acceptance_20261012/results/EPOCH_PASS.json` 验证两轮完成及平均模型评测。吞吐原始记录在 `../zapformer_gpu_profile_20261012/`。用户确认两个Libri100阶段均独立seed42、各10轮，再用S3T2** checkpoint扩展960。新正式入口为 `../zapformer_libri100_jsd_native_20261012/run.pbs`，状态和顺序控制记录在 `../zapformer_libri100_native_sequence_20261012/`。
