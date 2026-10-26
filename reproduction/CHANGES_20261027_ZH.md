# 近期修改整理（2026-10-22 至 2026-10-27）

本次同步收录这段时间新增的模型、解码、数据准备、训练与评估工具，以及相关方法说明和回归测试。代码存在、测试通过、任务提交、训练完成和评估完成是不同状态；本说明不将任何配方标记为已完成，也不据此宣称效果提升。实验完成须核对本地完成标记、预测覆盖和最终指标。

## Gamma 位置池化解码

- [gamma_decode.py](../ottc/utils/gamma_decode.py) 根据 learned frame mass 与指定输出长度构造单调 transport，支持加权 logits / log posterior 及独立的 posterior 平均规则。按输出位置预测，保留相邻重复音素。
- [run_exp.py](../run_exp.py) 与 [eval_metrics.py](../ottc/utils/eval_metrics.py) 新增 `--ottc_decoder`、`--ottc_aggregation`；默认仍为 greedy。`gamma_greedy_length` 使用音频 greedy 的长度，`gamma_reference_length` 使用参考长度，后者必须标为 oracle 诊断，不能算无文本识别。
- 归一化 alpha 不预测 quantity。诊断和已保存输出重评分入口位于 [gamma_pool_timit_20261022](gamma_pool_timit_20261022/)。

## 可选模型与目标

- [Monophone CTC + FiLM + OTTC](monophone_film_20261022/README_ZH.md)：CTC posterior 调制声学特征，OTTC 分类头与质量头读取调制后的特征。条件来自音频；默认 detach，另有独立 CTC 推理入口。沿用已有 upstream AG 代价。
- [gw_transport.py](../ottc/ops_loss/gw_transport.py) 与 [GW 模型](../ottc/models/sequence/wav2vec2_ottc_gw.py)：在 occurrence-level transport 上加入局部结构项，分别提供原单调计划正则和实验性 ASOT 风格有限步求解。ASOT 含 soft order prior，不声称严格单调或收敛。
- [ASOT 输出修复](../ottc/models/sequence/wav2vec2_ottc_gw_outputfix.py) 重新构造 `CausalLMOutput`，保证 loss 在 Accelerate 转换时保留；旧实现和历史续训契约保留。
- [单调 GW 模型](../ottc/models/sequence/wav2vec2_ottc_monotone_gw.py) 直接正则原 bucketized 单调计划，支持 CE、原 upstream AG、标准 JSD、Jeffreys/J-div；对应 [多 seed 配方](timit_monotone_gw_sweep_20261027/)。
- [TIMIT frame-wise CE](timit_frame_ce_20261026/) 及 [frame-wise divergences](timit_frame_divergences_20261026/) 使用 gold phone 时间区间投影的逐帧监督，复用已有散度函数。这类监督与 OT-only 不同，应分别解释；不同目标的 raw loss 不用于性能排名。

## 数据与评估协议

- [buckeye_dataset.py](buckeye_dataset.py) 通过 [dataset_defaults.json](dataset_defaults.json) 校验 full40 数据、manifest 与 metadata 哈希；支持检查训练/评估 speaker 重叠，拒绝未适配的词表 ID。
- 新 Buckeye 测试默认使用用户指定的 `buckeye_miyabi_2026-10-22.tar.gz`。准备及独立校验入口位于 [buckeye_full40_20261024](buckeye_full40_20261024/README.txt)。保留历史 42-ID 兼容视图、原始标注和精确时间戳；未知与复合标签的处理须显式记录。
- [exact_ctc](buckeye_full40_exactfa_20261026/exact.py) 提供 supplied-phone CTC 的无剪枝精确路径搜索；参考时间只用于评分，不参与路径搜索。识别、text-free segmentation、supplied-phone forced alignment 分开报告。
- 新增 TIMIT / Buckeye 的 PFER、重复与短片段诊断、posterior 导出/绘图和外部模型评分工具。图表、音频、概率矩阵和逐句结果由工具在本地生成。

## 训练与复现入口

- [TIMIT seed sweep](timit_seed_sweep_20261022/)、[Libri100 → TIMIT](libri100_timit_seed_sweep_20261022/)、[CROTTC-AG](crottc_ag_seed_sweep_20261022/)、[Wav2Vec2 S3T2**](wav2vec2_s3t2ss_20261022/)。
- [交互 TIMIT](timit_interactive_20261023/)、[temporal smoothing](ottc_temporal_smoothing_20261021/)、[Libri100 BPE500 upstream-JSD](libri100_bpe500_jsd_20261021/)。
- [Buckeye full40 seed sweep](buckeye_full40_seed_sweep_20261025/)、[运行时 microbatch 适配](buckeye_full40_retry48_20261026/)、[Buckeye frame-wise 四目标 sweep](buckeye_frame_sweep_20261027/)。

独立 loss/seed 使用独立模型、optimizer 和 W&B run。分组 PBS/MPI launcher 派发独立实验，每个实验 `world_size=1`。真实工作负载 profile 是正式训练的入口条件，不能替代训练或 benchmark 完成证据。

新模型、保存权重和模型平均使用 FP32，autocast 计算精度另行配置。新配方按各自保存策略管理 epoch weights、best-dev 与完整 resume；历史 best-three 配方继续保留其已批准的契约，不能据此声称支持 final-five/final-ten 平均。选择平均或单模型配置只使用 dev，test 单独报告。

## 同步范围与验证

GitHub 收录源码、方法文档、配置、PBS/shell 入口及测试。数据、模型权重、运行环境、运行日志、评估报告、预测、缓存和运行时源码快照保留在本地；运行入口需要另行准备所引用的模型、语料和依赖。既有已提交内容随 Git 历史保留。

训练标量沿用 W&B；独立评估默认 `OTTC_EVAL_UPLOAD=0`，遵循 [wandb_policy.py](wandb_policy.py)。本次整理不启动训练或提交 PBS 任务。

同步前的 Python/shell 语法检查通过。CPU 回归测试为 **153 passed / 1 skipped**，覆盖已有散度和 temporal smoothing、FiLM、GW/ASOT 输出与 BF16 Trainer、Gamma 解码、Buckeye 时间戳/词表/数据契约、精确 CTC 搜索、microbatch 梯度和 checkpoint/平均策略。跳过项需要 CUDA；本次没有重新运行 GPU profile、正式训练或识别/分段 benchmark。
