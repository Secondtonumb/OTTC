# 近期修改整理（2026-10-21）

本 fork 在原始 OTTC 上保留模型扩展、训练和评估配方，以及相关方法说明。代码归档不代表每项实验均已完成；实验状态以各配方中的完成记录和本地结果为准。

## 模型、目标与复现流程

- 保留原始 upstream-JSD，并将 standard-JSD 明确作为独立目标；包含单头、多头及子音素拓扑扩展。
- 收录 CIF、transducer 和不同声学前端的实现与配方。英语实验入口见 [english_suite](english_suite/README.md)，CIF 入口见 [TIMIT CIF](cif_timit_20261016/README.md)。
- 收录训练、checkpoint 管理、平均及评估工具。后续训练需保存独立 epoch 权重，保留 best-dev 与完整 latest 状态，平均成员及开发集选择依据需可追溯。
- 保留识别、text-free segmentation、forced alignment 的独立评估口径，以及归档 MFA CTM 身份未核实的说明；不能由目录名认定模型身份。

## 可选 temporal smoothing

底层 `batched_ottc_loss_bucketized` 新增仅限关键字参数 `smooth_weight=0.0`、`smooth_tau=4.0` 和 `smooth_mask=None`。默认关闭；启用后对完整类别的相邻帧 log-probability 加截断平方差，前一帧停止梯度，相邻帧均须有效。每条语音按有效帧对和类别数平均，再对 batch 求和。

另有可选模型与 Trainer 扩展、数值及梯度回归测试，以及交互 GPU probe 入口。详见 [平滑说明](ottc_temporal_smoothing_20261021/README_ZH.md)、[底层接口](ottc_temporal_smoothing_20261021/core_integration/README_ZH.md) 和 [交互测试用法](ottc_temporal_smoothing_20261021/INTERACTIVE_GPU_ZH.md)。实现测试通过不代表识别或分段质量提升；正式训练仍需对应配方的吞吐与安全性 profile。

TIMIT BPE 配方另增加基于保存输出的 I/D/S 回填和 A-B-A excursion 诊断脚本，不重新训练模型。

此次同步前重新运行 `test_smoothing.py` 与 `test_core_loss.py`：33 项 CPU 回归测试全部通过，覆盖数值、梯度、默认关闭行为和小模型 Trainer 集成；未运行正式训练或识别/分段 benchmark。

## 跟踪与归档范围

新训练保留 W&B 标量曲线和本地记录。评估使用 `wandb_policy.py`，默认不向 W&B/CoreWeave 上传；只有用户明确要求时才启用 `OTTC_EVAL_UPLOAD=1`。

此次 GitHub 同步包含源码、配方、方法文档与回归测试所需的小型固定基线；模型权重、数据集、运行环境、运行日志、逐句预测、缓存及 GPU probe 产物保留在本地。既有已提交报告随代码历史保留。
