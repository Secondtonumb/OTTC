# 声学前端的区别

2026-10-09：对照本地源码与实际配置。MFA 是音素标签来源；“MFA 标签英语全量”使用自训 Zapformer。四头中的 MFA IPA head 与其他三个 head 共享 Wav2Vec2，不单独加载 MFA 声学模型。

| 实验系列 | 声学输入与前端 | 编码器 / 初始化 | 输出帧步长 | head 结构 |
|---|---|---|---|---|
| 原始 OTTC 单头，CTC / CE / JSD / Soft-DTW | 16kHz waveform → 7 层卷积，CNN 冻结 | 官方 Wav2Vec2-large-LV60；24 层 Transformer，1024 维 | 20ms（stride 320 samples） | 单 phoneset 线性分类；OTTC 另有时间质量预测 |
| 本次 TIMIT / Libri100 / Libri960 四头 | 16kHz waveform，逐句均值/方差归一化 → 同一冻结卷积前端 | 同一官方 Wav2Vec2-large-LV60；共享声学编码器 | 20ms | 原始 OTTC ARPAbet、IF-MDD ARPAbet、MFA IPA、ZIPA IPA 四个独立分类头；OTTC 各有 mass head |
| MFA 标签英语全量 CTC / OTTC / SPOTTC | 80维 Kaldi Fbank，25ms 窗、10ms hop → 每帧 Linear＋LayerNorm | 原英语 MFA-phone 自训 Zapformer，恢复旧同系列 checkpoint；未加载 Wav2Vec2/WavLM SSL 权重 | 10ms；前端不降低帧数，最终输出下采样设为1；编码器内部仍可多尺度 | 105 个输出含特殊项；SPOTTC 313 个输出含三子状态与 blank |
| MFA 标签英语全量 FiLM K4 | 同一 80维 Fbank → Linear(80→384)＋LayerNorm | 4层、384维 Zapformer，权重共享循环4次，循环间 FiLM / posterior feedback；正式训练尚未启动 | 10ms | 第2轮 CTC；第4轮 SPOTTC；它不是与上述完整 Zapformer 相同容量的模型 |
| 之前 IF-MDD Libri100 CE 声学前端 | 16kHz waveform → WavLM 卷积前端 | microsoft/wavlm-large，24层、1024维；已训练的 IF-MDD 权重 | 20ms；10ms 版本把最后一层卷积 stride 从2改为1，非插值 | 1024→384 两层投影；content / topology / ContextFiLM fused heads |

10ms 网格每秒约100帧，20ms 网格每秒约50帧。较细网格降低边界离散化的限制，不能保证真实 segmentation 更准。Fbank 的25ms窗口也不等于25ms帧步长。

Wav2Vec2/WavLM 属于 waveform 的自监督预训练编码器；该 Zapformer 路线使用显式 Fbank，并按旧标签训练配置恢复。模型、初始化、训练数据量和 head 同时不同，不能把它们的 PER/F1 差异只归因于前端。

当前原始 OTTC 与四头实验的声学前端相同，CTC / CE / JSD / Soft-DTW 主要改变目标函数。之前 IF-MDD 的 content/topology/fused 三头处理标签内容与子状态；本次四头则对应四个 phoneset，语义不同。

旧 IF-MDD 范围以实际核对的 Libri100 CE acoustic frontend / 10ms、20ms 导出为准，不泛指目录中的所有历史模型。ZIPA / P-OWSM 本次未作为声学底座重新训练；ZIPA IPA head 使用标签/词表。

来源：

- [本次四头模型](../multihead.py)、[音频归一化与训练初始化](../train_multihead.py)、[官方 LV60 配置](../models/wav2vec2-large-lv60/config.json)。
- [MFA 标签 Zapformer](/work/gm64/m64000/zipa/mfa_miyabi_20260922/zipa/experiments/mfa_zapformer/model_10ms.py)、[FiLM K4](/work/gm64/m64000/zipa/mfa_miyabi_20260922/zipa/experiments/mfa_zapformer/film_k4_10ms.py)、[Fbank 配置](prepare_mfa_eval.py)。
- [旧 IF-MDD 前端配置](/home/m64000/work/IF-MDD/hf_exports/libri100_ce_acoustic_frontend/frontend_config.json)、[卷积步长切换及 head 实现](/home/m64000/work/IF-MDD/hf_exports/libri100_ce_acoustic_frontend/modeling_ifmdd_acoustic_frontend.py)。

[已跑实验标记](EXPERIMENT_STATUS.md) · [结果汇总](RESULTS.md)。
