# OTTC 的 MS-TCN 时序平滑项

后续按用户指定，已将可选参数直接接入 [`ot_loss.py` 的底层OTTC函数](core_integration/README_ZH.md)。本页其余内容记录前一轮的模型扩展方案；原模型扩展仍可使用。共享`ot_loss.py`如今有明确的底层改动，原保存的验证记录及源码hash保留为历史记录，新回归结果位于`core_integration/`。

Miyabi交互节点可直接运行 [`interactive_gpu.sh`](interactive_gpu.sh)，默认完整训练40 epochs、optimizer batch size 64，按dev PER保留最佳3份权重及最新续训状态；训练及短测试参数见 [GPU交互训练说明](INTERACTIVE_GPU_ZH.md)。

按用户提供的 [原始 MS-TCN 仓库](https://github.com/yabufarha/ms-tcn) 核对了 [model.py 的训练 loss](https://github.com/yabufarha/ms-tcn/blob/master/model.py)：完整类别 log-softmax，参考的前一帧停止梯度，逐类平方差截断到16（τ=4）。原作者系数0.15、按含padding的张量取mean并只使用当前帧mask；这里针对语音变长batch，要求相邻两帧都有效并排除padding归一化，未照搬0.15。

这是 **temporal smoothing**，不是把one-hot目标混成均匀分布的传统label smoothing。不做argmax，不需要帧边界标注。正则施加在分类logits上，覆盖全部类别，包括blank/sil；不直接平滑α。S3模型中直接平滑状态分布，不将状态折叠后冒充音素平滑。

## loss 与归约

`ottc/ops_loss/temporal_smoothing.py` 的 `temporal_smoothing_loss(logits, valid_mask, tau=4.0)` 与用户贴出的版本一致：按整个batch的有效帧对数×类别数平均。额外支持 `reduction='none'` 返回每条语音的均值，`reduction='sum'` 对这些逐条均值求和。少于两帧、没有有效相邻帧的语音贡献零，且保留合法零梯度。padding先置零再做FP32 log-softmax，避免padding中的NaN污染梯度。

原OTTC损失是batch求和，因此模型扩展采用：

\[
L_{\text{total,sum}}=L_{\text{OTTC,sum}}+\lambda\sum_{b=1}^B
\frac{\sum_{(t-1,t)\,\text{valid}}\sum_c\min[(\log p_{btc}-\operatorname{sg}(\log p_{b,t-1,c}))^2,\tau^2]}
{C\max(N_b,1)}.
\]

训练器仍按原约定除以logical batch大小。这样切分microbatch时正则可相加，长短语音各按自身有效帧对归一化。它与“整个batch先按所有帧对求一个均值，再直接加到batch-summed OTTC”在变长batch上权重不同，配置明确记录 `utterance_mean_then_batch_sum`，不混用两种raw loss。

配置 `temporal_smoothing_weight` 默认0，`temporal_smoothing_tau` 默认4。λ>0时有labels的前向（包括eval loss）返回总loss；无labels的音频推理不计算正则。λ=0精确回退原模型。正则过强可能抹掉真实短音素；大于τ的log-prob差分已经饱和，其平滑梯度为零，不能保证消除所有高置信度跳变。是否改善PER/分段需用开发集和正式测试验证。

## 已接入的模型接口

原单头模型可直接使用新增的模型类；原 `wvForOTTC`、`ot_loss.py`、`run_exp.py` 及历史配方均未编辑：

```python
from ottc.models.sequence.wav2vec2_ottc import wvOTTCConfig
from ottc.models.sequence.wav2vec2_ottc_smoothing import wvForOTTCTemporalSmoothing

config = wvOTTCConfig(
    wav2vec2_path=ssl_checkpoint,
    vocab_size=len(vocab),
    bins_metric="jsd",                 # 保留原始 upstream-JSD
    forward_type="regular",
    temporal_smoothing_weight=0.1,      # 仅展示配置方式，未调优
    temporal_smoothing_tau=4.0,
)
model = wvForOTTCTemporalSmoothing(config)
out = model(input_values=audio, attention_mask=sample_mask, labels=phones)
out.loss.backward()
```

模型内部自动取 `out.hidden_states[1]` 的**encoder下采样后有效帧mask**，不把sample mask直接用于平滑。logits、hidden_states、attentions及tuple返回结构保留原合同；没有新增state-dict键。权重和τ写进config，保存/加载时保留，不新增模型参数。加载已有checkpoint时显式使用这个类及所需配置；改变目标的新实验应使用新输出目录和新run，而非覆写历史resume。

本目录 `integration.py` 还提供了**当前原始SSL/native10ms配方**的可选model factory，支持mono/S3T2** × CE/upstream-JSD/standard-JSD：

```python
from reproduction.ottc_temporal_smoothing_20261021.integration import recipe, build_model

config = recipe("wav2vec2__timit__10ms__mono__jsd_upstream", smooth_weight=0.1)
model, topo = build_model(config, vocab, steps_per_epoch=73)
```

初始化、末层CNN的10ms stride/可训练设置、原始cost分支和S3新head复用20261019配方；配置有新实验名/源码hash。原训练脚本不会自动启用本项，也不会修改已完成或正在运行的模型。

## 分项日志

标准HF训练可用 `ottc.utils.temporal_smoothing_trainer.TemporalSmoothingTrainer`。已有custom Trainer（例如native10的 `QuietTrainer`）可将 `TemporalSmoothingTrainerMixin` 放在其MRO最前面。保留原training_step与loss缩放：

```python
from ottc.utils.temporal_smoothing_trainer import TemporalSmoothingTrainerMixin
from reproduction.timit_ssl_native10_20261019.train import QuietTrainer

class SmoothedTrainer(TemporalSmoothingTrainerMixin, QuietTrainer):
    pass
```

在原log写入本地history和W&B之前，记录 `loss_ottc_per_utterance`、`loss_temporal_per_utterance`、`loss_temporal_weighted_per_utterance`、`loss_total_per_utterance`、λ、τ。所有microbatch的detached统计先累加，分布式先跨rank求和再归一化；不增加模型输出字段或破坏预测收集。W&B初始化、独立run ID和resume仍由原训练器负责。此模块不自行联网或上传，测试/验证默认全部本地。

## 验证范围

测试包括作者未padding公式的数值/梯度一致性、stop-gradient方向、截断饱和、padding/非连续mask/NaN padding、无有效帧对、BF16输入FP32计算、microbatch可加性及梯度一致性。还用真实tiny Wav2Vec2+原OTTC loss检查三种cost的前向/反向、λ=0精确回放、无labels推理一致、save/reload配置与权重、一步Trainer参数更新与本地分项日志，以及native10六种head/cost设置的loss/梯度与CNN可训练策略。

CPU集成测试仅在内存中把原OTTC loss的CUDA字面量分配映射到CPU，沿用仓库既有CPU门控方法；未改公式或生产源码。测试使用临时小编码器，不下载模型、不读大语料、不启动长训、不上传W&B。它验证实现可用性，不构成识别/分段提升结果。DDP全量训练和吞吐尚未实测；正式训练前仍需新配方的准确profile及独立epoch checkpoint/best/latest策略。

```bash
PYTHONPATH="$PWD/reproduction/vendor:$PWD" OMP_NUM_THREADS=2 \
  /work/gm64/m64000/miniforge3/envs/sb_k2_cuda/bin/python -m pytest -q \
  reproduction/ottc_temporal_smoothing_20261021/test_smoothing.py
```

日志见 `validation.log`，机器可读测试结果见 `validation.xml`。
