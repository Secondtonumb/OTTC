# Monophone CTC + FiLM + Monophone OTTC

实现入口：`ottc/models/sequence/wav2vec2_ottc_film.py`。
这是独立可选模型；原 OTTC 类、已有 recipe 和 source snapshot 不变。

```text
audio -> Wav2Vec2 -> h -> CTC head -> monophone posterior p
                      |                         |
                      +---- ContextFiLM(h, p) --+
                                  |
                                  h'
                         +--------+---------+
                         |                  |
                    OTTC head        learned frame mass
                         |                  |
                         +---- native OTTC loss
```

两个分类头使用完全相同的 monophone ID 表和目标序列，不展开 subphone
状态，不使用 triphone。CTC blank 也占词表中的一个 ID，但不能出现在目标序列中；
目标右侧 padding 是 `-100`，同一个 phone 的连续出现必须保留。
调用者负责提供 monophone tokenizer；模型不会从整数 ID 自动辨别词表语义。

FiLM 逐帧计算 `h' = h * (1 + 0.5 * tanh(gamma_raw)) + 0.5 * beta_raw`，
末层权重和偏置初始化为零，初始输出严格等于 `h`。MLP 结构和调制运算来自
IF-MDD 的 `ContextFiLM`，有原类数值等价测试。默认沿用原实现的
`film_detach_condition=True`：CTC head 由自己的 CTC loss 监督；共享 encoder
和 FiLM 接收 OTTC 梯度。设为 `False` 后，OTTC 梯度也通过 posterior 回传
CTC head。FiLM 内层第一步梯度为零是恒等初始化的自然结果，末层可从第一步学习。

OTTC 分类和 learned frame mass 都读取 `h'`。CTC head 读取原特征 `h`。
条件始终是音频预测，不使用参考文本或参考音素。无标签推理不计算 transport。
主 `forward()` 返回 OTTC logits，并保留原 OTTC 输出字段；单独调用
`ctc_forward()` 可以取得辅助 CTC logits。两个 head 应分别按其解码协议评分。

默认 loss 是 `native_OTTC_batch_sum + CTC_batch_sum`，可用两个权重调整。
CTC 使用 PyTorch `reduction='sum'`，OTTC 保留仓库原有 batch sum；两者
数值尺度不同，默认权重只是实现默认值，尚未通过开发集选择。默认
`bins_metric='jsd'` 调用现有 upstream 分支，没有重写或替换 divergence。
`pop_film_loss_statistics()` 返回 detached 的 `[OTTC, CTC, total, examples]`
累计值，供 trainer 记录原始分量；分布式训练应先对这些量求和再归一化。

```python
from ottc.models.sequence.wav2vec2_ottc_film import (
    wvOTTCFiLMConfig, wvForOTTCFiLM,
)

config = wvOTTCFiLMConfig(
    wav2vec2_path='/path/to/local/wav2vec2-large-lv60',
    vocab_size=len(monophone_vocab),  # 包含 blank / pad ID
    blank_id=blank_id,
    bins_metric='jsd',
    ctc_loss_weight=1.0,
    ottc_loss_weight=1.0,
    film_detach_condition=True,
)
model = wvForOTTCFiLM(config)  # 模型参数为 FP32；autocast 设置独立
out = model(input_values=audio, attention_mask=sample_mask, labels=phone_ids)
out.loss.backward()

model.eval()
with torch.no_grad():
    ottc_logits = model(audio, attention_mask=sample_mask).logits
    ctc_logits = model.ctc_forward(audio, attention_mask=sample_mask).logits

model.save_pretrained('/path/to/new/film-model')
model = wvForOTTCFiLM.from_pretrained('/path/to/new/film-model')
```

使用时须 `import torch`，变量由已有数据管道提供。保存恢复沿用基础类的本地
encoder 路径契约，恢复环境仍需能读取 `wav2vec2_path`。
模型通过独立 API 和本目录的训练 recipe 使用，未改动 `run_exp.py` 的模型选择。
接入新训练 recipe 时，需要独立输出目录、训练 scalar W&B、本地完整评估、
每轮原子保存权重以及 resumable latest / best-dev；至少保留末十轮供平均。
模型及保存权重维持 FP32。不能将这个新结构直接作为旧单头模型的严格续训。

本地验证：

```bash
/work/gm64/m64000/miniforge3/envs/sb_k2_cuda/bin/python -m pytest -q tests/test_monophone_film.py
```

测试使用随机初始化的真实微型 Wav2Vec2，不下载模型。CPU 训练测试只把
原生 OT loss 中硬编码 CUDA 张量的分配位置适配为 CPU，保持实际 transport
和 divergence 运算不变。覆盖 FiLM 原实现等价性、loss/梯度、有效帧掩码、
detach 开关、音频推理、配置/FP32 权重保存恢复和错误目标检查。
这不构成 GPU 吞吐、正式训练收敛或识别效果验证。

## TIMIT 40 epoch 正式任务

用户随后请求提交 CTC + OTTC-AG。入口是 `run.pbs` / `train.py`，
recipe 为 `timit__mono_film__ctc_1__ottc_ag_1__seed42`。
完整 train/dev/test 为 4620/400/192 条，沿用既有 TIMIT monophone 数据与评分协议。
原始 Wav2Vec2 Large LV60 SSL 初始化，native 10 ms；CNN 前六层冻结，
末层 CNN、Transformer、两个分类头、质量头及 FiLM 均训练。
40 epoch、逻辑 batch64、seed42、LR 2e-4、warmup500、weight decay0.005。
CTC/OTTC-AG 权重各1；condition 默认 detach。

使用 FP32 参数/检查点和 BF16 autocast。profile 使用同一完整模型/目标，
比较 microbatch16/32/64 的真实逻辑 batch64，检查最长64条、代表64条、尾批12条，
记录耗时、显存、梯度，验证保存权重是 FP32，并检查两个 head 的 K2 FA。
只有 profile `PASS.json` 且源哈希一致才允许进入正式训练；profile 更新不会
作为正式训练初始化。预检见 `PREFLIGHT.json`。

每轮保存独立权重并保留40轮，另存 best-dev 和两个最新完整 Trainer 状态。
正式训练记录原始 CTC/OTTC-AG/总 loss、step、epoch、LR 到本地和 W&B。
逐轮 dev 及最终 best/final/final5/final10 的两个 head 识别、text-free 分段、
supplied-phone FA 和 R-value 完整保存在本地；test 不参与选择。
平均参数使用 FP32 累加。评估不上传 W&B。

PBS 每片段2小时；训练 deadline4800秒，留下保存和评估时间。
未完成40轮时，脚本在正常保存后自动提交同一 recipe 的续训，复用 W&B ID。
任务提交、排队、profile 通过与40轮训练完成是不同状态；完成以
`TRAIN_COMPLETE.json` / `BENCHMARK_COMPLETE.json` 为准。
