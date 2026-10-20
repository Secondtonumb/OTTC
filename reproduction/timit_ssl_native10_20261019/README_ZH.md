# TIMIT 原始SSL初始化、可训练末层CNN的原生10ms OTTC

用户2026-10-19明确授权重跑此前六组，并把此配方称为 from scratch：保留原始Wav2Vec2 SSL初始权重；最后一层CNN的stride从2改为1且可训练；新OTTC任务头可训练。这里不表示随机初始化SSL编码器。

## 六组范围

monophone / IF-MDD S3T2** × 原版upstream JSD / CE transport cost / standard JSD。每组独立W&B run和独立模型目录。此次只提交TIMIT；Buckeye和Libri100不在此重跑范围。之前Libri100结果门控保持不变，未来8卡偏好仍仅用于Libri100。

## 初始化与训练

- 编码器：原始`facebook/wav2vec2-large-lv60`，revision `0cde644b64dac88d8416bec1c92a4099b850ba0b`，本地原始权重SHA256 `d3c26ae7322ea631b59128407c0913b31c0d7d2c2a7a5f4375144a5b2422ef04`。初始化后逐张量核对全部编码器参数/缓冲区，完全相等才放行；不加载Libri监督OTTC parent。
- CNN stride `[5,2,2,2,2,2,1]`，16kHz输入对应160采样点/10ms hop；无插值或转置卷积上采样。
- CNN第1–6层冻结；第7层的卷积及归一化参数可训练。Transformer全程微调。
- 音素/子音素头与时间质量头从新随机权重开始，全程可训练。`forward_type=regular`，不在第30轮切换到冻结质量分支。S3头直接新建119类，不从已监督训练的音素头复制。
- OTTC目标函数复用未修改的`ottc/ops_loss/ot_loss.py`；CE、原版upstream JSD和standard JSD明确分开。S3 target mass保留此前state_uniform；mono为phone_uniform，没有把解码FSA误称为额外的训练FSA损失。
- train/dev/test = 4620/400/192，复用冻结数据清单、既有训练phoneset和人工PHN评分参考。
- 40轮，真实全局batch64，AdamW，学习率2e-4、warmup500、weight_decay0.005、seed42；BF16训练、独立FP32最终评测。microbatch依据每组真实GPU吞吐和最长输入内存确定。

旧10ms继承Libri960监督OTTC且冻结所有CNN，旧20ms冻结所有CNN并用regfrozen；都不匹配本次初始化/可训练参数，不能复用它们作为本次CE权重。详见REUSE_AUDIT.json；旧实验与原始源码/权重保持不变。

## 验证与资源

1. CPU门控：六种head/cost的有限loss/梯度、末层CNN和两个头更新、前六层不变、重复音素K2自由解码和给定序列FA、独立epoch权重与精确resume、原始SSL逐张量一致性。CPU仅测试时将上游loss中的CUDA分配字面量改到CPU；数学公式未变，生产GPU直接调用原始函数。
2. 每个作业先通过CUDA小模型门控，再独立profile完整实际模型：microbatch16/32/64、最长真实输入、真实logical batch64前向/反向/optimizer、数据加载耗时、峰值显存、非零梯度、真实K2自由解码与FA、checkpoint写盘。按吞吐选择安全microbatch。
3. 每作业一张GH200、32 CPU、PBS walltime两小时；profile+训练100分钟预算，预留20分钟保存和最终评测。若未完成，保留完整状态并用同配方及W&B ID继续；不请求24小时长作业。

`dispatch.json`是提交记录；`STATUS.json`结合调度器、profile标记、history、epoch权重、训练/评测完成标记区分排队、profile、正式训练和完成。只有qstat状态不能证明训练完成。

## 指标与checkpoint

每轮保存独立epoch模型权重，含epoch、optimizer step、SSL身份、配置和源码hash；保留全部40轮，以及完整最新Trainer/optimizer/scheduler/RNG状态和best-dev模型。最终同时评测best-dev、final、最后5轮平均、最后10轮平均，记录成员路径/epoch/step/hash。

训练期dev指标标为Trainer BF16；S3使用原IF-MDD K2 phone-loop作为正式识别/分段路径，greedy仅诊断。记录raw loss、optimizer step、epoch、LR、PER、边界P/R/F1/R-value，并分开无文本和给定音素FA。原始loss定义/归约不同，不横向比较raw loss绝对值。

最终重新构建无Trainer/Accelerate autocast包装的FP32模型，先完成四候选全部dev评测和选点，再打开test。识别按dev PER选；另单独保存非FA R、FA R、FA MAE选择。非FA统一固定10ms时钟；FA使用duration/frame_count和blank归前音素。边界唯一端点、排除首尾20ms、10/20/30ms一对一匹配；冻结IF-MDD R-value。K2 graph在inference_mode(False)中构建；无phone LM或参考时间输入。最终要求FA和S3自由解码0失败才能写完成标记。

W&B默认用仓库配置，独立profile/training run；resume复用保存ID。联网失败保留offline日志。仅同步实验配置、日志和数值结果，不上传音频或源码包。

运行/跟踪：

```bash
source reproduction/libri960_continue5_20261017/environment.sh
"$LIBRI_PYTHON" reproduction/timit_ssl_native10_20261019/validate.py --device cpu --full-ssl
"$LIBRI_PYTHON" reproduction/timit_ssl_native10_20261019/launch.py --submit
"$LIBRI_PYTHON" reproduction/timit_ssl_native10_20261019/status.py
```
