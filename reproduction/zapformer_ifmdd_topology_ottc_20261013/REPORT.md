# IF-MDD subphonetic topology OTTC → native Zapformer

用户指定复刻 subphonetic topology OTTC，保留当前 MFA 标签。先前 `zapformer_ifmdd_ctc_20261013` 是 content CTC 验证，不能算本次复刻结果。本目录使用独立模型与训练输出。

## 实现

直接调用 vendor 中未改动的 IF-MDD `SubphoneticTopologyLoss(loss_type="ottc")`，以及其 `batched_ottc_loss_bucketized`。源代码与 IF-MDD 当前源文件 SHA256 均记录在 `source_manifest.json`；源文件和已训练 checkpoint 均未修改。

- ordinary token 展开三个 states，保留相邻重复 phone occurrences；`sil/spn` 各一个 state。
- 当前 MFA vocabulary 有 105 个 ID，去掉 blank 和两个控制符后 102 个 token，生成 303 类 substate head（包括 acoustic blank）。ID 映射见 `inventory_mapping.json`。目标没有插入 blank。
- topology/mass heads 均为 nn.Linear；mass head 输出经 mask + softmax 传入原 criterion，再由原 criterion 添加 1e-6、归一化。原 logits 清理并 clamp 到 [-30,30]，target mass 为 state_uniform。
- **原 IF-MDD topology OTTC 硬编码 jsd=False：使用 CE transport cost；不是 frame CE，也不是 CTC。** 训练不含 FSA forward-sum；原始 upstream JSD 实现仍保留在此前的 archived_ot_loss.py，本迁移没有替换它或把当前 CE loss 叫作 JSD。
- 复刻 topology 分支，不含 content/fused/FiLM 的额外 objectives。参照 nofilm run 原来 content/topology 各 .5；那次完整模型的历史 PER 不可直接当成本次单 topology Zapformer 的结果。
- 保留 native Zapformer、Fbank80/10ms、BatchedRubik optimizer、原 scheduler/SpecAugment、model averaging 和断点恢复。原参考模型是冻结 WavLM，这个 encoder/optimizer 适配是显式差异。
- 保留 IF-MDD relative-length + floor 行为，日志记录 effective frame/token lengths。

## 解码与图

`text_free` 是原 IF-MDD SpeechBrain greedy substate collapse 再按 phone/state regression 合并。这条 decoder 保留 incomplete phones，不强制 FSA 路径合法。

`fsa_text_free` 是单独的 audio-only S3T2** phone-loop FSA dense intersection + shortest path：图来自未改动的 IF-MDD token_fst_text，没有 reference、phone LM 或额外惩罚。见 `fsa_b.svg`、`fsa_b.pdf` 和实际 `recognition.fsa.txt`。

`b0 b0 b1 b1 b2 b2 → b`；`b0 b2 b0 b2 → b b`。后者是图允许的完整重复 phone，不能简单消去。

FA 使用 supplied phone occurrences（包括 MFA silence），原 IF-MDD transcript graph，previous blank、duration/frames 时间协议。与 audio-only recognition 分开。boundary P/R/F1/R-value 复用冻结历史 scorer/R-value，采用 10/20/30ms 单匹配、排除首尾 20ms 的边界，substate 内部变化不计 phone boundary。本验证不声明 paper MFA topline 身份或完整 benchmark。

## 验证

`PARITY_PASS.json`：learned/uniform/clamped loss 和 logit/mass gradients 与 live IF-MDD 源码完全一致；独立 interval-overlap transport oracle 验证；padding、repeat、silence、blank 与原 decoder 测试；独立 max-plus DP 与 FSA 最短路径一致。

`validate.pbs` 提交单 GPU bounded validation，包含 tiny/full native training、2 steps 连续 versus 1+resume 到 2 的 replay、4 条真实训练语音 200 updates 学习验证。诊断 PER 仅是固定训练句，不是 held-out benchmark。所有训练保存 raw loss/steps/epoch/LR、greedy/FSA PER，W&B 默认启用并保留 resume run ID。

迁移验证通过后，正式 Libri100 topology OTTC 训练通过 `run.pbs` 提交（3542032.opbs）。新模型从 seed42 初始化，30 epoch 计划，单 GPU，4h walltime；checkpoint 与同 run ID 的恢复入口已实现。输出在 training/。当前不能宣称 held-out PER 或训练完成。

GPU validation 已完成，job 3541971.opbs。Loss 22.838825 → 0.709275；200 steps 两种 decode 对 60 tokens 的 PER 均为 0%。断点恢复 model/model_avg/optimizer 最大差为 0，见 INTEGRATION_PASS.json。逐句对照见 DECODE_COMPARISON.md 与 validation/fixed4/reference_decode.json，checkpoint 存在 validation/fixed4/step-000000200.pt。
