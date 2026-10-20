# LibriSpeech 100h 四头 OTTC 上游 JSD：FA-Bench

固定模型 `reproduction/multihead_libri100/models/jsd_upstream/final`，40轮完成。
本轮仅推理与评分，不更新 checkpoint。两个 ARPAbet head，K2 和原生 OT
分别报告；MFA/ZIPA 无独立参考边界，不填音素时间准确率。

输入参考词文本经现有固定 Charsiu G2P 得到音素；ao→aa、zh→sh，沿用
本模型的39音素监督。保留每个G2P出现位置（含重复音素），保持词分组。
该输入不同于直接给真实参考音素序列。gold phones/timestamps 只进入评分。

- TIMIT：官方 dev400 / core-test192，原始PHN/WRD，排除SA。
- Buckeye：现有处理版 CTM dev1610 / test1505，历史切句，参考来源不是
  已核实的原始人工标注；不视为完整论文 Buckeye benchmark。
- 本轮 clean；四种官方 Kaldi 降质条件未提供。本地原始 Buckeye 归档
  截断，缺少官方 split 录音，详见 FA-bench 的 preflight_audit.json。

评分直接调用未修改的 FA-Bench `score_pair` / `aggregate`，保留官方
F1、MAE/CI、容差扫描、PER、WER 与边界类型诊断。复用相同清单上已有
Charsiu FC10/FC20 输出并重新评分。K2 使用 IF-MDD 原函数、previous-blank
策略和 duration/frame 时间转换；原生 OT 使用冻结 alignment encoder 的
frame masses、等目标质量 inverse-CDF，每帧20ms，不调时间偏移。

主表官方 `bnd_f1_all_20ms` 为类别/时间匹配；其与严格对齐身份检查的
`bnd_f1_lbl_20ms`（仅内部边界）分开保留，分母不同。

运行 `evaluate.pbs`。每个语料先 smoke（首句、最长句、重复音素句），
再完整推理和评分。推理保存后可在相同源码/模型/数据hash下继续。
最终目录 `{timit,buckeye_processed}/full/` 包含逐句hyp、provenance、
metrics.json/CSV、REPORT.md、W&B ID 和完成标志。
