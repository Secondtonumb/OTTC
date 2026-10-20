# MFA / ZIPA / POWSM 的音素与分段单位

审计时间：2026-10-08。用户所说 POSWM 已确认指 POWSM。
审计只读取现有代码、词表及官方资料；没有启动训练。

## 主要结论

三者的 IPA 不代表相同的标签集合或 token 单位。MFA 是使用词典和文本的强制对齐器；ZIPA 和 POWSM 是能从音频预测音素序列的模型。完整音素、IPA 基础符号、附加符号和模型 token 需要分别记录。

| 系统 | 标签与来源 | 模型输出单位 | 对齐/分段 |
|---|---|---|---|
| MFA `english_us_arpa` | ARPAbet 发音词典；旧 IF-MDD 的 MFA 脚本默认此模型/词典，导出时去掉重音数字 | 词典 phone；使用 GMM-HMM 声学模型 | 给定文本和发音词典，导出完整 phone 区间 |
| MFA `english_mfa` + `english_us_mfa` | MFA 自己的 IPA 音素表，包含发音变体和修饰符；官方 US v3.1.0 词典列出 78 类，不含额外静音/blank | `pʰ`、`tʃ`、`ɾ`、`ɫ`、`ɑː` 等完整词典 phone | 每个完整 phone 一个区间，不自动导出 TIMIT 的 `pcl/tcl` 类别 |
| ZIPA | IPA-Pack++，原生 `unigram_127.model`；词表包含特殊项、基础 IPA 符号和单独的修饰符，127 不是 127 类完整音素 | 实际 tokenizer 将 `pʰ` 拆成 `p` + `ʰ`，`tʃ` 拆成 `t` + `ʃ`；还有单独的 `ː`、`˞` 等 | CTC 模型可导出 token 帧路径/强制对齐；完整音素时间必须按目标映射组合，不能把每个修饰符的 token 边界当成新的音素边界 |
| POWSM / POWSM-CTC | IPA-Pack++，官方预处理规范化 Unicode，用 PanPhon 最长匹配，音素放入 `/…/` 特殊 token | 检查公开 checkpoint：总词表 40,002 项，其中 `/…/` phone token 6,405 项，另有文本 BPE 和任务标记；`/pʰ/`、`/t͡ʃ/` 等是完整 segment token | POWSM 官方提供 encoder CTC forced alignment；POWSM-CTC 也有帧输出。PR、给定文本的 FA、模型文字时间标记是不同评估条件 |

MFA 的 78 类是特定英语词典的完整 phone；ZIPA 127 是模型 token 总词表；POWSM 6,405 是跨语言 PanPhon segment token。三个数字不应直接比较成音素类别数。

## IPA 的实际差异

- MFA US IPA 词典包含送气、腭化、圆唇、长音、成音节辅音和 flaps 等：`p/pʰ/pʲ/pʷ`、`t/tʰ`、`ɾ`、`ɫ/ɫ̩`、`ɑ/ɑː`。它不只是把 ARPAbet 标签替换为 Unicode。
- MFA 使用 `tʃ/dʒ`，POWSM checkpoint 对应的原子 token 是带连接符的 `/t͡ʃ/`、`/d͡ʒ/`；不能直接按字符串相等评分。
- ZIPA 原生 127 词表不包含连接符 `͡`；实测把 `t͡ʃ` 编码时，该字符走未知字符处理。转换前需要规范化，并检查实际 token IDs 是否出现 unk。tokenizer 的 `▁` 也是文本空白标记，不是音素或静音类别。
- ZIPA README 同时提供保留 diacritics 和 no-diacritics 的 checkpoint。删除 `ʰ/ː/̃/̚/...` 会改变目标定义；应作为明确的不同实验配置。
- POWSM 官方 recipe 为 decoder 的 phone 目标与 CTC 目标保留不同版本，CTC 的 `text.ctc` 去掉指定的长短/连接等 suprasegmental 标记；见 `data_prep.py` 和 `process_ipapack.py`。
- POWSM 的预处理还有英语条件规则，例如送气、鼻化和 lateral 处理。这些是标签生成规则，不能视为声学实测的真实实现。
- `pcl/tcl` 在检查过的 MFA 词典、ZIPA 词表和 POWSM checkpoint phone 词表中都不是独立类。IPA 的 `pʰ`（送气）或 `p̚`（无可听释放）也不等于单独的 closure 时间区间。
- ARPAbet `q` 是 TIMIT 的声门塞音标记，可在 IPA 层映射为 `ʔ`；IPA 字符 `q` 本身是小舌塞音，二者不能按字符相等处理。

## 旧 IF-MDD 的两类 IPA 数据

1. `data/librispeech100_ipapack127_json` 从 IPA-Pack++ SHAR `supervision.text` 得到 IPA transcript，用 ZIPA 的 SentencePiece 127 tokenizer。生成脚本复制原有 record 并替换三个目标字符串；原有 `phonemes` 时间字段没有在该步骤重新对齐为 IPA127 token 时间。
2. `data/librispeech100_zipa_ipa_monophone` 是本地 ARPAbet 标签到 ZIPA IPA token 的显式转换：默认 `ah → ə`，静音被删除；多 token 的音素区间平均切分。metadata 明写 `multi_token_phone_timing: split_original_phone_interval_evenly`。

这些 IPA 标签可以用于识别训练，但保留的 ARPAbet 时间或均分产生的 token 时间不能当作新的、人工确认的 IPA 子音素边界。对 segmentation，应按原始完整音素的区间评估，或重新获得与目标 phone 集一致的 MFA 对齐。

旧 ZIPA 评估适配器还把 `ɾ/ʔ` 映射到 `t`，默认会删除一组 diacritics；这与先前 TIMIT `dx → d` 的评估不一致，需要统一评分协议。

## 建议的下一组实验

如果重点是英语识别和完整音素 segmentation，并希望与 MFA 的对齐单位一致：

1. 固定一个 MFA 声学模型与配套词典版本。ARPAbet 路线使用 `english_us_arpa`；IPA 路线使用 `english_mfa` + `english_us_mfa`，不要混用词典与声学模型。
2. IPA 训练以完整 MFA dictionary phone 为单位，保留 `pʰ/tʃ/ɾ/ɑː` 等组合符号，另设 `sil` 和 CTC blank。实际训练词表从训练数据和固定词典确定，不把整个多语言 POWSM 词表直接搬入英语训练。
3. 保留原始 phone/时间/重音或修饰信息，再生成用于评分的 coarse-phone 映射。不要在原始标签导出时不可逆地丢掉信息。
4. 训练用 MFA 对齐可作为监督；TIMIT 测试仍用原始人工 PHN，并明确 closure→sil 与 closure→stop 的两个评分协议。给定文本的 FA 与音频单独输入的 segmentation 分开报告。
5. ZIPA127、POWSM PanPhon phone 保持各自原生 tokenizer，比较时做显式、可审查的映射和完整音素重组。保留 detailed/coarse 两套 PER 和边界指标。
6. 官方 OTTC fused-phoneme 复现继续保留其原始映射；新的 MFA/IPA 实验另建配置和 W&B run，以便归因。

## 来源

### 多语言训练的补充建议

如果目标扩大为 ZIPA 式多语言识别，英语 MFA US 的 78 类不适合作为全模型词表。

- 官方 MFA 模型目录按语言分发声学模型和词典；如 `english_mfa`、`german_mfa`、`mandarin_mfa`。它们使用 MFA phone-set 体系，但每个语言的实际音素子集不同。目录中还有 ARPA/XPF/Epitran phone-set 的模型，不能把所有下载模型直接视为相同 IPA 标准。
- MFA 多词典功能要求声学模型兼容所有词典。增加词典文件不会自动让单语言声学模型变成覆盖所有语言的模型。此次检查没有在官方目录看到 ZIPA 式通用、免文本多语言音素识别 checkpoint；MFA 的常规用法仍是已知语言和文本的强制对齐。
- 多语言主训练优先沿用 IPA-Pack++ 音素监督，固定 original/clean 版本、Unicode 和修饰符规范。为与 ZIPA 直接比较，可保留其 127-token tokenizer；若主要研究完整音素分段，可另建完整 IPA segment 目标并明确与原始 phone groups/边界的对应关系。
- MFA 用来给有可靠文本、受支持语言的数据补充对齐监督，按语言使用配套模型和词典。MFA 输出需统一到共同 IPA 规范，送气、长音、鼻化、声调等信息保留，不能一律删掉。
- 不把 MFA 词典发音自动视为真实 perceived pronunciation；代码转换、均分时长或 MFA 预测都不是人工边界。评估使用独立人工参考，并分别报告每个语言与宏平均。
- 跨语言共享同一规范化 IPA 单位，但保留语言 ID、原始词典 phone 和映射版本。句内 code-switching 需要独立验证，不能由每语种模型的存在推断得到支持。

补充来源：[MFA 官方模型目录](https://mfa-models.readthedocs.io/en/latest/acoustic/index.html)、[多词典兼容要求](https://montreal-forced-aligner.readthedocs.io/en/latest/user_guide/dictionary.html)。

- MFA ARPA 模型：[官方说明](https://mfa-models.readthedocs.io/en/latest/acoustic/English/English%20%28US%29%20ARPA%20acoustic%20model%20v3_0_0.html)
- MFA IPA 模型：[官方说明](https://mfa-models.readthedocs.io/en/latest/acoustic/English/English%20MFA%20acoustic%20model%20v3_1_0.html)
- MFA US IPA 词典（78 类）：[官方词表](https://mfa-models.readthedocs.io/en/latest/dictionary/English/English%20%28US%29%20MFA%20dictionary%20v3_1_0.html)
- ZIPA：[上游 README](https://github.com/lingjzhu/zipa)、[论文](https://aclanthology.org/2025.acl-long.961/)、[公开 127-token 词表](https://huggingface.co/datasets/anyspeech/ipapack_plus_meta/blob/main/ipa_simplified/unigram_127.vocab)
- POWSM：[模型卡](https://huggingface.co/espnet/powsm)、[官方 recipe](https://github.com/espnet/espnet/tree/master/egs2/powsm/s2t1)
- POWSM-CTC：[模型卡](https://huggingface.co/espnet/powsm_ctc)
- 本次模型 revision：POWSM `21ffa410432ace159f3c1fdeed304c70eddf7d34`；POWSM-CTC `f9d7f5c0e21f4e821b743eeab07344cb2d517d35`。

相关源码/模型配置/抽取的词表保存在同目录的 `phone_inventory_sources/`，ZIPA 实际 tokenizer 的例子见 `zipa_tokenizer_examples.json`。
