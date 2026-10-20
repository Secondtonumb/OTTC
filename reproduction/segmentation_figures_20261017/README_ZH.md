# 频谱、识别分段与 FA：两个独立子页

打开[总览](interactive/index.html)，选择：

1. [Recognition / 无文本识别与分段](interactive/recognition.html)：模型推理只输入音频。TIMIT 有 CTC、纯 OTTC CE/upstream-JSD/standard-JSD、WavLM CE/upstream-JSD、三种 CIF 和 Zapformer，另加本次OTTC upstream-JSD的α分数帧解码，共11轨；Buckeye/Libri clean/other各有三种纯OTTC、三种CIF及新增α解码，共7轨。
2. [FA / 强制对齐](interactive/forced_alignment.html)：四条示例均有三种纯OTTC的给定参考音素 K2 FA，以及本次新运行的MFA词转写FA；TIMIT另有Zapformer给定音素FA。另加入本次CIF+CTC/CIF+OTTC辅助头K2 FA，以及三种CIF的数量条件分段（非K2 FA）。TIMIT归档CTM身份未核实，默认隐藏，可手动勾选。

两页都包含波形、spectrogram、参考音素与模型区间。点击区间持续高亮、只播放该区间，到终点暂停；支持重播、清除、缩放和平移。页首可互相切换，保留当前语音选择。音频与频谱内嵌，每个子页可独立离线打开；子页间导航需要将HTML放在同一目录，完整包已按此组织。

## 新增无文本 α 质量解码

已完成 [完整测试集 α / argmax 对比](../ottc_alpha_full_20261017/RESULTS_ZH.md)：TIMIT192、Buckeye1840、Libri clean2620/other2938，共7,590条。下列交互图仍展示原先四条固定示例；全量数值、图表和W&B记录见新报告。

Recognition页的 `OTTC upstream-JSD (α)` 与原upstream-JSD固定同一识别音素序列，仅用音频预测的α及自身预测音素数Û设置β=1/Û，做逆CDF分数帧切分。没有输入参考音素序列、参考数量或参考时间。它可能重分配整句边界，并不只是微调现有argmax边界。静音仅来自原预测序列，不额外插入sil。统一固定20ms frame cell，末尾<tail>只补齐显示。

[解码数值报告](../ottc_alpha_decode_20261017/RESULTS_ZH.md)另比较同一α取整到20ms后的结果，区分质量重分段与帧内插值的影响。raw α、运输矩阵、逐帧后验、原识别序列、checkpoint与源码hash均保存在该目录；下载包内位于ottc_alpha_decode/。图中四例只是probe，不能代表全测试集。

## FA 输入条件

OTTC/Zapformer提供参考音素序列，不提供参考时间。MFA提供原始词转写，由发音词典生成音素。这两种输入条件不同，不能视作完全相同的FA任务。FA页不报告识别PER。

CTC、WavLM目前没有与这批模型匹配的已验证FA输出。纯CIF CE+Quantity没有逐帧辅助头，只提供数量条件分段，不填写K2 FA；另两种CIF已实际测试其辅助分支的K2 FA。`length`行只给定音素数量，标为Length F1；`K2 aux`行给定音素序列，标为FA F1。两者均未使用参考时间。三种CIF的recognition结果仍全部保留在Recognition页。

原页面拆分复用已有预测；随后本次CIF测试实际重新推理了9个既有checkpoint，未训练模型。12个模型×语音组合，共12条数量条件分段和8条辅助头K2 FA，后者0次失败。新MFA已经在上一次操作中实际完成：工具2.2.17，官方english_us_arpa声学模型/词典v3.0.0；不是论文MFA3.0复现。原始TextGrid、输入来源、版本、日志与hash见[实际MFA运行记录](../mfa_direct_20261017/README_ZH.md)；下载包内位于mfa_direct/。

## 示例和导出

|语料|ID|Recognition轨数|FA轨数|
|---|---|---:|---:|
|TIMIT|FDHC0_SI1559|11|7条序列FA＋3条CIF数量条件＋1条默认隐藏归档|
|Buckeye|s2101a_10|7|6条序列FA＋3条CIF数量条件|
|Libri clean|1089-134686-0001|7|6条序列FA＋3条CIF数量条件|
|Libri other|1688-142285-0002|7|6条序列FA＋3条CIF数量条件|

各split共同ID中时长2.5–4秒的第一条字典序ID；没有按结果挑例。全部只覆盖这四条可视化示例，不能代表完整测试集。

每个模式的全句/放大PNG及PDF位于 `subpages/recognition/figures/` 和 `subpages/fa/figures/`；每个模式各8张PNG、8份PDF。分开的Praat文件在同路径的 `textgrids/<split>/`；分开的区间JSONL、输入条件和摘要在 `alignments/`。对应WAV在 `interactive/audio/`。

此前的[混合对照页](interactive/combined.html)、`figures/`、`textgrids/`和 `examples.json`仍保留；新子页数据为 `recognition_examples.json`和 `fa_examples.json`。

CIF实测数值、严格加载与权重hash检查见[四例测试报告](../cif_fa_probe_20261017/RESULTS_ZH.md)，下载包中位于cif_fa_probe/。K2使用既有冻结IF-MDD后端：GPU编码器推理，CPU K2图搜索。仅四条示例，不作完整测试集结论。

## 时间与评分

TIMIT/Buckeye参考为人工音素时间，Libri参考为身份未核实的既有自动对齐。灰色虚线是参考边界，彩色线是当前模型边界。

Recognition保留固定20ms时钟（CIF保留分数帧触发），Zapformer保留原生duration/T时钟。K2 FA保留已保存的duration/frames时钟与blank归前音素策略；MFA保留原生时间戳。没有拟合偏移。Zapformer保留原生dx符号，OTTC/共同参考采用dx→d映射，主要比较边界。Zapformer FA保存了全部内部音素起点，图示首尾用音频端点补齐；内部评分点不变。CTC blank显式显示；CIF重复音素不合并，<tail>只补齐显示不新增评分边界。

STFT：16kHz音频、25ms Hann窗、5ms hop、512 FFT、相对最大功率−80至0dB。频谱时轴按真实STFT帧中心标定。

各行F1/R-value为所选整句、20ms容差，放大后不重新评分。全部模式使用相同可视化参考、唯一边界、距句首句尾20ms内剔除、一对一时间匹配；10/20/25ms的P/R/F1/R-value随JSON和HTML保存。R-value复用冻结IF-MDD公式并记录源码hash。归档CTM不填写分数。

这套图示协议与历史raw-PHN评分分开。实际MFA的raw-PHN结果另存在其RESULTS_COMPLETE.json，历史冻结raw-PHN scorer的hash也随SUBPAGES_COMPLETE.json保存。没有报告等音素数TSE/共同子集比较，也没有与论文MFA数字直接比较。

## 复现与检查

在已存在原始四例bundle的OTTC根目录运行：

```bash
/work/gm64/m64000/miniforge3/envs/sb_k2_cuda/bin/python -m reproduction.cif_comparison_20261016.segmentation_subpages
```

完整绘图入口 `python -m reproduction.cif_comparison_20261016.segmentation_figures` 也会在生成原始图后调用两个子页的导出。仅CPU绘图，无模型推理。

来源、FA覆盖和缺项见 `SUBPAGES_COMPLETE.json`；各模式精确TextGrid回放与语法检查见 `SUBPAGES_VALIDATION.json`；真实离线浏览器的页面切换、同一语音保留、模型勾选、高亮和区间播放检查见 `SUBPAGES_BROWSER_VALIDATION.json`。此前混合页的验证记录保留为历史记录。
