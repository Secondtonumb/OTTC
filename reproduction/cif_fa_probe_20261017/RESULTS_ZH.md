# CIF 条件分段与辅助头 K2 FA：四例实测

本次实际运行9个已训练checkpoint，覆盖4条语音、12个模型×语音组合，未重训。共12条CIF长度条件分段和8条辅助头K2 FA；仅为可视化四例probe，不是完整测试集。

输入条件：K2辅助头给定音素序列；CIF长度条件只给定音素数量，音素内容不参与边界搜索；二者均不输入参考时间。纯CE+Quantity没有逐帧辅助头，不虚构K2结果。

共同目标沿用图中参考音素序列：Libri省去CIF训练目标中的首尾sil但不裁剪音频；native dx保留，检查共同标签时映为d。所有新预测时间均保留duration/valid_frames时钟，无拟合偏移。

下表为当前整句20ms容差的 **F1 / R-value（%）**。自由边界也从同一checkpoint在同一duration/frames时钟下重新计算，便于隔离长度条件的作用；因此不承诺逐位等于此前固定20ms图示数字。

|语音|模型|自由CIF边界|CIF给定数量|辅助头K2 FA|
|---|---|---:|---:|---:|
|timit__FDHC0_SI1559|CIF CE+Quantity|66.67 / 71.38|67.35 / 72.13|无逐帧头|
|timit__FDHC0_SI1559|CIF + CTC|63.92 / 69.34|71.43 / 75.61|46.94 / 54.71|
|timit__FDHC0_SI1559|CIF + OTTC|63.27 / 68.64|57.14 / 63.42|69.39 / 73.87|
|buckeye__s2101a_10|CIF CE+Quantity|71.60 / 75.70|69.88 / 74.38|无逐帧头|
|buckeye__s2101a_10|CIF + CTC|57.50 / 64.26|53.01 / 60.17|40.96 / 50.01|
|buckeye__s2101a_10|CIF + OTTC|65.85 / 71.04|50.60 / 58.14|53.01 / 60.17|
|libri_clean__1089-134686-0001|CIF CE+Quantity|64.41 / 69.81|38.60 / 49.08|无逐帧头|
|libri_clean__1089-134686-0001|CIF + CTC|41.38 / 50.96|35.09 / 46.26|42.11 / 51.91|
|libri_clean__1089-134686-0001|CIF + OTTC|61.02 / 66.97|35.09 / 46.26|66.67 / 71.57|
|libri_other__1688-142285-0002|CIF CE+Quantity|44.00 / 52.20|50.00 / 58.14|无逐帧头|
|libri_other__1688-142285-0002|CIF + CTC|53.06 / 60.36|58.33 / 64.91|37.50 / 47.95|
|libri_other__1688-142285-0002|CIF + OTTC|48.00 / 55.62|66.67 / 71.66|70.83 / 75.02|

辅助头K2对齐失败：0/8。K2路径逐条验证折叠后等于给定目标音素序列；权重严格加载、epoch/step核对、推理前后hash一致。

边界协议：共同图示参考的唯一端点，距语音两端20ms内剔除，一对一时间匹配；10/20/25ms的P/R/F1/R-value见BOUNDARY_METRICS.csv。R-value使用冻结IF-MDD实现。TIMIT/Buckeye参考为人工标注，Libri参考为模型身份未核实的既有自动对齐。

TIMIT另按历史冻结raw-PHN scorer复算，保存于predictions.jsonl的timit_raw_PHN_separate，scorer/PHN hash保存在EVAL_COMPLETE.json。raw-PHN与图示参考协议不同；没有执行共同等音素数TSE比较，不把样例数值与论文MFA基线直接比较。

文件：INPUTS.json记录输入与源模型；runs/各模型目录保存完整区间、逐帧K2路径、计时、显存、源hash和完成标记；telemetry.csv/job.log为实际GPU运行证据。checkpoints/是既有权重的硬链接，无重复权重副本。

复现：在PBS分配的GH200上执行 `bash reproduction/cif_fa_probe_20261017/evaluate.pbs`。已完成项跳过；没有优化器更新。W&B只记录数值摘要，不上传音频/转写/TextGrid附件。

[查看FA与CIF条件分段](../segmentation_figures_20261017/interactive/forced_alignment.html)。
