# TIMIT topology 多版本图与统一复算

复用保存预测，未训练、未推理；完整core-test192，7149个参考边界。当前六组12项（无文本/FA）结果精确复算通过。历史11版本重新对同一参考评分。

打开 interactive/index.html 看汇总；recognition.html / forced_alignment.html 可听、缩放、选择区间。默认显示本次六组，历史轨手动勾选。示例沿用先前按ID/时长挑选的 FDHC0_SI1559，不按模型好坏挑选。

协议：唯一边界点、排除音频首尾20ms、排序一对一10/20/25/30ms容差、冻结IF-MDD R-value。非FA本次统一固定10ms；FA保留duration/frame_count与blank归前音素；旧20ms与CIF分数帧保留，Zapformer保留原生时钟。没有拟合偏移。

选择：边界F1/R按dev边界R选；FA MAE按dev MAE选；PER按dev PER选。候选仅best-dev/final/final5/final10，测试集不参与选点。CSV中的PER来自识别所选权重，与边界所选权重可能不同，勿视为同一checkpoint。

历史模型：TIMIT40轮旧配方、不同冻结/学习率/batch/backbone；当前10ms是同一Libri960 parent再迁移TIMIT10轮。相同参考与评分只保证benchmark口径，不能由此把差值归因于hop/topology/loss单一因素。CIF使用此前dev所选权重，未重选、未重训。

FA图的历史length行只给定数量，aux行是辅助头K2；MFA是词转写输入；其余OTTC是给定音素。参考时间只用于评分。归档CTM身份未核实，不称为论文MFA3.0；本次没有重算MFA raw-PHN/TSE或声称超越论文MFA。

图与Praat仅导出这一条固定示例：recognition17模型轨，FA17模型轨。alignments/每文件1句；全量复算的metrics.json、CSV、per_utterance_metrics.jsonl分别标明192句范围。

可复现：`python -m reproduction.topology_comparison_figures_20261019.build`。source_manifest.json记录输入及源码SHA256。paired_changes图复用既有4000次配对bootstrap。

|当前目标|无文本Mono/S3 F1|FA Mono/S3 F1|FA Mono/S3 MAE ms|
|---|---:|---:|---:|
|upstream-JSD|58.31/51.03|56.07/49.60|27.79/40.49|
|CE|57.42/48.57|54.70/48.66|29.12/42.50|
|standard-JSD|56.55/47.13|50.10/48.25|31.52/44.23|

全部10/20/25/30ms的P/R/F1/R见 boundary_tolerances.csv。实际浏览器验证见 VALIDATION_PASS.json；W&B仅数值与汇总图，backfill run: https://wandb.ai/kevingenghaopeng-/ottc-timit-reproduction/runs/f7q3d15l 。源码、清单、逐句预测、音频均仅保留本地。
