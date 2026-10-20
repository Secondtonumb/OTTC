# CTC、纯 OTTC 与 CIF：模型全集比较

本轮主表纳入单头 Wav2Vec2-large-LV60：CTC、纯OTTC CE、原 upstream-JSD、standard-JSD、CIF CE+Quantity、CIF+CTC、CIF+OTTC；另列 WavLM 与 scratch Zapformer。仅重用训练/预测/评分产物，没有新训练或推理。

主表均为无文本识别与分段，PER去sil；20ms容差、固定20ms时钟、同一参考、一对一边界匹配及冻结IF-MDD R-value。CIF保留分数帧crossing。各语料内比较，不把不同语料指标汇总成排名。

历史单头CTC/OTTC固定第40轮；CIF按dev R-value选单点。冻结日程、decoder、重复音素/静音处理不同，因此是现有完整配方比较，非单因素消融。全部单seed。Libri边界为自动对齐（身份未验证），Buckeye为人工边界。Libri CIF已统一到保留重复音素的参考，故PER有别于旧CIF报告。

## TIMIT → core-test192

|模型|PER↓|P|Recall|F1↑|R-value↑|
|---|---:|---:|---:|---:|---:|
|CTC|8.672|42.514|75.185|54.315|23.678|
|OTTC CE|9.680|62.125|63.463|62.787|68.021|
|OTTC upstream-JSD|9.645|64.698|66.037|65.361|70.247|
|OTTC standard-JSD|18.491|63.351|60.400|61.840|67.710|
|CIF CE+Quantity|14.581|59.664|61.057|60.353|65.900|
|CIF CE+Quantity+CTC|10.167|59.232|58.022|58.621|64.867|
|CIF CE+Quantity+OTTC|15.867|58.434|57.714|58.072|64.333|

## WavLM supplementary → core-test192

|模型|PER↓|P|Recall|F1↑|R-value↑|
|---|---:|---:|---:|---:|---:|
|WavLM OTTC CE (batch128)|10.045|63.676|64.638|64.154|69.267|
|WavLM OTTC upstream-JSD (batch64)|9.732|70.376|72.178|71.266|75.289|

## Buckeye → test1840

|模型|PER↓|P|Recall|F1↑|R-value↑|
|---|---:|---:|---:|---:|---:|
|OTTC CE|20.188|49.919|54.271|52.004|57.474|
|OTTC upstream-JSD|19.635|51.587|56.788|54.063|58.971|
|OTTC standard-JSD|21.152|51.359|56.190|53.666|58.781|
|CIF CE+Quantity|20.225|52.991|51.246|52.104|59.494|
|CIF CE+Quantity+CTC|20.632|54.840|54.997|54.919|61.486|
|CIF CE+Quantity+OTTC|20.793|54.306|54.333|54.320|61.003|

## Libri100 → test-clean2620

|模型|PER↓|P|Recall|F1↑|R-value↑|
|---|---:|---:|---:|---:|---:|
|OTTC CE|3.156|74.012|72.043|73.014|77.014|
|OTTC upstream-JSD|3.353|77.398|75.388|76.380|79.842|
|OTTC standard-JSD|3.258|76.590|74.594|75.579|79.169|
|CIF CE+Quantity|6.242|54.424|53.297|53.855|60.846|
|CIF CE+Quantity+CTC|6.115|53.263|51.375|52.302|59.685|
|CIF CE+Quantity+OTTC|4.969|55.528|53.245|54.363|61.455|

## Libri100 → test-other2938

|模型|PER↓|P|Recall|F1↑|R-value↑|
|---|---:|---:|---:|---:|---:|
|OTTC CE|5.986|70.915|68.944|69.916|74.413|
|OTTC upstream-JSD|6.223|74.423|72.386|73.390|77.330|
|OTTC standard-JSD|6.051|73.678|71.566|72.607|76.672|
|CIF CE+Quantity|8.815|55.306|53.916|54.602|61.519|
|CIF CE+Quantity+CTC|9.911|53.917|51.488|52.674|60.081|
|CIF CE+Quantity+OTTC|8.173|55.487|52.773|54.096|61.292|

## Libri100 → Buckeye zero-shot1840

|模型|PER↓|P|Recall|F1↑|R-value↑|
|---|---:|---:|---:|---:|---:|
|OTTC CE|31.473|50.697|54.579|52.566|58.203|
|OTTC upstream-JSD|31.148|54.394|59.318|56.749|61.577|
|OTTC standard-JSD|29.932|52.687|57.056|54.784|60.016|
|CIF CE+Quantity|32.470|50.058|54.516|52.192|57.596|
|CIF CE+Quantity+CTC|32.534|50.265|52.206|51.217|57.762|
|CIF CE+Quantity+OTTC|30.652|49.885|53.046|51.417|57.469|

## Scratch Zapformer native protocol → core-test192

|模型|PER↓|P|Recall|F1↑|R-value↑|
|---|---:|---:|---:|---:|---:|
|Zapformer upstream-JSD (10ms)|39.252|49.169|56.964|52.780|56.248|

WavLM CE使用batch128/1480步，JSD使用batch64/2920步，不能将差异全部归因于loss。scratch Zapformer为10ms、从零训练，表中保留其原生协议，不能与预训练Wav2Vec2主表直接做架构优劣归因。

## Checkpoint平均（单独的给定音素FA结果）

Libri100三种纯OTTC均已完成last3、dev FA best3和dev PER best3；成员仅来自保留的epoch39–40共5个step checkpoint。这不是最后3个epoch平均，也不是last-5/last-10。下表FA使用参考音素输入，与前面text-free分开。

|目标|平均策略|测试集|PER↓|FA F1↑|FA R-value↑|
|---|---|---|---:|---:|---:|
|ce|last|test_clean|3.156|60.951|67.008|
|ce|last|test_other|5.986|57.946|64.560|
|ce|last|buckeye_test|31.473|47.960|55.895|
|ce|last3|test_clean|3.162|60.962|67.017|
|ce|last3|test_other|5.970|57.964|64.575|
|ce|last3|buckeye_test|31.442|47.995|55.924|
|ce|dev_fa_best3|test_clean|3.167|61.078|67.113|
|ce|dev_fa_best3|test_other|5.961|58.075|64.666|
|ce|dev_fa_best3|buckeye_test|31.540|48.050|55.970|
|ce|dev_per_best3|test_clean|3.167|61.078|67.113|
|ce|dev_per_best3|test_other|5.961|58.075|64.666|
|ce|dev_per_best3|buckeye_test|31.540|48.050|55.970|
|jsd_upstream|last|test_clean|3.353|62.970|68.670|
|jsd_upstream|last|test_other|6.223|60.127|66.348|
|jsd_upstream|last|buckeye_test|31.148|49.651|57.320|
|jsd_upstream|last3|test_clean|3.347|63.040|68.728|
|jsd_upstream|last3|test_other|6.241|60.224|66.427|
|jsd_upstream|last3|buckeye_test|31.151|49.686|57.350|
|jsd_upstream|dev_fa_best3|test_clean|3.349|63.261|68.910|
|jsd_upstream|dev_fa_best3|test_other|6.248|60.352|66.532|
|jsd_upstream|dev_fa_best3|buckeye_test|31.272|49.746|57.401|
|jsd_upstream|dev_per_best3|test_clean|3.349|63.164|68.830|
|jsd_upstream|dev_per_best3|test_other|6.238|60.261|66.458|
|jsd_upstream|dev_per_best3|buckeye_test|31.245|49.737|57.393|
|jsd_standard|last|test_clean|3.258|61.886|67.778|
|jsd_standard|last|test_other|6.051|58.940|65.375|
|jsd_standard|last|buckeye_test|29.932|49.606|57.283|
|jsd_standard|last3|test_clean|3.255|61.934|67.817|
|jsd_standard|last3|test_other|6.032|58.983|65.410|
|jsd_standard|last3|buckeye_test|29.968|49.625|57.299|
|jsd_standard|dev_per_best3|test_clean|3.248|61.925|67.810|
|jsd_standard|dev_per_best3|test_other|6.048|58.970|65.400|
|jsd_standard|dev_per_best3|buckeye_test|29.986|49.523|57.213|
|jsd_standard|dev_fa_best3|test_clean|3.248|61.967|67.845|
|jsd_standard|dev_fa_best3|test_other|6.023|59.019|65.440|
|jsd_standard|dev_fa_best3|buckeye_test|30.011|49.577|57.259|

CIF九组只保存best/latest，不能做last-5/last-10平均。

## 其他已完成系列的入口

- 四头联合训练及各phoneset、给定音素FA：../english_suite/RESULTS.md。其标签体系和训练任务独立，不能将四个head冒充四个独立模型或与单头主表混排。
- 四头checkpoint比较：../checkpoint_comparison/README.md。
- Soft-DTW诊断：../RESULTS_SOFTDTW_ZH.md；该配置出现严重删除，独立保留。
- Libri100平均权重的成员、hash、PFER和完整FA：../libri100_singlehead_average_20261016/REPORT_ZH.md。

Buckeye/Libri100没有本次同配方单头CTC对照；不拿四头CTC替代。错配的Quantity=0历史实验已排除。归档MFA CTM身份未核实，不列为MFA 3.0基线。

来源与校验：metrics.json保存输入hash和逐组完整指标；metrics.csv可直接导入表格。原训练曲线和W&B run均保留，不插值、不改变原时间轴。

复现：python -m reproduction.cif_comparison_20261016.all_models。
