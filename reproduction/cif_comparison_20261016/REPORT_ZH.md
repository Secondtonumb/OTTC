# CIF 与历史单头 TIMIT 实验：统一口径评分

复用192条保存的预测，没有训练或推理。参考统一为历史core192，边界统一为20ms固定帧时钟、20ms边缘排除和旧一对一匹配；R-value复用冻结IF-MDD公式。CIF仍保留分数帧crossing，没有拟合偏移。

|模型|选择epoch|no-sil PER %|P %|Recall %|F1 %|R-value %|
|---|---:|---:|---:|---:|---:|---:|
|CTC|40|8.672|42.514|75.185|54.315|23.678|
|OTTC_CE|40|9.680|62.125|63.463|62.787|68.021|
|OTTC_upstream_JSD|40|9.645|64.698|66.037|65.361|70.247|
|WavLM_OTTC_upstream_JSD|40|9.732|70.376|72.178|71.266|75.289|
|CIF_no_ctc|10|14.581|59.664|61.057|60.353|65.900|
|CIF_ctc_0p5|24|10.167|59.232|58.022|58.621|64.867|
|CIF_ottc_jsd_0p5|11|15.867|58.434|57.714|58.072|64.333|

历史四组的PER、匹配数、参考数和预测数均精确复算通过。CIF源报告使用7307个参考音素，旧报告为7215；此处统一用旧参考重新计算，所以数值有别于CIF原报告。

保留原生duration/T时钟和改成固定20ms右帧边缘的敏感性结果都在metrics.json；不将原生时钟数字和固定时钟数字混报。

训练均为40轮。前三组历史模型和CIF均为Wav2Vec2-large-LV60，4620条TIMIT训练语音；历史模型用regfrozen（前2165步冻结声学transformer），CIF全程训练transformer。旧模型取最终checkpoint，CIF按最佳dev分段R选点，非严格仅改变loss的消融。历史CTC学习率5e-5，其他上述W2v实验2e-4。WavLM行是不同编码器，只作历史参照。

本次单seed结果尚未显示CIF分段超过历史原upstream-JSD OTTC。不能将CIF+OTTC辅助loss与原生OTTC目标直接等同；其归一化、decoder、编码器冻结策略均不同。

复现：`python -m reproduction.cif_comparison_20261016.rescore`。数据源、源码SHA256及逐语音匹配计数均随报告保存。
