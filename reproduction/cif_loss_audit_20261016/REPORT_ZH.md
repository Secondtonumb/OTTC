# CIF loss 收敛审计

2920个训练更新、40轮验证均完整，三组无非有限loss或梯度范数。总loss的定义不同，不跨组比较绝对大小。

|实验|train epoch30→40|相对变化|dev epoch30→40|相对变化|最低dev loss epoch|
|---|---:|---:|---:|---:|---:|
|no_ctc|0.8192→0.7870|-3.93%|1.0042→1.0076|+0.34%|22|
|ctc_0p5|0.8777→0.8271|-5.76%|1.1770→1.2084|+2.66%|21|
|ottc_jsd_0p5|1.7920→1.6697|-6.82%|2.3055→2.3004|-0.22%|40|

判断：训练稳定并已进入后期缓慢下降阶段；不能宣称所有训练项已完全收敛。验证已大体平台，CTC组后期回升，存在过拟合迹象。OTTC验证总loss第40轮仍为最低，但30→40轮只改善约0.2%，同时训练OTTC项仍下降。

OTTC辅助项 train/utterance：epoch30为1.8279、epoch40为1.6499；dev为2.5618→2.5582。CTC辅助项 train/phone：0.1504→0.1175；dev为0.3295→0.3656。

quantity的dev每句绝对计数误差在epoch40约0.97/0.97/1.02个phone（无辅助/CTC/OTTC）。quantity稳定不保证每个边界准确。CE使用0.1 label smoothing，其目标最小值不是0。

分段R-value最优发生于epoch10/24/11；dev loss最优发生于22/21/40。训练目标与分段指标的最优点不同，不能把较低loss等同于更好的分段。

历史CTC/CE/upstream-JSD的原始记录已一并保存到metrics.json。它们最后数轮eval_loss也趋于平台；但历史验证为kaldi_test240、loss定义和reduction也不同，不同CIF dev400做绝对值比较。

复现：`python -m reproduction.cif_loss_audit_20261016.analyze`。逐epoch CSV为原始记录的明确聚合，非平滑或插值；原始训练、checkpoint和W&B运行均未改变。
