两个 monophone 对照：完成120步短程验证

均为完整原生 Zapformer、原始 archived OTTC-JSD、原生 BatchedRubik、BF16、FS=0、固定四句真实 Libri100 音频、相同seed42、无SpecAugment、未缩放aux=.05。LR1=.0171189578476，LR2=.0342379156952。未改动原生OTTC divergence或mass规则，也未改任何已有50轮模型和冻结源码。

| case | 初始JSD/句 | 最终JSD/句 | 最终固定四句原生音素PER% | 最大单帧mass |
| --- | --- | --- | --- | --- |
| mono105_seed42_lr1 | 6.175851 | 5.911585 | 137.85 | 0.0032 |
| mono105_seed42_lr2 | 6.175851 | 3.714387 | 2.26 | 0.0030 |
| pooled313_jsd10_lr1 | 6.010915 | 5.856006 | 309.60 | 0.9984 |
| pooled313_jsd10_lr2 | 6.010915 | 5.823439 | 337.29 | 0.9998 |
| sp313_seed42_lr2 | 6.203339 | 4.920950 | 1.69 | 0.0028 |

主结论：真正105维的seed42随机初始化在LR2下能学会固定小批次（PER约2.26%）；相同backbone/mass初始化的SP对照使用原生三状态完整解码也能学会（PER约1.69%，音素marginal解码为3.39%）。因此短程证据不支持“SP本身无法学习”。基准LR1的真正105模型在120步内仍学习不足，这也说明LR敏感性。

313合并方案的原生参数和优化器完整复用第10轮checkpoint，只取消SP目标展开，并以logsumexp合并三个状态的输出。该方案仍然表现很差，mass集中到极少帧。真正105模型是独立随机初始化；两个方案的历史不同，不能以跨库存的raw loss绝对值比较模型优劣，也不能从这个对照单独证明mass是唯一原因。继承旧checkpoint的bad state是优先排查方向。

INVARIANTS_PASS.json验证：合并后posterior等于原三个state的概率之和；313参数与旧checkpoint逐位一致；真正105与SP的seed42 backbone/mass逐位一致；真正输出head为105×384。GPU作业3546461.opbs正常结束（单GPU、7分44秒）。原始归档OTTC函数依赖CUDA，所以完整loss/backward在计算节点完成。

这只是训练集四句的过拟合诊断，不是LibriSpeech识别基准或TIMIT跨语料成绩。随机初始化105模型通过后进入完整8卡DDP/断点一致性验证；313合并方案保留为失败诊断，不立即投入完整50轮。正式105训练重新从seed42初始化，不使用过拟合后的四句权重。

每个case使用独立W&B run；checkpoint.pt保留模型、优化器、RNG及诊断配置，可在完全相同配置下恢复，沿用其W&B ID。已有 COMPLETE.json 的case不会重复训练。所有原始点、初始化hash、mass分布与音素序列见report.json；未插值。

复现：source ../zapformer_s3t2ss_native_20261012/environment.sh 后，在GPU上执行 python diagnose.py --steps 120；PBS入口diagnose.pbs。旧结果保留原样，新的全量复现使用新的输出目录。regular-g覆盖提交记录：qsub -q regular-g diagnose.pbs。
