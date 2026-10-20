# Zapformer SPOTTC-JSD S3T2：补齐 recipe 的独立从头训练

2026-10-10 提交；随机初始化 seed42，完整约261M Zapformer，MFA 英语全量2888h，10ms输出，10 epochs，8 GPU DDP。
沿用上一组的 MFA packed Fbank、phone inventory、上游JSD和原S3T2 forward-sum。原实验输出保留。

## 当前作业

| 阶段 | Job | 启动条件 |
|---|---|---|
| 单卡 tiny/完整模型更新与新进程恢复验证 | 3521267 | GPU资源可用 |
| 八卡实际JSD/S3T2更新、模型一致性与各rank状态恢复 | 3521279 | 3521267正常退出且SMOKE_PASS存在 |
| 正式随机初始化训练 | 3521280 | 3521279正常退出且两级SMOKE_PASS存在 |

2026-10-11核对：3521267在10月10日14:42 JST以exit1结束；保存状态恢复的hash一致，但恢复后第二步的完整状态hash不一致。完整模型验证未执行。3521279和3521280因前置失败未启动，正式training/尚无训练checkpoint或W&B run。详见submission_status.json和smoke.log。
本地CPU小模型的骨干/优化器/增强/平均参数恢复检查通过；CPU使用代用目标，归档JSD只支持CUDA。

## 训练机制

- 原归档 BatchedRubik：beta1=.99，原矩阵更新、标量分支、参数尺度限制和mixed-dtype buffers；沿用icefall参数组构造。
- 原 InterpCosineLRScheduler：base LR=.02，linear_scale=.5，min_factor=.025；按真实epoch/batch进度衰减，每epoch实际步数偏差沿用scheduler调整。
- 恢复encoder辅助梯度：base aux_loss_scale=.05，global_step>2000时乘.25，再乘当前GradScaler scale；encoder内每stack/layer的原系数保留。
- 梯度使用每rank batch-sum再经DDP平均，移除旧实验额外clip5；不再用global sample mean缩放。日志同时保存全局sum、每utterance/frame损失、JSD与topology分量。
- 原AlternatingSpecAugment默认时间/频率mask及time warp。整批增强后再microbatch切分。每rank/epoch独立种子；私有CPU/CUDA Generator也进checkpoint。
- BF16 autocast及GradScaler(init_scale=1)，沿用低scale恢复规则；保存scaler状态。训练异常停止，不自动重复提交。
- 累计float64 model_avg每200 batches按icefall函数更新，和当前model都保存；最终分别评估last与累计平均，无test选择。
- phone/substate输出恢复ScaledLinear(initial_scale=.1)，质量头也以.1初始化；10ms前端投影保留。

每轮约4次checkpoint：初始sampler估计ceil(epoch_steps/4)，预计1722steps；实际数值写入run_config。轮末、每个进程启动/恢复后第100步、正常限时退出另存。
checkpoint包含模型、累计平均、optimizer、scheduler、各rank的scaler/consumed sampler/RNG/增强Generator、progress和完整源码合约。
每8h allocation预留约30min退出余量，未完成则自动续训同一W&B run；达到10轮提交最终TIMIT dev4096/test192及K2 FA。

## 复现范围

vendor保存本地审计过的zipa/icefall Zapformer源码，来源与hash见vendor/ORIGIN.json和runtime.sha256；不是最新upstream版本下载。
这里恢复适用的Zapformer训练机制，任务仍是MFA phone+10ms SPOTTC-JSD。未重新制作ASR recipe的3倍speed-perturbation数据；默认multicopy仍为1。
microbatch limits固定为(28000,40000000)，切分会影响相关性辅助项的batch统计，不能宣称与无切分ASR recipe逐位一致。

## 记录

独立目录training/和W&B group `zapformer_recipe_s3t2_20261010`；project/entity/mode读取仓库配置并支持环境覆盖。
训练开始时创建独立run ID，续训复用；网络失败保留offline日志。单卡smoke的W&B记录独立于训练。
逐卡GPU采样写gpu_rank*.csv；原始loss、LR、epochs、steps、辅助系数、scaler、显存、吞吐和dev指标写updates/dev JSONL与W&B。

[提交记录](submission_status.json) · [训练代码](train.py) · [模型与目标](model_s3t2.py) · [此前审计](../ZAPFORMER_TRAINING_AUDIT.md)
