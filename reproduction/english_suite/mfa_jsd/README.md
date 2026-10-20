# Zapformer OTTC：复用已复现的上游 JSD

原 MFA Zapformer 的 OTTC 只有 CE。这个独立入口直接调用之前 JSD 复现归档中的
[`batched_ottc_loss_bucketized(..., jsd=True)`](../../provenance/js_source/ottc/ops_loss/ot_loss.py)，
完整复用它的 transport、JSD 代价与 sum reduction，没有重写 loss 公式。
归档中的 `jsd=True` 是上游实现，不是数学标准 JSD；不调用 IF-MDD 另一个标准 JSD 版本。

[`loss.py`](loss.py) 仅负责加载归档与适配 MFA 类别；[`model_jsd.py`](model_jsd.py)
复用现有 Zapformer OTTC forward。现有 CE 模型、训练源文件、checkpoint 和续训合约保持原样。

MFA 的 105 个输出 ID 中，1/2 是非声学控制符。原 forward 将其 logits 设为 -1e4，
上游 `log(softmax)` 会产生非有限 loss。适配器只在 loss 中删除这两列；
blank 0、声学 phones 3..102、silence 103/104 保留。输出仍为 105 类，
target 的外部 ID 和 10ms 输出网格保持原样，不改散度、不做概率 clipping。
如果有效声学类别也出现 softmax 下溢，原训练器会以 Nonfinite loss 停止，不能宣称原实现数值稳定。

## 验证

[`smoke.py`](smoke.py) 在 GPU 上验证：与归档 loss/梯度逐项一致、独立 dense transport
参考、padding/control 梯度、原 control-mask 非有限问题，以及 tiny/完整 Zapformer 的
bf16 forward/backward、encoder/classifier/duration-weight 梯度与一次 optimizer update。
smoke 单独写入 W&B，不能当作正式训练或收敛结果。

验证结果保存在 `validation/SMOKE_PASS.json`，作业日志为 [`smoke_job.log`](smoke_job.log)。

2026-10-09 验证作业 `3512964.opbs` 已完成，Exit status 0 / PASS。
tiny 与完整 261,367,784 参数 Zapformer 均通过 forward/backward/update，
loss 与梯度在同一输入上和归档实现逐项一致。
[W&B 验证记录](https://wandb.ai/kevingenghaopeng-/ottc-mfa-english/runs/n6x02go0)。

## 训练入口

[`train_jsd.py`](train_jsd.py) 使用原八 GPU DDP trainer，加上独立 JSD loss 合约、
归档源文件 SHA256、W&B 与初始/逐轮 dev PER、text-free segmentation F1。
checkpoint 保存原 optimizer、scheduler、sampler、各 rank RNG；恢复时严格检查合约，
因此 CE optimizer checkpoint 不能误作为 JSD 续训。当前配置从 seed42 开始训练。

[`train.pbs`](train.pbs) 沿用 MFA English 数据、8 GPU、LR .02、10 epochs、
每 rank duration budget 315s、每 epoch 4 次 checkpoint。独立输出目录为
`english_suite/mfa_models/ottc_jsd_upstream/`。正式八卡首段 `3515322` 已正常完成并保存
step 18688（第 3/10 轮），续训 `3520102` 已排队，实时状态见 [TRACKING.md](../TRACKING.md)。
每段到达墙钟限制会保存，再提交同一脚本可恢复该 JSD run。

```bash
qsub reproduction/english_suite/mfa_jsd/train.pbs
```
