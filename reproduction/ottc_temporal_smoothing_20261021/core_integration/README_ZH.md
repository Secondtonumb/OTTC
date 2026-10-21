# 直接在 ot_loss.py 接入时序平滑

按用户本轮指定，修改 `/home/m64000/work/OTTC/ottc/ops_loss/ot_loss.py`，其真实路径为 `/work/gm64/m64000/OTTC/ottc/ops_loss/ot_loss.py`。

`batched_ottc_loss_bucketized` 增加三个仅限关键字参数：`smooth_weight=0.0`、`smooth_tau=4.0`、`smooth_mask=None`。已有位置参数及四元组返回合同不变。默认0不计算、不添加正则，原始CE/upstream-JSD/standard-JSD/Euclidean cost保持原样；λ>0时在各分支的OTTC cost上加正则，运输索引及质量仍原样返回。

```python
from ottc.ops_loss.ot_loss import batched_ottc_loss_bucketized

loss, index_logits, index_labels, transport = batched_ottc_loss_bucketized(
    logits, one_hot_labels, alpha, beta,
    bmask=labels_mask,
    jsd=True,                       # 原始 upstream-JSD
    smooth_weight=0.1,              # 示例系数，未调优
    smooth_tau=4.0,                 # 平方差上限16
    smooth_mask=output_mask,        # encoder输出帧的[B,T]有效mask
)
```

平滑项从尚未flatten的完整`[B,T,C]`分类logits计算：FP32 log-softmax，参考前一帧detach，逐类平方差截断，要求相邻两帧都有效。每条语音先按有效帧对及类别数平均，然后batch求和，以匹配原OTTC归约并支持microbatch相加。独立的`smooth_mask`只作用于正则，不改变OT运输路径；没有提供时可以复用显式`amask`，两者均缺失则报错，避免静默把padding算进去。不会用学习得到的α是否为零来猜有效帧。

独立函数也可从此模块导入：`from ottc.ops_loss.ot_loss import temporal_smoothing_loss`，其默认归约仍是整个batch的有效帧对平均。

前一轮新增的可选模型扩展仍可使用：它调用底层默认关闭的OTTC，然后只在模型层加一次正则，不会重复计算。选择直接调用本文件的新参数时，无需再在外部加一遍。现有模型不会自动启用；要在其原调用处传入这些参数。

本次直接修改共享底层文件，源码hash会变化。未改写任何已保存训练/resume合同、profile门控、已完成源码快照或模型权重；新目标的新实验仍需独立输出目录、配置及相应profile。历史完成标记不会被重写为新源码已验证。

修改前文件的精确内容保存在`ot_loss_before_temporal_smoothing.py`，由git HEAD恢复并与前一轮验证记录中的修改前SHA256核对，来源见`BASELINE.json`。回归测试在CPU内存中仅将原CUDA分配字面量改成CPU分配，不改变公式；生产代码保留原设备合同。测试核对默认关闭时四种cost的loss/梯度/运输信息逐位一致，以及启用后的loss/梯度、mask与原20项模型测试。全部测试本地运行，不上传W&B，不启动长训。
