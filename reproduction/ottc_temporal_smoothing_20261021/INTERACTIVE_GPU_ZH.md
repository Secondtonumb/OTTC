# Miyabi GPU 交互训练入口

在已有 PBS GPU 交互分配中运行，提前申请足够的 walltime：

```bash
cd /work/gm64/m64000/OTTC
bash reproduction/ottc_temporal_smoothing_20261021/interactive_gpu.sh
```

默认遍历完整 TIMIT train（4620条），训练40 epochs，每次优化器更新64条语音，最后不足64条的batch按实际条数归一化，不设steps上限。使用原始SSL初始化、native10ms、mono、upstream-JSD，平滑系数0.1、tau=4。0.1沿用原测试默认值，尚未调优。

首次训练自动对当前模型、loss、数据及BF16精度执行GPU profile，验证原始SSL初始化、冻结策略、梯度和最长输入，按吞吐与显存安全选择microbatch。64是完整optimizer batch；每个microbatch反向后才统一更新，不把microbatch当作独立优化器更新。profile后重新初始化正式模型。

```bash
bash reproduction/ottc_temporal_smoothing_20261021/interactive_gpu.sh \
  --epochs 40 --batch-size 64 --weight 0.1 --tau 4 \
  --output-dir reproduction/ottc_temporal_smoothing_20261021/training/my_run
```

每轮在dev上验证，按dev PER保留最佳3份独立模型权重（`best_weights/`、`BEST_THREE.json`）；另外只保留最新一份完整`checkpoint-*`，包含优化器、调度器、随机数状态及Trainer状态，用于续训。不保留所有epoch权重，不执行final-5/final-10平均。此策略只管理本次新训练目录的文件。

训练结束后，对保留的best-3在FP32下评测，先完成dev比较并选定模型，再评测test；不会用test选择checkpoint。分段、强制对齐、识别指标、预测及源码/数据hash全部本地保存。训练标量（包括每轮dev指标）写W&B，网络不可用时回退offline；profile和独立checkpoint比较不上传。

输出默认写入新的`training/<配置_UTC时间_PID>/`。同一配置续训：

```bash
bash reproduction/ottc_temporal_smoothing_20261021/interactive_gpu.sh \
  --resume --output-dir reproduction/ottc_temporal_smoothing_20261021/training/my_run
```

续训须使用与原训练相同的epochs、batch size、head、cost、weight、tau和源码。可用`--max-seconds N`在时间预算到达后保存暂停；默认不主动限时，PBS本身的walltime仍有效。默认每轮保存完整状态，节点被强制终止时可从最近已保存轮次续训。

只做profile：`--profile --output-dir 新目录`。profile目录用于检查本次配置，不代表训练已完成。

原有短测试显式使用`--probe`：

```bash
bash reproduction/ottc_temporal_smoothing_20261021/interactive_gpu.sh --probe --steps 1 --batch-size 2
bash reproduction/ottc_temporal_smoothing_20261021/interactive_gpu.sh --core-only
```

短测试保留原来的steps 0..20、batch size 1..8限制，只更新内存中的临时模型，不写checkpoint、不上传W&B。完整训练不受这些测试上限约束。脚本不会自动提交、抢占或结束PBS作业。
