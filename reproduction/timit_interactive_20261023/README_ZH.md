# Interactive TIMIT OTTC

在已分配的 GPU interactive 节点直接运行 Python。默认单头 Wav2Vec2 Large LV60 SSL 初始化、原始 upstream `jsd`（AG 语义）、native 10 ms、TIMIT 全量 train/dev/test。CNN 前六层冻结，最后一层 CNN、Transformer 和任务头可训练。此入口独立于历史实验和双头 FiLM recipe。

```bash
cd /work/gm64/m64000/OTTC
source reproduction/timit_interactive_20261023/environment.sh

python -u -m reproduction.timit_interactive_20261023.train \
  --output-dir reproduction/timit_interactive_20261023/runs/ag_debug01 \
  --cost jsd_upstream --head mono \
  --epochs 40 --batch-size 64 \
  --learning-rate 2e-4 --warmup-steps 500 \
  --weight-decay 0.005 --dropout 0.1 \
  --seed 42 --max-grad-norm 1.0 \
  --precision bf16 --microbatch 16 \
  --logging-steps 1
```

`--batch-size` 是每次 optimizer 更新的总句数；`--microbatch` 是单次 forward/backward 的句数，梯度累加由入口处理。`--microbatch 0` 自动 profile 后选吞吐最佳的安全候选；指定非零值也必须先通过相同模型/loss/精度的 GPU profile。`--precision fp32` 关闭 BF16 autocast；两种精度下模型参数、保存权重和最终平均累加都为 FP32。

命令加 `--max-seconds 120` 可在正式训练计时到达后保存并暂停；profile 和保存/评估时间不计入这 120 秒。保持原参数和目录并追加 `--resume` 续训，复用 optimizer/scheduler/RNG/W&B ID。改变训练超参数请更换 output-dir。`--profile` 只执行 GPU 检查，不进入正式训练；随后用同一命令去掉 `--profile` 可以正式开始。

可选 cost 为 `jsd_upstream`、`ce`、`jsd_standard`；前者是原始 AG 语义，后者是标准 JSD。`--head s3t2ss` 为独立 subphone topology 模式。当前单头 monophone 识别使用 argmax + collapse greedy；text-free 分段与 supplied-phone K2 FA 分开评分。默认每轮评估 dev，test 只用于训练完成后的评估。

每个完成的 epoch 原子保存 `epoch_weights/epoch-NNN.pt`，另存 best-dev 与最近两个完整 Trainer checkpoint。训练完成后评估 best/final/final5/final10（轮数不足则跳过相应平均），平均成员/epoch/step/hash 全部记录，选择只依赖 dev。标量训练曲线默认记录 W&B，profile 和最终评估仅存本地；预测和大文件不上传。

`history.jsonl` 保存训练 loss/LR/step/epoch；`development_metrics.jsonl` 保存 dev PER 与分段/FA P/R/F1/R-value/边界误差。配置和代码源哈希见 `run_config.json`；训练与最终评估分别以 `TRAIN_COMPLETE.json` / `BENCHMARK_COMPLETE.json` 为完成依据。

本次验证：Python `--help` 加载、语法、自定义参数映射、原始 AG 分支映射、CPU 原子保存、final5/final10 的 FP32 平均与成员记录、FP64 保存拒绝均通过。未在 GPU 上启动本入口；首次启动自动执行实际 GPU gate。
