# TIMIT 散度对照：interactive 单卡训练

在已经分配好的 GPU interactive 终端运行。脚本不提交 PBS 作业。

```bash
cd /work/gm64/m64000/OTTC

# 每条命令是独立实验；同一张 GPU 上逐个运行。
bash reproduction/timit_divergences_20261022/train_kl.sh
bash reproduction/timit_divergences_20261022/train_jdiv.sh
bash reproduction/timit_divergences_20261022/train_kl_reverse.sh

# 可选对照：AG 和真正的标准 JSD。
bash reproduction/timit_divergences_20261022/train_ag.sh
bash reproduction/timit_divergences_20261022/train_jsd_standard.sh
```

默认：TIMIT 全量 train 4620 / dev 400 / core-test 192；原始
`facebook/wav2vec2-large-lv60` SSL encoder 初始化、全新单头，native 10 ms，
前六层 CNN 冻结、最后一层 CNN 和 Transformer/两个任务头可训练，
`forward_type=regular`，无 temporal smoothing。40 epochs，逻辑 optimizer batch 64，
LR `2e-4`、warmup 500 steps、BF16、seed 42。各实验仅散度不同。

`kl` = 预测到目标的 KL；`kl_reverse` = 目标到预测的 KL；`jdiv` = 两个方向
KL 的和，不除以 2。KL/J 的目标用 `1e-12` 下限并归一化。`ag` 精确复用旧
`jsd` 的 AG 公式与 ε 处理；`jsd_standard` 是标准 Jensen-Shannon。
不同散度的原始 loss 尺度不可直接比较。

训练前自动用**当前散度、真实 TIMIT、相同精度和模型** profile，比较 microbatch
吞吐与显存，确认 SSL 权重、梯度、识别/对齐路径；正式模型重新初始化，不继承
profile 的更新。逻辑 batch 仍然是 64，最后不足 64 的 batch 按真实样本数归一化。

按本次要求减少保存/评估频次：

- 第 5、10、15、20、25、30、35、40 轮：保存独立模型权重、完整最新续训状态，
  并做一次 dev 识别及分段评估（PER、P/R/F1/R-value，区分 text-free / supplied-phone FA）。
- 第 38、39 轮：额外只保存模型权重，不触发 dev 评估。
- 结束后：平均第 **38、39、40** 轮的模型权重，保存成员路径、epoch、step 和 SHA256。
  比较 best-dev 单模型与该平均模型：先完成两者 FP32 dev 评估并选定，再分别评估 test。
  不计算 final5/final10 平均，不在中途反复跑 test。
- 保留训练 dev-PER 最佳三份；独立 epoch 权重保留上述 10 个轮次。
  HF `checkpoint-*` 只保留最新一份完整续训状态。

训练标量及配置默认记录 W&B。GPU 节点网络不通时自动回退 offline；可以显式设置
`WANDB_MODE=offline` 或 `WANDB_MODE=disabled`。Profile、最终 checkpoint 比较、
预测、音频、表格和评估报告仅保存在本地，`OTTC_EVAL_UPLOAD=0`。

默认输出目录固定，便于续训：

```text
reproduction/timit_divergences_20261022/training/
  wav2vec2__timit__10ms__mono__kl__e40_b64/
    run_config.json              # 源码/data/目标/训练参数合同
    source_snapshot/             # 所引用源码的冻结副本
    profile/PASS.json            # 精确配置的 GPU gate
    history.jsonl               # loss、step、epoch、LR
    development_metrics.jsonl   # 每5轮 dev 指标
    epoch_weights/              # epoch-005…035、038、039、040.pt
    best_weights/               # 最佳三份权重
    checkpoint-*/               # 最新 optimizer/scheduler/RNG/Trainer 状态
    averaged_weights/last3.pt    # 默认38/39/40的均值
    averaged_weights/last3.json  # 成员与hash
    DEV_SELECTION.json
    benchmark_*_{dev,test}.json
    benchmark_*_{dev,test}.predictions.jsonl
    TRAIN_COMPLETE.json
    BENCHMARK_COMPLETE.json
```

续训同一个实验（保留配置、输出目录和 W&B ID）：

```bash
bash reproduction/timit_divergences_20261022/train_kl.sh --resume

# 有主动时间预算时，在完成当前 optimizer step 后保存并退出。
bash reproduction/timit_divergences_20261022/train_jdiv.sh --max-seconds 18000
bash reproduction/timit_divergences_20261022/train_jdiv.sh --resume --max-seconds 18000
```

Ctrl-C / SIGTERM 也会请求在当前 optimizer step 结束后保存状态。SIGKILL/节点硬中断
只能恢复到已完成的最新 checkpoint。`--max-seconds` 从正式训练开始计时，
不含 profile 和最终评估，也不是硬 walltime 限制。

只读检查可在登录节点执行，不启动训练、GPU profile 或 W&B：

```bash
bash reproduction/timit_divergences_20261022/train_kl.sh --dry-run
bash reproduction/timit_divergences_20261022/train_jdiv.sh --help
```

需要先只 profile 时使用 `--profile`；随后原命令启动正式训练即可。可用
`--output-dir PATH` 指定新的独立目录，续训时使用相同路径并加 `--resume`。
可设置 `--epochs`（至少3）、`--batch-size`、`--eval-every`；改变后属于新实验，
输出名记录 epochs/batch，配置合同也会阻止误续训。默认40轮平均38/39/40；
自定义 epochs 时平均最后三轮。

脚本按单 GPU 编写。若 allocation 暴露多张卡，请先设置
`CUDA_VISIBLE_DEVICES` 为其中一张；也可设置 `OTTC_PYTHON` 覆盖默认环境。
