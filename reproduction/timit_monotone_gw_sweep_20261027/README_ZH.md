# 单调 Γ + GW：四种代价 × 三个 seed

用户于 2026-10-27 确认范围：CE、原 upstream AG、标准 JSD、Jeffreys/J-div，seed43/44/45。
共12组。复用 `reproduction/timit_gw_20261026/models/monotone_w0.1__seed43` 的完整40epoch结果；只新训其余11组。
所有不加GW基线来自 `reproduction/timit_seed_sweep_20261022/models/{cost}__seed{seed}`，不重训已完成CE/AG等模型。
ASOT 两个已停止任务不属于本矩阵，不会重启。

## 训练条件

Wav2Vec2 Large 原始SSL初始化，native10ms，单头，40epoch，global batch64，LR2e-4，warmup500，weight_decay0.005。
每个独立实验一个节点/一个GPU，world_size=1；不跨div/seed同步梯度。
模型参数、保存权重、epoch平均累积FP32；主干BF16 autocast。
不同条件仅更换已有 `bins_metric` 与 seed，其余保持相同。

L = 原 OTTC 匹配代价 + 0.1 R(Γ)。Γ取原 bucketized OT 的 indices/weights，保留严格单调与目标边缘质量。
R=0.5*N^2/有效有向邻帧对数 * <Γ,AΓ(11'-I)>；左右各20ms，排除自身/padding。
目标列为occurrence，重复音素保留独立列。没有ASOT重求计划、CE辅助、CTC或后验smoothing。
CE/AG/标准JSD/J-div全部调用仓库既有函数；AG仍是原upstream jsd，不是标准JSD；J-div不带1/2。
四组GW权重固定0.1，不按test调整。基础代价尺度不同，所以不能称为各div分别调优后的最优结果。

核心新模型 `ottc/models/sequence/wav2vec2_ottc_monotone_gw.py`。
数学结构仍复用 `ottc/ops_loss/gw_transport.py`，原历史实现和完成产物不变。
AG与历史方案2的loss/梯度已做等价测试；全部四种cost经过实际BF16 Trainer optimizer检查。

## 运行及验收

先每组运行真实longest64/representative64/tail12前向+反向+optimizer profile，验证SSL初始化、参数梯度、保存权重dtype，
再跑原scorer的小样本text-free/FA检查。profile PASS 的源码/config必须完全匹配后才能正式训练。
每epoch原子保存独立权重；保留best-dev和最新完整optimizer/scheduler/RNG resume；W&B ID可恢复。
完成后评估best-dev/final/final5/final10，dev选配置，test单独报告。训练标量W&B，其余评估及文件local-only。

```bash
source reproduction/libri960_continue5_20261017/environment.sh
"$LIBRI_PYTHON" -m pytest -q tests/test_gw_transport.py reproduction/timit_monotone_gw_sweep_20261027/test_integration.py
qsub -v GW_RECIPE=ce__seed43 reproduction/timit_monotone_gw_sweep_20261027/run.pbs
```

`ag__seed43` 训练入口明确拒绝重训：此项直接复用。
提交清单 `SUBMISSIONS.json`；实验矩阵 `MATRIX.json`；数据与12组基线核对 `DATA_PREFLIGHT.json`。

## 汇总

`report.py` 输出 `RESULTS.json`，每seed保留：无GW best-dev、加GW best-dev、开发集选中的平均/单模型，以及匹配的差值。
逐项核对400句dev/192句test预测覆盖、唯一utterance与reference哈希。
每个div必须3/3完整才生成均值与sample SD；缺失/排队/训练中的seed不能被混入均值。
报告PER/PFER、text-free P/R/F1/R-value@20ms、单独FA R/MAE、插入删除、短ABA、短预测片段、短参考音素删除率。
历史AG43所缺PFER在本目录派生缓存计算，不改写完成的原指标文件。
