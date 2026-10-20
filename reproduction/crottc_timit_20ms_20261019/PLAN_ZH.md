# TIMIT 20 ms monophone CROTTC：实现盘点与三组候选

历史草案：用户随后已明确三组为 CE / upstream JSD / standard JSD，并授权接入 IF-MDD 语音增强。本文件保留最初盘点过程；最终实现和归约决定见 `README_ZH.md`，实际提交/运行状态见 `dispatch.json` / `STATUS.json`。

## 已有实现

- `../..` 下的 OTTC 当前源码没有 CROTTC 训练分支。已有 CE / upstream JSD / standard JSD 是 OTTC transport cost，不包含双视图 CR。
- 相邻仓库 `/work/gm64/m64000/IF-MDD/utils/LossManager.py`，242–288 行有 CROTTC loss；528–556 行有双向 stop-gradient KL。
- `/work/gm64/m64000/IF-MDD/models/phn_mono_ssl_model_v3_refactored.py`，3434 行起在训练时生成两次独立增强并拼接为 2B，随后共用编码器前向；验证/测试不做双视图增强。
- `/work/gm64/m64000/IF-MDD/hparams/phnmonossl_crottc.yaml` 使用 DropFreq + DropChunk；SpeedPerturb 在复制两视图之前共享执行。不能对两视图分别改变语速后直接逐帧算 KL。

历史代码对两视图合并的 OTTC batch sum 乘 0.5，再加 `cr_loss_weight * CR`；权重缺省 0.1。该 CROTTC 分支写死 `jsd=False`，因此只直接覆盖 CE transport。JSD 两组需要显式接入本仓库已存在的分支，不应把 IF-MDD loss 文件整体替换成本仓库的 upstream JSD。

历史 CR 分母为 `B*T*C - padding_frame_count`，padding mask 的末维是 1；它不等于有效元素数 `(valid_frame_count*C)`。而 OTTC 部分是 batch sum，CR 部分是元素平均。迁移时必须明确记录是否精确保留历史归约，以及 logical batch / microbatch 下 CR 的缩放；不能不作说明就更改分母、复用 0.1 并声称等价。此处尚未选择或实现新的 loss 组合。

## 三组候选

| 候选名称 | OTTC transport cost | 一致性项 | 当前可用性 |
|---|---|---|---|
| CROTTC-upstream-JSD | 本仓库原上游 `jsd=True` | 相同双视图 CR | 需接入；优先报告 |
| CROTTC-CE | 本仓库 `jsd=False` | 相同双视图 CR | IF-MDD 有历史参考，OTTC 需接入 |
| CROTTC-standard-JSD | 本仓库 `jsd='standard'` | 相同双视图 CR | 需接入；与 upstream 分开 |

暂拟共用同一个 Wav2Vec2-Large-LV60 原始 SSL 初始化、seed42、42 维单音素头、phone-uniform target mass；原生卷积 stride `[5,2,2,2,2,2,2]`，16 kHz 对应 20 ms，无上采样。backbone、CNN 是否可训练及是否采用 regfrozen 必须在实际训练配置中固定；不能把当前 10 ms 可训练末层 CNN 的配方直接改名为 20 ms。

训练设置候选为 40 epochs、logical batch64、AdamW lr2e-4、warmup500、weight_decay0.005；三组仅改变 transport cost。数据拟复用 `reproduction/multihead_timit/data/{train-train,train-dev,test}.jsonl`，4620/400/192，保留既有训练 phoneset；时间标注只用于评分。实际提交前记录 manifest、vocab、SSL 和源码 hash。

原始完成的 20 ms OTTC-CE 位于 `reproduction/models/large_models_results/wav2vec2-large-lv60/runexp_ottc_fuse_phoneme/`，CNN frozen、regfrozen2165、batch64/40epochs。它是无 CR 控制候选。必须核对实际音频、目标、参数训练状态及评测协议后才能称为匹配控制；不为补平均重训旧 CE，也不能把它当作已经训练的 CROTTC-CE。

## 开跑与验收顺序

1. 明确三组维度及共同 backbone / loss 归约，使用新 recipe，不改已有完成快照或活跃作业源文件。
2. CPU 小模型验证：两视图顺序、相同帧时钟、padding 不参与 KL、teacher detach、两视图及 phone/mass head 的有限非零梯度、逻辑 batch 缩放、精确恢复。
3. 每组真实 GH200 profile：代表 batch、最长输入、双视图前向/反向/optimizer、加载/计算耗时、峰值显存及吞吐；profile 通过才启动正式训练。
4. 资源沿用最近 TIMIT 约定：每作业 1 GPU、32 CPU、regular-g、2 小时；保留完整 latest 以便续跑。具体 microbatch 由双视图 profile 确定。
5. 每完整 epoch 原子保存独立模型权重和 epoch/step/config/source 身份；保留全部 epochs、best-dev、完整 latest optimizer/scheduler/RNG。
6. 每组独立 W&B run，默认 `reproduction/wandb_config.json`，resume 复用 ID；本地和 W&B 分别记录总 loss、OTTC 分量、原始 CR、加权 CR、step/epoch/LR、PER 和边界指标。网络不可用保留 offline 日志。
7. 完成后评测 best-dev、final、final5/final10 平均，记录平均成员及 hash。先完成所有 dev 候选评分再做 test，只按 dev 选择。
8. dev400 / core-test192 分开报告识别 PER、无文本分割 P/R/F1/R-value、给定音素 K2 FA P/R/F1/R-value/MAE；使用既有冻结 IF-MDD R-value 和明确的一对一边界协议，10/20/30 ms 容差。识别只输入音频。原始 loss 归约不同不直接比较绝对值。
