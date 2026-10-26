# Buckeye full40 真实 posterior

与 TIMIT posterior 导出使用相同的四模型和文件口径：CE、标准 JSD、AG-DIV（原 upstream `jsd`）、J-DIV。默认 seed43，使用本批训练在导出开始时各自的 single best-dev 权重，**这是训练中的快照，尚非最终完成模型**。具体 epoch、step、hash 和完成标记状态保存在每次导出的 `manifest.json` 中。

`LATEST_EXPORT.json` 指向最近一次完整交付目录。失败尝试不写此指针，也不写导出完成标记。每次输出使用独立 UTC 时间目录，不覆盖已有结果；不改训练、权重或续训文件。

固定使用 full40 dev 第一条 `buckeye_s0701a_0000`：**okay one two**，1.518875 秒，speaker s07。不是根据模型差异挑选的例子。数据通过 `load_buckeye_rows` 和受校验的默认 registry 读取，核验训练与 dev speaker 不重叠。保留完整 42-ID 历史词表，包括特殊类。gold 的 mapped/native 区间原样保存，不补间隙、不合并、不量化。

每个交付目录包含：

- `posterior_3d_interactive.html`：离线交互图，嵌入音频、完整数据和四模型总览；可切换模型、旋转和显示全42类。
- `posterior_3d_comparison.png/pdf/svg`：第一词的四模型同视角、同颜色、同 0–1 概率尺度对照。
- `posterior_full42_heatmap.png`：全句全部42类，并标出精确 gold 区间。
- `buckeye_*.posterior-v1.json`：与之前 Posterior Lab 相同的导入格式。只是本地生成，没有发布或上传。
- `ce/jsd_standard/ag/jdiv.npz/json/csv`：完整 logits、FP32 posterior、argmax、标签、两种帧时间和独立的 OT mass。
- `statistics.json`：本句全帧的平均最大概率、熵（nats）、top1-top2 间隔、argmax 切换次数，以及逐类均值/峰值/argmax帧数。
- `reference.json`、`manifest.json`、`*.BEST_DEV.snapshot.json`：参考、模型/数据来源和选点记录。
- `COMPLETE.json`：**仅指本次导出完成**；训练和 benchmark 完成状态另行记录。

推理为 CPU FP32、eval、无 autocast，仅输入音频，不传参考标签，不计算参考条件化的 transport/alignment，不更新模型。posterior 是各自模型原始 logits 在全部42类上的 softmax。OT mass 是另一头在时间轴的 softmax，不是 phone posterior。不做平滑、温度修改、重采样或子集重新归一化。

原始图/数组的 `frame_time_seconds` 沿用 `frame_index * .01`；站点格式 `time_ms` 使用真实 CNN 感受野中心，两者相差 12.46875 ms。gold 时间不随此转换移动。不要把 posterior 时间轴拉伸到整段音频长度。

统计和图只描述这一句，不能作为 corpus 统计、边界评估或损失函数优劣排名。训练未完成，best-dev epoch 也不同。导出开始即用只读文件描述符固定四个权重，逐个核验 BEST_DEV SHA256；避免训练保留策略更新文件时混入不同 checkpoint。

复现（仓库根目录）：

```bash
source reproduction/libri960_continue5_20261017/environment.sh
export OTTC_EVAL_UPLOAD=0 WANDB_MODE=disabled
"$LIBRI_PYTHON" -m reproduction.buckeye_posterior_20261026.export
# 只从已有数组重画、重算统计与校验，无须重新推理：
"$LIBRI_PYTHON" -m reproduction.buckeye_posterior_20261026.export --render-only <交付目录>
```

全部结果保存在本地，无 W&B/CoreWeave 上传。`NPZ/JSON/CSV` 的 posterior 逐值核验一致，逐帧和为1，站点格式也保持原始概率值。
