# TIMIT Frame-wise CE / JSD / AG / J-DIV posterior

与之前OTTC posterior使用同一语音、同一seed：TIMIT dev首句`FADG0_SI1279`，
文本`bricks are an alternative`，1.817625秒，seed43。
本次四组均来自新训练的**人工PHN逐帧监督模型**。

输出目录：`seed43/FADG0_SI1279/`。

## 查看与导入

- `timit_framewise_FADG0_SI1279_seed43.posterior-v1.json`：合并四模型，兼容已有Posterior Lab v1导入格式。
- `posterior_3d_interactive.html`：离线交互查看器，内嵌四模型数据、音频、Plotly和四模型总览；可切换模型、热图/3D和全部42类。无WebGL时自动切换热图。
- `posterior_3d_comparison.png/.pdf/.svg`：四模型同视角、同音素颜色、统一0–1概率尺度，展示`bricks`片段。
- `posterior_full42_heatmap.png`：全句、完整42类热图，保留人工参考时间。

## 原始数据

每个模型均提供`ce` / `jsd_standard` / `ag` / `jdiv`的NPZ、JSON和CSV。
`posterior`为真实FP32`softmax(logits,dim=phone)`，形状`[180,42]`。
完整保留pad、q、sil、unk；没有平滑、插值、温度调整或对子集重新归一化。
另含原始logits、argmax_ids和完整词表；NPZ/JSON/CSV逐值核对一致。

```python
import numpy as np
x = np.load('ag.npz')
P = x['posterior']                 # [180,42]
phones = x['labels'].tolist()
t = x['receptive_center_seconds']  # 全部新图和导入JSON采用此时间轴
r = P[:, phones.index('r')]
```

时间轴为实际CNN感受野中心：`(199.5+160*i)/16000`秒，首帧12.46875ms、步长10ms。
全部新图与合并JSON使用同一中心时间轴。为兼容旧导出，原始文件另保留
`frame_time_seconds=i*0.01`历史帧起点约定，两者不能混用。
人工PHN参考时间不量化、不填补间隙、不合并区间。参考仅作为显示叠层，未输入推理。

Frame-wise模型的mass/duration head冻结且未使用，因此本次不导出`ot_frame_mass`字段。

## Checkpoint与核验

全部采用各模型独立按dev PER选择的**single best-dev**，不混入checkpoint平均：

| 模型 | best-dev epoch | step |
|---|---:|---:|
| Frame-wise CE | 16 | 1168 |
| Frame-wise标准JSD | 39 | 2847 |
| Frame-wise AG | 18 | 1314 |
| Frame-wise J-DIV | 16 | 1168 |

CPU FP32、eval、无autocast；模型只收到音频及attention mask，没有labels、frame targets或参考时间。
四组argmax路径均与已保存的FP32 GPU dev预测逐帧完全一致。
`manifest.json`保存checkpoint路径/哈希、配置哈希、epoch/step、实际模型类、数据来源及推理设置。
可导入的合并JSON使用可分享模型标识，不含服务器绝对路径。

数据、音频和报告均保存在本地，没有上传或修改公共站点。
该语音沿用固定示例，未根据视觉差异筛选；单句图不替代整体评估。

复现：

```bash
source reproduction/libri960_continue5_20261017/environment.sh
export OTTC_EVAL_UPLOAD=0 MPLCONFIGDIR=/tmp/timit-frame-posterior-mpl
"$LIBRI_PYTHON" -B -m reproduction.timit_frame_posterior_20261026.export
python3 -B reproduction/timit_frame_posterior_20261026/validate_assets.py
```
