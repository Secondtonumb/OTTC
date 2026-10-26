# TIMIT 真实 posterior：CE / JSD / AG-DIV / J-DIV

真实语音：dev 第一条 `FADG0_SI1279`，文本为 **bricks are an alternative**，长 1.817625 秒。没有根据模型之间的视觉差异筛选语音。用户选择只导出四个完整模型，因此不包含 KL 的早期或未完成模型。

交付目录：`seed43/FADG0_SI1279/`。

## 直接看图

- `posterior_3d_interactive.html`：离线查看器，单一 3D 画布切换四个模型，支持 2D 热图及全 42 类；下方嵌入四模型同视角总览。音频、图片、数据和 JS 均已内嵌，可单独打开 HTML。浏览器没有 WebGL 时自动使用热图。
- `posterior_3d_comparison.png`、`.pdf`、`.svg`：同视角、同概率尺度、同音素颜色的静态图。SVG 的文字可编辑。
- `posterior_full42_heatmap.png`：全句、全部 42 类的热图，附精确 mapped PHN 时间标记。

## 接入 Phonetic Posterior Lab

站点：<https://phonetic-posterior-lab-kg.kevingenghaopeng.chatgpt.site/>。

在页面的 **导入真实 posterior** 文件选择器中选择 `timit_FADG0_SI1279_seed43.posterior-v1.json` 即可。该文件按站点 [接入说明](https://phonetic-posterior-lab-kg.kevingenghaopeng.chatgpt.site/posterior-format.md) 的 v1 格式合并四模型，`source_type=model`，包含完整 42 类和 mapped human PHN 参考区间。只在浏览器本地读取，不需要修改或发布站点。

此接入文件的 `time_ms` 使用 **CNN 感受野中心**，首帧 12.46875 ms、帧距 10 ms；与原静态图和原始导出 `frame_time_seconds` 的历史帧起点约定相差 12.46875 ms。这只是显式时间坐标约定转换，概率矩阵没有改变、重采样或插值。`metadata` 保留时间定义、模型 checkpoint hash、seed/epoch/step，使用可分享标识，不含私有服务器绝对路径。

`SITE_VALIDATION.json` 记录站点原始浏览器校验器的验证结果；`SITE_INTEGRATION.json` 记录格式适配和查看器来源。站点 schema、说明和转换器的下载快照也附在目录中。

原先单容器四 WebGL 场景的本地 HTML 已替换为站点提供的单场景查看方式；四模型总览为独立自适应图片，因此下方总览不依赖额外 WebGL 场景。

## 给画图程序的数据

每个模型均有 `ce` / `jsd_standard` / `ag` / `jdiv` 对应的 `.npz`、`.json`、`.csv`；内容相同。

| 数据字段 | 形状/含义 |
|---|---|
| `posterior` | `[180, 42]`，原始 `softmax(logits, dim=phone)`，FP32 |
| `logits` | `[180, 42]`，未经 softmax 的分类头输出 |
| `labels` | `[42]`，第 j 列对应的音素/特殊 token；完整保留 pad、q、sil、unk |
| `frame_time_seconds` | `[180]`，历史 native 网格 `frame_index * 0.01` |
| `receptive_center_seconds` | `[180]`，CNN 感受野中心：上述网格加 0.01246875 秒 |
| `ot_frame_mass` | `[180]`，单独的质量头在时间轴上的 softmax；不是音素 posterior |
| `argmax_ids` | `[180]`，逐帧最大概率类 |

JSON 中语音 ID 的字段为 `utterance`。CSV 为长表，一行一个 frame/phone，共 7,560 行数据，列名含 `phone_id`、`phone`、`posterior` 等。`reference.json` 原样保留两种 mapped reference，包括未量化的时间戳和参考中的间隙；参考未输入模型。

```python
import numpy as np
d = np.load('ce.npz')
time = d['frame_time_seconds']
phones = d['labels'].tolist()
P = d['posterior']
# 例如 /r/ 的完整逐帧概率
r_probability = P[:, phones.index('r')]
```

绘图要求：横轴用提供的真实帧时间，纵轴用 `labels`，高度用原始 `posterior`。四模型保持相同视角、音素颜色及 0–1 高度范围。只显示部分音素时不要重新归一化；不要平滑、插值成假数据或把时间拉伸到整句音频长度。当前静态图展示 0.10–0.47 秒的 b/r/ih/k/s/sil 六类，完整数据仍保留全句 42 类。

## 模型和来源

全部为 `timit_seed_sweep_20261022/models/*__seed43`，同一 wav2vec2-large-lv60 / monophone / native10ms / batch64 / 40 epoch 配置，独立按 development PER 选择 single best-dev 权重：

| 导出名 | 方法 | best-dev epoch | step |
|---|---|---:|---:|
| ce | CE | 28 | 2044 |
| jsd_standard | 标准 Jensen–Shannon | 38 | 2774 |
| ag | 原始 upstream `jsd` 的 AG-DIV 数值实现 | 32 | 2336 |
| jdiv | KL(P∥Qε)+KL(Qε∥P)，不乘 1/2 | 29 | 2117 |

四个训练均有 `TRAIN_COMPLETE.json` 和 `BENCHMARK_COMPLETE.json`。推理使用 CPU FP32、eval 模式，无 autocast，不传 labels，不算参考条件化的 transport/alignment，不更新权重。概率来自各自训练完成的权重；不是对同一 posterior 临时切换 loss 获得的结果。

四模型导出的 argmax 均与原有 FP32 GPU dev 预测逐帧完全一致；逐帧概率和为 1；NPZ/JSON/CSV 逐值一致。单句图只作实例，不用于散度排名。

`manifest.json` 保存 checkpoint 绝对路径及 SHA256、epoch/step、配置/数据 hash、推理设置、文件 hash。`COMPLETE.json` 是此导出的完成与校验记录。全部文件保存在本地，无 W&B 上传。

复现（在 OTTC 仓库根目录）：

```bash
source reproduction/libri960_continue5_20261017/environment.sh
export OTTC_EVAL_UPLOAD=0 MPLCONFIGDIR=/tmp/ottc-posterior-mpl
"$LIBRI_PYTHON" -m reproduction.timit_posterior_20261026.export
# 只重画已有数据并校验，不再推理
"$LIBRI_PYTHON" -m reproduction.timit_posterior_20261026.export --render-only
# 只重建站点接入 JSON 和离线 HTML，不重复推理或重画静态图
python -m reproduction.timit_posterior_20261026.site_adapter
```
