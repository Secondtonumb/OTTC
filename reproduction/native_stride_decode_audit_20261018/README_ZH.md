# K2 topology 解码核查与独立FP32复评

用户要求核对 S3T2** 是否按原始 topology 解码，并重新查看指标。本目录只读取既有模型，不重新训练。最终表见 `REPORT_ZH.md`、`RESULTS.json`；同精度单音素对照也已复评。

- `audit_saved_paths.py`：从冻结 IF-MDD 源码提取原始函数，与当前 IF-MDD 函数 AST 比对；验证旧 7104 条无文本 K2 路径合法。
- `gpu_probe.py` / `GPU_PROBE.json`：复现 K2 图在 inference_mode 内构建失败，以及冻结 encoder 在 inference_mode 内 deepcopy 后无法普通加载权重的问题。
- `redecode_fp32.py` / `run_evaluate_fp32.pbs`：当前复评入口。新建普通 FP32 模型，保持原权重、数据和拓扑；K2 图在 inference_mode(False) 中构建。新结果写入本目录，原源文件、训练权重、平均权重和历史结果不改动。
- `redecode.py` / `run_evaluate.pbs`：首版诊断快照。模型解码与FA计算完成后，曾因要求新旧数值完全一致的检查而报错。独立FP32与原训练末AMP包装评测不同，`finalize_metrics.py`按实际覆盖、dev-only选择和FA成功率验证完成结果，并保留数值差异；后续作业使用上面的FP32入口。
- `audit_corrected_paths.py`：重新解码的7104条无文本路径和7104条给定音素FA路径均接受于原图，全部覆盖、零FA失败。
- `publish.py` / `TRAINER_AMP_PROBE.json`：实测Trainer/Accelerate可以内部BF16计算、外层返回float32；旧训练末评测“FP32”标签不准确。本次fresh standalone模型无该包装。
- `summarize.py`：保留原duration/frame_count边界分数，并从保存的FSA状态路径还原准确帧位置，按统一固定10ms时间轴比较。沿用冻结的一对一匹配和IF-MDD R-value；PER不依赖这项时钟换算。
- `publish_comparison.py`：将最终同时间轴比较上传至每个原W&B run，`COMPARISON_PUBLICATION.json`记录状态。

权重候选固定为训练期greedy best-dev、final、final5平均、final10平均；S3T2**最终候选仅按dev K2 PER选择，test不参与。没有声称遍历所有epoch寻找K2最优。

复现本地路径审查和汇总：

```bash
python reproduction/native_stride_decode_audit_20261018/audit_saved_paths.py
python reproduction/native_stride_decode_audit_20261018/audit_corrected_paths.py
python reproduction/native_stride_decode_audit_20261018/summarize.py
```

`evaluation_dispatch.json`也记录九个已训练完成WavLM模型的评测恢复作业，全部只复评。旧模型根目录的完成标记不代表本目录纠正后的FA已经完成，应读取本目录的`CORRECTED_EVALUATION_COMPLETE.json`。

Libri100仍未提交，未创建通过审查的LIBRI_GATE.json；8卡安排仅适用于将来放行后的Libri100。
