# Miyabi GPU 交互节点测试入口

已有GPU交互分配时，直接在那个节点运行第二段命令。需要新分配时，先在登录节点申请：

```bash
qsub -I -q interact-g -W group_list=gm64 \
  -l select=1:ncpus=16:mpiprocs=1 -l walltime=00:30:00
```

看到计算节点提示符后运行：

```bash
cd /home/m64000/work/OTTC
bash reproduction/ottc_temporal_smoothing_20261021/interactive_gpu.sh \
  --weight 0.1 --tau 4 --steps 1 --batch-size 2
```

默认是native10ms、mono、原始upstream-JSD。0.1只作为测试系数，尚未调优。脚本要求PBS分配及CUDA可用，不能在登录节点运行GPU测试；它不自动申请、结束或抢占任何作业。完成后仍保留你的交互shell，是否`exit`释放节点由你决定。

首先在真实CUDA上核对CE、upstream-JSD、standard-JSD和Euclidean四种底层cost：默认关闭与修改前函数的loss及运输信息一致；加入正则后的值符合OTTC+λ×平滑项；梯度有限且相符，α没有直接增加平滑梯度。CUDA重复索引反向可能有原子加法次序差异，梯度使用rtol=1e-5/atol=1e-6比较。这里不会把CUDA分配改成CPU。

随后加载原始SSL编码器及新随机任务头，采用现有TIMIT原生10ms/可训练末层CNN配置，在训练清单的前2条真实音频上执行前向、反向和1次AdamW更新。直接调用本轮修改的`ot_loss.py`，单独记录OTTC、平滑项、加权平滑项及总loss，检查四组梯度、CNN冻结策略、分类头和末层CNN更新。没有使用音素边界时间；只需要普通OTTC训练音素序列。它验证loss和训练数据链路，不代表已训练模型的识别/分段质量，也不是完整训练或吞吐profile。

只看底层loss，不加载大编码器或真实音频：

```bash
bash reproduction/ottc_temporal_smoothing_20261021/interactive_gpu.sh --core-only
```

更多测试方式：

```bash
# 只前向/反向，不更新参数
bash reproduction/ottc_temporal_smoothing_20261021/interactive_gpu.sh --steps 0

# 在同一小batch上多跑几步，观察loss；最多允许20步
bash reproduction/ottc_temporal_smoothing_20261021/interactive_gpu.sh --steps 5

# S3状态头；同样在状态logits上平滑
bash reproduction/ottc_temporal_smoothing_20261021/interactive_gpu.sh --head s3t2ss

# 换cost或关闭正则作对照
bash reproduction/ottc_temporal_smoothing_20261021/interactive_gpu.sh --cost ce
bash reproduction/ottc_temporal_smoothing_20261021/interactive_gpu.sh --weight 0
```

用`--help`查看全部参数。两次调用均从seed42和相同SSL/新头开始，因此可以比较系数设置；不要据同一个极小batch的loss趋势选择最终超参数。

每次输出到新的`gpu_probes/<UTC时间_PID>/`目录，也可以`--output-dir 新目录`指定位置。保留`CONFIG.json`、`CORE_CHECKS.json`、`real_batch_steps.jsonl`、`REAL_BATCH.json`和`probe.log`；只有全部成功才写`PROBE_COMPLETE.json`。失败时写`FAILED.json`并保留日志。记录真实GPU、PBS job、精度、源码/清单hash、每步显存和耗时。测试只在内存里更新临时模型，不写模型checkpoint，不覆盖旧模型、resume或正式训练配置，不上传W&B。
