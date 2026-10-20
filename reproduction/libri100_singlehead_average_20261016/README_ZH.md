# Libri100 三种 OTTC 代价：checkpoint 平均与 PFER

原实验为 `../libri100_singlehead_20261014`，保留所有原模型、结果和源文件。单头42类、20ms、wav2vec2-large-lv60，训练40epoch，CE / 官方upstream JSD / standard JSD。

只比较保留的17394、17400、17600、17800、17840权重。最后3个平均为17600、17800、17840；另用Libri dev-clean+dev-other池化K2 FA F1@20ms或PER选3个。平均FP64累加、保存原dtype，包含当前及frozen encoder参数；不平均优化器，不训练。相同集合复用推理。dev选择在对应平均模型的test评估前冻结，test/Buckeye/PFER不参与选择；last3已预先指定，可独立评测。

`prepare.py` 已冻结 `plan.json`，源scorer和推理循环AST一致；`worker.py` 首先重放Last的8句dev-clean并要求完整prediction记录与原结果相同，再执行候选dev评估、选权重、平均、六个split推理和PFER。所有完成标记只在对应检查成功后创建，可重复运行恢复已完成任务。

PFER复用已验证的PanPhon0.22.2特征编辑距离和音素映射表。输入为text-free hypothesis，展开双元音，去sil/q、dx→d。零参考句保留插入，未知映射直接失败；PER与原结果验证一致。逐句距离、S/I/D、feature-close替换诊断、预测/参考IPA段数比和配对CI保存。此序列PFER不是帧级feature error，也不是supplied-phone K2 FA的标签PER。

调度使用已读取的 `/home/m64000/.codex/skills/miyabi-dispatch/SKILL.md` 单节点单任务模式A：每个单元独立单GPU，没有DDP。用户要求改为20分钟：`run_one.pbs`，NODES=1、QUEUE=short-g、WALLTIME=00:20:00，ncpus=32，mpiprocs=1；变量为`LIBRI_OBJECTIVE`（ce/jsd_upstream/jsd_standard）及`LIBRI_AVG_TASK`（默认last3）。推理batch=1、FP32，沿用原评测，无learning rate或训练batch变化。W&B在线默认，失败保留offline。

`short_worker.py` 拆为每目标5个单元：last3（六个split，历史推理约12分钟），dev_a（17394/17400的两个dev split），dev_b（17600/17800的两个dev split），dev_fa_best3、dev_per_best3（各一个平均模型六个split）。后两项afterok依赖两个dev任务，不依赖last3；相同选中权重集合通过文件锁复用完整推理，STATUS合并也加锁。完成标记和数值评分口径沿用原流程。

提交示例：

```bash
qsub -v LIBRI_OBJECTIVE=jsd_upstream,LIBRI_AVG_TASK=last3 /work/gm64/m64000/OTTC/reproduction/libri100_singlehead_average_20261016/run_one.pbs
```

CPU补评分：

```bash
source /work/gm64/m64000/OTTC/reproduction/libri100_singlehead_20261014/environment.sh
"$LIBRI_PYTHON" /work/gm64/m64000/OTTC/reproduction/libri100_singlehead_average_20261016/pfer.py --baselines --publish
```

结果见 `REPORT_ZH.md` / `COMPARE_STATUS.json`，全部三组平均完成后才写`COMPARE_COMPLETE.json`。PBS job ID见`dispatch.json`，日志`worker_{objective}_{task}.log`。原2小时提交记录保留在`dispatch_2h_original.json`。

用户随后要求直接使用interactive：所有short-g队列任务已撤销，转入`interact-g`专用队列。每人仅允许一个interactive allocation，队列提供1节点或8节点类型；本次成功请求为1节点20分钟、8节点10分钟，8节点20分钟被scheduler资源限制拒绝。初始节点mg0007 / job3565241，同一GH200并行计算三种last3（各4个CPU线程，原FP32/K2协议）。第二次job3565383为8节点10分钟，8个独立任务均已通过MPI实际逐节点启动。`interactive_driver.py`之后对多个可运行单元使用8节点、每GPU一个独立任务，通过MPI逐节点启动而不进行DDP；只有一个剩余任务时使用1节点。dev-best任务只在该目标两个dev任务均完成时启动，同一权重集合继续加锁复用推理。`interactive_driver_state.json`记录每次allocation与完成情况。

`resume_evaluation.py`会校验并复用已完成整个split的预测文件，只在新allocation上计算缺失split。重放评分与原CE的dev-clean、dev-other（含空参考句）、Buckeye-test结果完全一致；原scorer和推理代码保持不变。整体完成状态以`COMPARE_COMPLETE.json`为准。

interactive续申请会先等待上一job从qselect消失；若PBS仍返回njobs_int-g占用，间隔10秒重试。driver重新启动保留已有allocation历史，并用文件锁避免重复启动；正常walltime续算不计作任务失败。CPU可提前冻结已完成候选的dev选择并平均权重，GPU只计算缺失推理。报告同时列出dev选中的steps、候选dev成绩，以及各dev-best平均相对Last和last3的配对PFER差异。

10月16日14:15，续算的8节点申请3566367预计等待约24分钟，已撤销并保留cancellation记录。改为1节点20分钟申请3566402；设置LIBRI_INTERACTIVE_NODES=1并在driver状态中保留偏好。单节点pool最多同时运行3个独立进程（与首轮已验证的3模型共享GH200相同），优先补完官方JSD/CE的Buckeye，再启动standard JSD；相同CE平均仍加锁复用。模型、FP32/batch1推理和评分源代码均未改变。LIVE_DEV_BEST_TEST_CLEAN_FA.json仅为已完成test-clean的PER/K2 FA预览，不代表完整PFER/六split评测结束。
