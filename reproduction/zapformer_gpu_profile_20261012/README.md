# Zapformer native GPU throughput measurements

Run only after source-bound single-GPU, four-node DDP and epoch-completion
acceptance. The full Libri100 IPA manifest and full-size ordinary S3T2 model
are used with the accepted deterministic k2 adaptation and original OTTC JSD.
Each duration/configuration is a separate native training invocation and an
independent offline W&B run. No prior checkpoint is rewritten or reused.

`native_gpu_probe.py` instruments cached input loading through the real native optimizer
post-hook, synchronizes CUDA for timing, and samples nvidia-smi every 500 ms.
Sixteen optimizer updates include four warmup updates. Evaluation and checkpoint
overhead are outside these measured update intervals. Disposable profile runs
skip multi-GB checkpoint writing only; they retain raw loss/local/W&B records
and source/configuration identity. These are performance probes, not convergence
results. CPU model averaging allocation remains the native full-size one.

`profile_single.pbs` compares duration budgets 100/200/320 seconds on one GH200.
Failed/OOM configurations retain logs and partial profile JSON, and do not count
as successful profiles. `summarize.py` ranks successful measurements by useful
audio seconds per node-second, recording allocated/reserved GPU memory and
padding. Multi-node aggregation uses each update's slowest rank elapsed time,
not an average of rank rates. Batch budgets change global batch size and must
be recorded as separate formal training configurations if selected.
