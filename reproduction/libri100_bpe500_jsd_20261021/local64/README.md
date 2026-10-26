# Single-GPU physical64 continuation

The user corrected local_batch to 64 and prioritized conserving GPU resources.
This continuation uses one GPU per experiment, local/global batch64, without
microbatch splitting or gradient accumulation. It replaces the old two-GPU
local32 runs, keeping the same global batch and original optimizer/LR schedule.
It is not a fresh model or a global128 experiment.

Both imported checkpoints are completed epoch10, step4460. Original weights,
optimizer and scheduler are hardlinked and verified; old artifacts are untouched.
The imported Trainer state changes only train_batch_size to64, preserving
global_step, epoch and max_steps. Rank0 RNG is used, with a loader supporting
both distributed CUDA RNG tensors and single-process RNG lists. Changing world
size/local batch changes length-grouped sample ordering, so this is not a
bitwise-identical replay. Imports prefer completed epoch boundaries and record
any partial-epoch fallback. Local ancestor logs and eligible epoch weights are
retained, but are not uploaded as newly recorded training curves.

```bash
source reproduction/libri960_continue5_20261017/environment.sh
"$LIBRI_PYTHON" reproduction/libri100_bpe500_jsd_20261021/local64/train.py --check
"$LIBRI_PYTHON" reproduction/libri100_bpe500_jsd_20261021/local64/train.py --submit --recipe ottc_jsd
"$LIBRI_PYTHON" reproduction/libri100_bpe500_jsd_20261021/local64/train.py --submit --recipe spottc_jsd
```

Each job requests one GPU for four hours, reserving one hour for saving and
evaluation. The target remains 50 total epochs/22300 optimizer steps, not 50
additional epochs. Deadline pauses preserve complete state; resubmit this same
recipe to continue. There is no automatic fallback to batch32.

GPU utilization, used/total memory and power are sampled every ten seconds in
jobs/gpu_JOB_RECIPE.csv. The first three training steps record actual batch size,
forward/backward time, peak allocated and reserved GPU memory in
models/RECIPE/batch64_profile.jsonl. These timings exclude dataloader waiting and
optimizer updates; use whole-job timing for total GPU-hour comparisons.

Original JSD, greedy/constrained evaluation and separate TIMIT cross-domain
segmentation are unchanged. Epoch checkpoints follow best3+last10 retention;
best-dev and latest2 full states are kept. Evaluation artifacts stay local.