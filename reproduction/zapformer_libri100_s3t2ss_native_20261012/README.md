# Direct Libri100 S3T2** native Zapformer experiment

**Latest resource change:** user requested eight cards to reduce training wall
time. The still-queued single-GPU job 3536393.opbs was cancelled before training.
Eight-node/eight-card true DDP validation 3536584.opbs passed restart and every
rank's saved state with zero model/average/optimizer differences and exact raw
losses. Its first full-model profile failed because four evaluation utterances
left four ranks empty; only the profiler was corrected to provide at least one
utterance per rank. Corrected full-model job 3536653.opbs passed all 16 updates
per rank at 640 seconds/rank in 60 seconds. Evidence: EIGHT_GPU_PASS.json. Total
measured training-update throughput is 1,903.48 audio seconds/second, about 6.7
times the 283.49 single-card profile. Peak allocated/reserved memory per card is
47.66/60.53 GiB. Timing excludes evaluation, persistence and startup overhead;
these short tests are not completed 10-epoch training results.
Formal job 3536658.opbs depends on corrected verification success and the
source-bound EIGHT_GPU_PASS.json; the failed old dependency job was terminated.
Eight ranks train one model with an independent seed42
initialization for 10 epochs; the effective duration budget is 5,120 seconds.
Native LR .02 and objective/update policy are unchanged. World size and budget
are recorded in the run/checkpoint/W&B configuration; cross-world-size bitwise
identity is not asserted. No single-GPU formal checkpoint/run was created.

Active entry: `run_8gpu.pbs` (also mirrored by `run.pbs`), regular-g, 8 nodes
with one GPU/rank each, four hours per segment. `run_ddp.sh` checks all acceptance
evidence and holds one output lock; `ddp_worker.sh` launches torchrun with eight
node ranks. Normal checkpointed exits resume with the same eight-card script
and W&B ID. Failed validation/training stops automatic continuation. Superseded
single-card scripts are preserved with cancelled_single_gpu suffixes. The
single/four-card measurements below are retained as earlier evidence.

Latest user instruction skips the ordinary S3T2 formal stage. Its queued
3536313.opbs and old sequence controller 3536361.opbs were explicitly cancelled
before execution. Their validation/history remains preserved.

Run original upstream OTTC JSD plus the existing S3T2** forward-sum topology,
directly from independent seed42 random initialization for 10 epochs. Native
Zapformer loop, optimizer, scheduler, BF16, augmentation and auxiliary gradients
are frozen. The k2 deterministic dependency adapter is explicitly approved.
The verified native IPA Libri100 manifest has 28,538 utterances. Native vendor,
original JSD and raw-PHN/R-value scorer hashes are recorded; no loss is rewritten.

S3T2** acceptance passed: single/full model 3536295, four-node resume 3536321,
and two-epoch complete/boundary resume/final average scoring 3536322. Source-bound
markers reject changed source. This does not mean the formal 10 epochs completed.

Formal job 3536393.opbs uses one node and max_duration640 for ample memory
headroom. Dedicated full-model Star-Star probe 3536403.opbs passed 16 updates:
283.49 useful audio seconds/second, 45.95 GiB allocated / 52.27 GiB reserved.
Full-model four-node Star-Star DDP probe 3536425.opbs passed 16 updates on every
rank: total 1,045.80 audio seconds/second, 261.45 per node-second, at most 47.66
GiB allocated / 60.53 GiB reserved. Single-node has about 8% better throughput
per allocated node, so it remains selected for conserving node-hours. Source-
bound checkpoint-resume checks separately passed on all four ranks with the
tiny fixture. Neither these probes nor the formal submission mean convergence
training has finished. Evidence: ../zapformer_s3t2ss_gpu_profile_20261012/profile_summary.json.
W&B defaults online with local/offline fallback and saved run ID resumes.

`run.pbs` requeues one afterok successor only after a normal wall-time checkpoint
exit; failures stop via set -e. Completion writes INIT_CHECKPOINT_READY.json,
pointing to the final checkpoint containing current and cumulative-average
weights. The designated next dataset is Libri960; that stage is not submitted.
