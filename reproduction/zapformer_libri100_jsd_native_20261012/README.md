# Libri100 ordinary S3T2 upstream JSD, native Zapformer

**Skipped at the user's latest explicit request.** Job 3536313.opbs was
cancelled while queued; no formal optimizer update ran. Direct S3T2** job
3536393.opbs is the active experiment. Below is the superseded preparation.

User-confirmed plan: independent seed42 initialization, 10 epochs, followed by
an independently initialized 10-epoch S3T2** experiment. The final S3T2**
checkpoint will initialize the subsequent Libri960 stage. No historical CE
or other architecture checkpoint is used as this experiment's initialization.

All ordinary S3T2 migration checks passed before submission: multi-utterance
single GPU/full model, four-node DDP resume, and two-epoch completion plus
final cumulative-average scoring. Source and environment are frozen in
`../zapformer_native_acceptance_20261012/accepted_source.tar.gz` and its JSON.

Training uses the verified native IPA Libri100 manifest (28,538 utterances),
original archived OTTC JSD plus original S3T2 forward-sum, native full-size
backbone/augmentation/BatchedRubik/scheduler/BF16/auxiliary gradients, and the
explicitly approved deterministic k2 dependency adaptation. One node and
640 seconds/rank were chosen after full-model measurements: 298 useful audio
seconds per node-second, versus 283/280 on two/four nodes at the same budget.
Single-GPU budgets 960/1280 were faster but reserved about 90/94 GiB of the
actual 95.6 GiB visible GPU memory; 640 reserved about 51 GiB and leaves room
for batch-length variation. Larger batch size affects the number of updates
and is explicitly recorded rather than treated as an identical old experiment.

W&B defaults online from repository project/entity settings, with offline
fallback, local raw losses and recognition/segmentation/R-value. Each 4-hour
segment checkpoints after 13,800 seconds; rerun the same PBS script to resume
the saved run ID and all training state. Source/data/world-size/config changes
reject resume. A failed process does not automatically repeat. Completion
requires TRAIN_COMPLETE.json and final current/average benchmarks.
