# Independent S3T2** native Zapformer adapter

This separate source tree preserves the accepted ordinary S3T2 migration and
its snapshots unchanged. All 135 frozen vendor source hashes are identical.
The original archived OTTC JSD is identical. Only topology selection is added
to the phone adapter, directly reusing the existing
`ifmdd_losses.py::topology_forward_sum_loss(..., topology='s3_t2_star_star')`;
no loss or graph implementation is rewritten. The 313-output architecture,
native loop/optimizer/scheduler/augmentation and approved k2 adapter are reused.

`topology_metrics.py` retains the frozen raw-PHN scorer and R-value formula,
selecting the actual model topology for native greedy completion and supplied-
phone forced alignment, and recording that topology and adapter hash. Original
historical metrics/source files remain unchanged. Text-free and forced metrics
stay separate; MFA directory identity is not asserted.

Accepted JSD tests: single-GPU multi-utterance resume and full-size update
3536295.opbs (60 s), four-node true-DDP resume 3536321.opbs (52 s), and two-epoch
completion/boundary restart/final average scoring 3536322.opbs (93 s). Model,
average and optimizer differences are zero; the two-epoch 58-update saved
states and raw losses match bitwise. Evidence: validation/STAR_JSD_PASS.json,
validation_ddp/STAR_DDP_PASS.json and
../zapformer_s3t2ss_acceptance_20261012/results/EPOCH_PASS.json. The four-node
resume fixture uses the tiny model; full-size ordinary topology DDP throughput
is documented separately and is not relabeled as full-size Star-Star evidence.

Latest user instruction skips the ordinary S3T2 formal stage entirely. Direct
Star-Star job 3536393.opbs starts independently from seed42 for 10 epochs, with
normal checkpointed wall-time resumes; the old ordinary/sequence jobs were
cancelled before execution. The final Star-Star checkpoint is designated for
future Libri960 initialization. Full-model Star-Star 640-second GPU probe
3536403.opbs also passed 16 actual updates in 47 seconds. Formal source/config
checks and W&B run ID resumes remain enforced; the formal 10 epochs are not
completed yet.
