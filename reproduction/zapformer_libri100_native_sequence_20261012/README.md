# Authorized Libri100 sequence

**Cancelled by the latest user instruction.** Ordinary job 3536313.opbs and
controller 3536361.opbs were cancelled while queued/held. Direct S3T2** job
3536393.opbs replaces this sequence. The old controller is preserved as
continue.pbs.cancelled_sequence; continue.pbs now exits without submission.
The following is historical context, not the active plan.

First ordinary upstream-JSD S3T2, then independently initialized S3T2**, both
seed42 and 10 epochs, as explicitly confirmed by the user. Each experiment
has separate output/checkpoints/W&B identity. Both source-bound validation
suites and epoch completion checks have passed.

Initial ordinary segment: 3536313.opbs. Continuation controller: 3536361.opbs,
depending on initial success. Each segment keeps the selected one-node,
640-second/rank configuration unchanged. A normal wall-time checkpoint exit
queues one successor with an afterok dependency; any validation/training failure
stops the chain. Flocks prevent concurrent controllers or experiment writers.
S3T2** starts only after ordinary TRAIN_COMPLETE, with a remaining-time budget
that reserves ten minutes for cleanup. Source changes reject resume.

After both stages finish, LIBRI100_STAGES_COMPLETE.json records the final
Star-Star checkpoint containing current and averaged weights. This is not a
Libri960/MFA completion marker. Subsequent stages are not submitted yet.
