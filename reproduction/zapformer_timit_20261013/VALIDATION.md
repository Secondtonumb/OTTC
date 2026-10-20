# Validation: 2026-10-13

Final GPU job3544772 completed successfully after keeping the OT mass branch in
FP32 under BF16 encoder autocast. Both upstream-JSD and CE tiny models matched
continuous4 updates versus2+resume-to4 exactly: model, cumulative average,
optimizer, scheduler and all rank/RNG states had zero difference.

The full native Zapformer completed2 real TIMIT optimizer updates, checkpoint
save/load, dev selection and final recognition + K2 FA on2 dev/2 test utterances.
These scores are workflow diagnostics after2 updates, not trained-model results.
No40-epoch production run was launched. Default all4620-utterance batch plans
were separately audited for all40 epochs:163–165 batches/epoch, no omissions or
duplicates. Complete train/dev/test feature manifests and SHA256 are in data/.

Evidence: validation_3544772/RESUME_PASS.json and
validation_3544772/full_model/TRAIN_COMPLETE.json. Training code/data/vendor
identities are recorded in that run's run_config.json. Short validation W&B
runs are stored offline alongside local logs.
