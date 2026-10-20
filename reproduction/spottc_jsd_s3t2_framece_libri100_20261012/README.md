# SP-OTTC upstream JSD, ordinary S3T2: Libri100 framewise CE stabilization

**2026-10-12 audit:** This completed run is a non-equivalent adaptation of the
historical CE recipe: full Zapformer updates, head architecture, DDP reduction
and recognition decoder differ. It also uses Adam, constant LR, aux0 and no
SpecAugment/cumulative model average, rather than the archived Zapformer recipe.
See [audit_ce/AUDIT.md](audit_ce/AUDIT.md).

Authorized 2026-10-12. Start from completed ordinary S3T2-JSD epoch 10,
step 68868. Independent CE-only fine-tuning stage; original checkpoints stay unchanged.
S3T2** is a separate experiment and is not changed by this job.

Historical reference is the IF-MDD Libri100 10ms frame-CE family: CE weight 1,
CE-only, three epochs. Historical head Adam LR was 4e-4 for the e30 continuation
and 1e-4 for the Libri960 continuation; both froze WavLM. The exact historical Libri100 training
manifest (28538 utterances) and token-to-time function are reused. Timestamp
rounding is replayed against the historical CE loss, including padding/gap masks.
ARPAbet centers are explicitly adapted to the existing MFA IPA representatives.
No new MFA alignment or TIMIT gradient training is performed.

The existing 313-state head is kept. Framewise CE uses the logsumexp of the
three state probabilities per phone, retaining blank in the softmax denominator.
No equal-duration S0/S1/S2 timing labels are fabricated. This stage has CE=1,
JSD=0 and forward-sum=0, matching the old CE-only objective (not the full training recipe). The
Zapformer acoustic backbone and output head are trainable; the unused mass
predictor is frozen. Unlike the historical WavLM recipe, this model has no
separate pretrained SSL frontend to freeze. This adaptation is recorded and is not equivalent to freezing the old WavLM backbone.

- Formal mode B: four nodes, one GPU per node, true DDP, regular-g, 4h/segment.
- Batch 32 per rank, effective batch 128; memory-bounded microbatches preserve global valid-frame CE normalization.
- Save recoverable checkpoints every 50 optimizer steps and at each epoch end; retain all checkpoints.
- Evaluate official TIMIT dev400 every 50 steps and at epoch end; core-test192 before fine-tuning and at every epoch end, for diagnostics only.
- Recognition: audio only, ordinary S3T2 greedy; both native IPA PER and frozen mapped ARPAbet PER with S/D/I counts.
- Text-free segmentation: original phone argmax-change boundaries including blank, 10ms grid, original 10/20/30ms one-to-one boundary scorer.
- Forced alignment: exact ordinary S3T2 training graph, supplied phones including silence, previous-blank convention, duration/frames timing; report separately from text-free results.
- Record segmentation P/R/F1 and frozen IF-MDD R-value in dev/test; source hashes accompany every evaluation.
- Independent W&B run using reproduction/wandb_config.json defaults, raw CE loss/NLL/valid frames, steps, epochs and LR; resume saved run ID, fallback to offline on network failure.
- Atomic checkpoint resumes preserve Adam, scheduler, each rank's consumed batch cursor and RNG. Four-hour segments automatically resume until three epochs complete.

Validation is a separate single-GPU job: historical CE loss/gradient replay,
all Libri100 Fbank preparation, full checkpoint loading and backward pass,
save/load/resume, and both TIMIT evaluation modes. Formal dispatch depends on
successful validation and checks the completion marker and frozen source hashes.

Artifacts: timeline_provenance.json, data/PREPARE_COMPLETE.json, training/run_config.json,
training/updates.jsonl, training/timit_metrics.jsonl, training/evaluations/ and training/wandb_run.json.
