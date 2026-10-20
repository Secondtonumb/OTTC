# Reproducibility and validation ledger

This project adapts the frozen **Zapformer** implementation to the ordinary
10ms S3T2 phone task. The user confirmed Zapformer on 2026-10-12; this is not
a migration to Icefall Zipformer/Zipformer2. The native ASR dataset and
SentencePiece/RNN-T/attention objectives are outside this phone-task scope.

## Source and objective identity

- `vendor_manifest.json` records SHA256 for all 135 native Python files. Every
  training process checks them before training. No native file was changed.
- The actual loop is `vendor/zapformer/train.py::train_one_epoch`; BatchedRubik,
  InterpCosineLRScheduler, AlternatingSpecAugment, native auxiliary gradients,
  BF16/GradScaler and float64 cumulative averaging come from the same source.
- `upstream_jsd_provenance.json` identifies repository commit
  `0d97e7e03b6e139fb0d72eb508a18625685074e9`, the original OTTC loss hash and
  the previously reproduced archived loss hash. The upstream divergence AST
  is identical. The bucketized loss AST is identical after removing its
  unreachable `jsd == 'standard'` branch. This task always calls `jsd=True`.
- Inventory adaptation: exclude control columns 1..6; retain original phone
  IDs. S3T2 forward-sum is a separate loss component with weight 1.

## Checkpoint contract

Each run records the data manifest, initialization, source hashes, objective,
world size, epoch budget, duration budget, LR, precision, scoring protocol and
sampling counts. A changed contract or world size rejects resume. Checkpoints
contain current and averaged models, mixed-dtype optimizer buffers, scheduler,
progress, and per-rank scaler, batch-plan hash/cursor, Python/NumPy/PyTorch RNG
and native augmentation's private CPU/CUDA generators. Saved W&B run IDs are
reused; validation runs remain offline and separate from formal experiments.

## Debug findings (2026-10-12)

1. Restored GradScaler is lazy. Native auxiliary loss reads `_scale` before
   the first `scale(loss)`, so the adapter materializes the restored CUDA scale
   before the first forward. The native scale policy stays unchanged.
2. Independent GPU runs can already diverge at their first update. k2 dense
   intersection backward calls `_k2.index_add` to accumulate arc gradients.
   Its GPU atomic additions are outside PyTorch's deterministic-algorithms
   checks; changing their order produces small gradient differences which
   BatchedRubik's BF16 matrix update can amplify.
3. Diagnostic jobs `3535224.opbs` and `3535229.opbs` tested default/strict
   matmul precision with/without `CUDA_LAUNCH_BLOCKING=1`. Neither setting
   reliably eliminated the differences. Each job took 29 seconds.
4. Job `3535240.opbs` kept k2 graphs, forward scores and posteriors, and used
   PyTorch's deterministic index accumulation in a **diagnostic adapter**.
   Both precision variants produced bitwise-identical initialization,
   gradients and updated weights across separate processes. Records are in
   `diagnostics_k2/repeatability.json`. The user explicitly approved adoption
   on 2026-10-12; these diagnostic results are not an integration PASS.
5. CE freezes the unused mass head. Native ParamPlotter assumes all parameters
   have gradients and crashes on this frozen head. The task adapter supplies
   a trainable-parameter view to the unchanged native ParamPlotter and excludes
   frozen parameters from native optimizer groups. Phone token encoding also
   implements the API used by native failure diagnostics, preserving the
   original exception instead of masking it with a missing SentencePiece API.
   The failed CE probe (`3535256.opbs`) is retained as
   `validation_ce_probe.attempt1/` and `validation_ce_probe.log.attempt1`.
6. CE retry `3535265.opbs` passed in 72 seconds on one GH200. Model,
   cumulative average and optimizer differences were all exactly zero after
   process restart, and scheduler/sampler/scaler/all saved RNG states matched.
   The historical full-size checkpoint completed a native CE optimizer update
   and TIMIT loss/recognition/segmentation hooks. Evidence:
   `validation_ce_probe/PARTIAL_PASS.json`. This partial marker cannot unlock
   a formal run. Two-node CE DDP probe `3535288.opbs` also passed, in 57 seconds:
   model, cumulative average and optimizer differences were exactly zero, and
   both ranks' sampler/scaler/RNG states matched. Evidence:
   `validation_ce_ddp_probe/DDP_PARTIAL_PASS.json`.
7. A multi-utterance loss/gradient diagnostic (`3535313.opbs`) exposed a
   PyTorch 2.6 deterministic CUDA `index_put` broadcast assertion in native
   DenseFsaVec's terminal-row assignment. A forward-only low-level dispatch
   attempt still failed in backward (`3535339.opbs`). The retained successful
   intermediate probes are `3535344.opbs` and `3535353.opbs`.
8. The candidate dependency adapter now expands native terminal-row values
   **before autograd records the assignment**, scoped only to the native
   DenseFsaVec constructor. Native construction and backward remain reused;
   there is no global interception around the network or optimizer. The
   final module diagnostic (`3535367.opbs`, 5 seconds, GH200) passed 12 repeats
   of each gradient path: the native scatter produced 7 distinct gradient
   hashes; deterministic scatter produced 1. All forward losses were exactly
   equal; maximum gradient difference was `1.862645149230957e-9`. One- and
   two-dimensional float32/float64/int32 scatter cases with duplicate and -1
   indices and nonzero initial outputs matched native semantics exactly.
   Evidence: `diagnostics_k2/gradient_equivalence.json`. After explicit user
   approval, this module is now enabled by default in train_s3t2.py for both
   validation and formal training. Graphs, forward scores/posteriors, original
   OTTC JSD and native model/optimizer remain unchanged. The contract records
   PyTorch/CUDA/k2 versions, k2 source/binary hashes and determinism settings.
9. Libri100 candidate preparation is complete in
   `../zapformer_libri100_data_20261012/training.json`: 28,538 unique matched
   utterances, 100.586496 hours, 251 speakers; all 11 existing Fbank shards'
   complete SHA256 and bounds were verified. Targets retain native MFA
   word-transcript IPA phones, distinct from historical mapped ARPAbet CE
   timelines. No features were copied or recomputed and no training submitted.
10. The JSD default now selects this Libri100 manifest; other datasets require
    an explicit --jsd-manifest. Legacy MFA-full worker retains an explicit
    MFA-full selection. Small 64-row fixtures with 30 native sampling buckets
    failed to cover multi-utterance batches (100-second JSD yielded none).
    The current suite uses at least 1,024 rows, 100 seconds/rank, and requires
    actual multiple utterances in both tiny and full-size optimizer updates.
    Old validation outputs are preserved under before_k2_adoption timestamps.
    Job 3536077.opbs passed the integrated single-GPU suite in 125 seconds:
    JSD/CE continuous versus restarted models, cumulative averages and optimizer
    states had exactly zero difference; raw losses matched exactly. Full-size
    JSD and historical full-size CE initialization each completed a four-row
    native optimizer update with TIMIT hooks.
11. Job 3536124.opbs passed the four-node true-DDP suite in 94 seconds. Both
    objectives processed 16 utterances per global update; current/average models
    and optimizer states had zero difference after restart, and all four ranks'
    sampler, scaler and saved random states matched. Source-bound evidence is
    validation_ddp/DDP_PASS.json.
12. Extended job 3536162.opbs passed in 160 seconds. Both objectives completed
    two epochs and 58 real tiny-model optimizer updates. Restart at the first
    epoch's final step 29 reproduced every saved state and every raw training
    loss bitwise; current and cumulative-average development/test hooks and
    TRAIN_COMPLETE markers were verified. Evidence is
    ../zapformer_native_acceptance_20261012/results/EPOCH_PASS.json, including
    the separate validator's source hash. These are short integration fixtures,
    not completed Libri100 convergence experiments.
13. The initial performance script filename shadowed Python's stdlib profile
    module: job 3536217.opbs failed at import in 17 seconds without training.
    Its log is retained. After renaming to native_gpu_probe.py and validating
    imports, 3536260.opbs passed full-model single-GPU budgets 100/200/320 in
    101 seconds. 3536266.opbs passed 640/960/1280 in 146 seconds. Full-model
    two-node 3536273.opbs and four-node 3536272.opbs each completed 16 real
    native optimizer updates, in 47/48 seconds. Performance probes explicitly
    use offline W&B and omit multi-GB checkpoint persistence; loss, augmentation,
    BF16, auxiliary gradients and optimizer are unchanged.
14. User confirmed independent seed42 initialization and 10 epochs for each
    ordinary/Star-Star Libri100 experiment, followed by the final Star-Star
    checkpoint for Libri960. Formal ordinary Libri100 job 3536313.opbs uses
    one node, 640 seconds/rank and original upstream JSD. This budget provides
    substantial memory headroom while retaining 298 audio seconds/node-second.
    Controller 3536361.opbs resumes normal wall-time checkpoints, then begins
    independently initialized S3T2** only after ordinary TRAIN_COMPLETE. It
    does not repeat failed processes. Formal convergence is not yet complete.

## Acceptance criteria

The complete migration requires all of the following, with actual GPU
evidence and source/configuration identity:

- Native source hashes unchanged; augmentation features and private RNG match
  the unadapted native augmentation exactly; CE time warp moves labels with
  features and preserves ignored padding.
- JSD and CE continuous training versus process restart: compare model,
  cumulative average, optimizer, scheduler, progress, sampler, scaler and all
  saved random states. Current floating checks remain `rtol=1e-5, atol=1e-6`;
  integer and random states must match exactly. Do not loosen tolerances to
  conceal a resume failure.
- Full-size random JSD model and historical ordinary-S3T2 JSD checkpoint used
  for CE both complete real native optimizer updates and evaluation hooks.
- Real multi-node DDP validates both objectives and independent rank states.
- Single-GPU `validation/SMOKE_PASS.json` and multi-node
  `validation_ddp/DDP_PASS.json` both exist and match the validated source.

The markers contain the validated adapter source hashes. Formal entrypoints
verify these hashes instead of treating file existence as acceptance. Resume
checks are scoped to the same environment/hardware and world size; cross-size
or cross-version bitwise identity is not asserted.

Native ParamPlotter and smoothed reporting windows restart with the process;
their window summaries are not treated as continuous training-state evidence.
GPU throughput tuning and Libri100 formal training follow acceptance, in that
order. Do not start the old MFA-full formal script merely because tests pass.
