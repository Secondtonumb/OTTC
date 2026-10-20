# Native Zapformer monophone reproduction

This directory keeps two user-requested monophone experiments separate:

- `pooled313_jsd10_*`: the unchanged 313-state output and all epoch10 model/optimizer parameters; logsumexp pools S0/S1/S2 into phone probabilities, and the original upstream OTTC receives single-phone targets.
- `mono105_seed42_*`: a genuinely 105-dimensional output head, independently initialized from seed42. Backbone/frontend/mass parameters are bitwise identical to a fresh native SP seed42 control. Nothing is loaded from the old trained checkpoint.

Both use the frozen native Zapformer and the archived original OTTC-JSD directly. Forward-sum is disabled; mass normalization remains unchanged. Physical head dimensions are 105 or 313; inactive control phones 1/2 are masked, valid training phones are 3..104, and recognition removes non-speech IDs103/104. Archive/adapter hashes, original inventory adaptation, exact LR, RNG, optimizer state and utterance IDs are recorded.

`diagnose.py` completed five full-model, four-utterance, 120-update tests. LR1 is .017118957847577866 and LR2 is .03423791569515573. No SpecAugment is used in these isolated diagnostics; native unscaled auxiliary coefficient is .05. Native module batch-count is fixed at each initialization (0 for random,1060 for inherited epoch10). This is a training-sample fitting test; it is not a LibriSpeech WER benchmark. SP and monophone raw losses use different inventories/targets, so compare fitting behavior and recognition rather than absolute raw values.

At LR2, fresh true105 PER was 2.26%; fresh SP native-complete PER was 1.69%. Epoch10 pooled313 PER was337.29%, with maximum frame mass .9998. These results shift attention toward the inherited bad training state; they do not establish a unique mass-collapse cause.

The original complete diagnosis is in `report.json`, `REPORT.md`, `diagnostic_manifest.json`, and `cases/*/COMPLETE.json`. Every independent case has its own W&B identity. `comparison/wandb_run.json` is an aggregate reporting run, not one of the training cases. Histories are not interpolated. Finished cases are reused; interrupted cases restore model/optimizer/RNG and reuse their W&B ID.

The original native training loop, Rubik optimizer, scheduler, SpecAugment and complete rank-specific checkpoint persistence are reused by `train_native.py`. Its task adapter changes parser/inventory/metric hooks and adds detached mass statistics. It does not replace the OTTC divergence. Native training logs now include globally normalized mean maximum frame mass and effective frame fraction, so emerging concentration is observable.

`monophone_metrics.py` uses the frozen IF-MDD TIMIT boundary and R-value scorer, the same mapped recognition scorer as the SP suite, and a supplied-phone standard CTC FSA for evaluation. This FSA is separate from the OTTC training objective. Text-free and supplied-phone FA metrics remain separate; the unchanged full-boundary protocol and source/R-value hashes are recorded. TIMIT dev400/core-test192 scores are cross-corpus diagnostics, not LibriSpeech WER or verified MFA paper topline comparisons.

Single-GPU continuous versus resumed native training was bitwise identical for model, cumulative float64 average, optimizer, scheduler and RNG before adding detached mass reporting. The final adapter undergoes the same check across eight GPUs in `validate_ddp.pbs`; `check_validation.py` produces `DDP_RESUME_PASS.json` only when full-model, all-rank state equality passes.

Commands from the repository root:

```bash
qsub -q regular-g reproduction/zapformer_monophone_20261013/diagnose.pbs
qsub reproduction/zapformer_monophone_20261013/validate_ddp.pbs
qsub -v MONO_EPOCHS=10 -W depend=afterok:VALIDATION_JOB.opbs reproduction/zapformer_monophone_20261013/run_8gpu.pbs
```

The first command reuses completed cases; a new complete diagnostic reproduction must use a new output directory. The validation command must likewise use a clean validation directory for a new independent replay. GPU/CUDA is required by the preserved upstream OTTC kernel.

The pending formal true105 run starts afresh from seed42, uses OTTC-JSD only, base LR .03423791569515573, native InterpCosine min_factor .025/linear_scale .5, native BF16+GradScaler, max_duration640/rank, and eight ranks. It initially trains10 epochs, following the user's earlier per-stage choice. A different epoch horizon must be explicitly reconciled with saved contracts on resume. It never uses the fitted four-utterance weights. The job cannot start until the eight-rank validation succeeds and matching adapter hashes are present. Each resume uses its saved W&B identity and checkpoints. Final ordinary/averaged recognition and boundary P/R/F1/R-value summaries are automatically logged/uploaded by the retained native tracking hooks.

All older checkpoints, source snapshots and W&B histories remain unchanged. The manifest and eleven immutable Fbank shards are the same native Libri100 dataset as the SP suite; the formal contract records actual manifest SHA256, full configuration, native vendor identity and dependency adaptation.
