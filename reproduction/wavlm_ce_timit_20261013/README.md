# WavLM Large single-head OT-weighted CE, larger batch

This is the user-requested CE experiment accompanying the original single-head
WavLM JSD batch64 run. It uses **original OT-weighted crossentropy**, with
`bins_metric=crossentropy` and `model.jsd=False`. It is not gold-frame CE.
The encoder starts from official `microsoft/wavlm-large` pretrained weights;
it does not continue a JSD checkpoint or an IF-MDD model. No matching completed
WavLM single-head CE checkpoint was found when this entry was prepared.

Run on a single allocated GPU:

```bash
CE_BATCH_SIZE=auto WAVLM_MAX_WALL_SECONDS=1800 \
bash /work/gm64/m64000/OTTC/reproduction/wavlm_ce_timit_20261013/interactive.sh
```

The invocation budget includes profiling/initialization and leaves five minutes
for saving and post-training work. It must be below the allocation's remaining
time. The same command resumes a matching unfinished CE run and its W&B ID.
The first selected batch is saved in `active_run.json` so subsequent auto
invocations resume that configuration. Explicit `CE_BATCH_SIZE=128` or `256`
selects that separately named output after it passes the profile gate.

The user authorized formal training after preparation. `run.pbs` invokes the
same single-GPU entry with a50-minute invocation budget in a1-hour allocation.
When a profile job is already queued, submit with its success dependency:

```bash
qsub -W depend=afterok:3547535.opbs reproduction/wavlm_ce_timit_20261013/run.pbs
```

The dependency requires successful profiling before formal training receives
its allocation. The exact-source profile gate is also checked by the trainer.
The current formal job is3547542.opbs. The initial probe3547316 failed because
the probe rejected the first native FP16 GradScaler overflow instead of allowing
its protected skip and scale reduction; no formal CE updates were performed.
That failed source/log/run identity is retained under `profile_failures/`.
The corrected probe3547535 records every real AMP skip, advances the LR scheduler
only on native successful updates, and requires finite optimizer updates in
both normal phases and the longest-input phase before issuing its PASS gate.

The bounded `profile.pbs` job measures batch64,128,256 using real TIMIT training
data, both sides of the regfrozen transition, and each batch's longest utterances.
Each regular phase visits the same complete4620-utterance training set once,
so candidate throughput comparisons do not depend on different partial subsets.
It logs raw loss, finite gradient norm, loading/transfer/forward-backward/update
time, peak allocated/reserved memory, and single-rank GPU telemetry. Each batch
profile has its own W&B run and never initializes production from probe updates.
Batch256 is selected only if measured throughput is at least5percent greater
than128 and both are safe; otherwise128 is preferred. If neither larger batch
passes, production is blocked. Profile and interactive entries share a lock.

Both physical and effective batch increase; there is no gradient accumulation.
TIMIT train4620, fused40 inventory, waveform normalization, released20ms WavLM
frontend, linear phone/mass heads, LR2e-4, weight decay .005, FP16, frozen CNN,
seed42, total40 epochs, and original loss code are preserved. Larger batches
reduce update counts. Warmup and alignment-freeze thresholds are scaled by
steps per epoch to preserve their original positions in epochs:

| Batch | Updates/epoch | Updates/40 epochs | Warmup updates | Alignment freeze threshold |
|---:|---:|---:|---:|---:|
| 64 reference | 73 | 2920 | 500 | 2165 |
| 128 | 37 | 1480 | 253 | 1097 |
| 256 | 19 | 760 | 130 | 563 |

LR remains2e-4; there is no unreported LR scaling. These schedule adaptations
are recorded in runtime, model/run config, and W&B. CE large-batch versus the
existing JSD batch64 changes both objective and batch and cannot isolate the
objective's effect. Raw CE and JSD losses also have different definitions;
compare recognition and segmentation on the same split and protocol.

Per-epoch dev400 loss/PER and text-free P/R/F1/R-value are logged while preserving
training RNG. Final-epoch40 selection follows the original single-head recipe.
Final test240, core192, dev400, Buckeye and L2-ARCTIC are evaluated separately
with the original text-free protocol and frozen IF-MDD R-value. Forced alignment
is not reported by this entry. W&B defaults and explicit overrides are honored;
offline logs are retained when needed. Completed matching runs are reused.

Production outputs include the actual batch:
`../models/large_models_results/wavlm-large/runexp_ottc_fuse_phoneme_ce_wavlm_b128_20261013/`
or `b256`, and the corresponding `../results/` directory. Profiling evidence is
in `profile/PASS.json`, `profile/candidates.json`, and per-batch JSONL here.

CPU validation covers syntax and epoch-relative scheduler scaling. GPU safety
and throughput remain pending until the probe or interactive entry executes.
