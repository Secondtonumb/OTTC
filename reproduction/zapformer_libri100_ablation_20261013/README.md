# Libri100 Zapformer SP-OTTC loss ablation

User-confirmed on 2026-10-13: all four arms branch from the same completed
JSD epoch-10 checkpoint and continue to **total epoch 50**, with eight-rank DDP.
The existing JSD + Charsiu forward-sum run is reused without changing its source.

| Arm | OTTC cost | Forward-sum weight | Output directory |
| --- | --- | --- | --- |
| SPOTTC-JSD only | Original upstream `jsd=True` | 0 | `models/jsd_only` |
| SPOTTC-JSD + forward-sum | Original upstream `jsd=True` | 1 | `../zapformer_jsd_charsiu_fs_20261013/training` |
| SPOTTC-CE + forward-sum | Original upstream `jsd=False` | 1 | `models/ce_fs` |
| SPOTTC-CE only | Original upstream `jsd=False` | 0 | `models/ce_only` |

CE is the original OTTC cross-entropy **transport cost**, not framewise CE.
The OTTC code is imported directly from the frozen, previously reproduced
`archived_ot_loss.py`; this suite changes only its existing `jsd` argument.
The native three-state targets, learned frame masses, target masses, bucketized
transport, excluded control columns, and inventory adaptation remain identical.
JSD is the original upstream objective, not a replacement divergence formula.

All four arms share the native Zapformer backbone, 80-channel 10-ms Fbank inputs,
313-state acoustic output, trainable mass head, S3T2** inventory and supplied-phone
evaluation graph. Their model parameter names and shapes are identical. The
native Zapformer auxiliary gradient regularizers also remain enabled in every
arm. `only` disables both native-topology and Charsiu forward-sum contributions.

## Shared initialization and data

The starting checkpoint is
`../zapformer_libri100_s3t2ss_native_20261012/training/step-000001060.pt`,
SHA256 `302ff5e361fa2add46e336e63873d5597558855360cc25b8f10d950af3e2f74c`.
It contains epoch 10, optimizer step 1060, model, cumulative float64 average,
Rubik optimizer, scheduler counters, scaler and all eight ranks' RNG, sampler
and augmentation states. Each arm restores these states. This is a common
JSD warm-start comparison over epochs 11–50; epochs 1–10 used the original JSD
and native-topology forward-sum, with the original 10-epoch learning-rate plan.
It is not four independent 50-epoch runs from random initialization.

All arms use the same `../zapformer_libri100_data_20261012/training.json`,
SHA256 `5b41fd4cf9a8ae04088e0c9e79fa3af8ac98ad05f9fceb5ce209157b5975cb7c`:
28,538 utterances, 100.586496 hours, 11 immutable Fbank shards and native IPA
phone targets generated from word transcripts. Dynamic bucketing uses seed42,
duration 640 seconds per rank and 106 optimizer updates per epoch.

Training retains native Rubik beta1.99, base LR .02, native SpecAugment, BF16,
average period 200, save/evaluation interval 50 and the approved deterministic
k2 accumulation adaptation. The native scheduler is reconstructed for total
50 epochs while restoring its counters; step1061 LR is approximately .01712.

## Forward-sum and hyperparameters

The Charsiu class is extracted unchanged from its MIT-licensed `src/models.py`.
`charsiu_source.json`, `charsiu_forward_sum.py` and `CHARSIU_LICENSE` retain its
source identity and license. Its input is explicitly adapted from the existing
313-state output: logsumexp over three states per phone, gather each transcript
occurrence into a separate position and normalize over valid positions. This
adds no text encoder or attention parameters.

The loss is `ottc_weight * original_ottc_sum + forward_sum_weight * forward_sum_raw`.
OTTC uses local batch sum. Charsiu defaults to target-length normalization and
local batch mean; raw losses with these different reductions are not directly
comparable. CPU CTC with differentiable transfers preserves deterministic
backward computation. Fixed blank log-score is -1, temperature1, zero-infinity
true, and both enabled loss weights are1.

Each new arm's full configuration is in `configs/<arm>.json` and copied to its
output. Configurable fields include OTTC cost/weight, forward-sum weight,
kind, normalization, reduction, blank log-score, temperature, zero-infinity,
CTC device and native-graph output beam. The existing JSD + forward-sum arm
retains its previously validated configuration. To change parameters, create
and validate a new experiment; a checkpoint resume checks its saved configuration
and refuses silent changes. CLI overrides must match that saved specification.

## Validation and execution

`test_criterion.py --gpu` checks original Charsiu loss/gradient equality,
unchanged legacy total loss/gradients, unchanged JSD, CE loss and gradients
against the original upstream transport reference, and identical model structure.
It verifies exact zero forward-sum in both only arms.

`validate_8gpu.pbs` then tests each new arm using the full model and actual
Libri100 batches: updates1061–1062 continuously versus stopping after1061 and
restarting all eight processes. Models, averages, optimizer, scheduler, all
rank states, progress and raw loss records must match bitwise. Acceptance
markers bind the validated source/configuration hashes. Native reporting-only
accumulators persist across partial-epoch resumes; no training-loop formula is
changed. The existing JSD + forward-sum arm has its own equivalent validation.
`python reproduction/zapformer_libri100_ablation_20261013/audits/compare_arms.py`
replays the shared-contract/batch/LR checks and the first-step equality of JSD,
CE and forward-sum components across matching arms, writing
`SUITE_COMPARISON_PASS.json` with its source and input-log hashes.

From the repository root, with unused output directories:

```bash
source reproduction/zapformer_s3t2ss_native_20261012/environment.sh
stage=/work/gm64/m64000/OTTC/reproduction/zapformer_libri100_ablation_20261013
for arm in jsd_only ce_fs ce_only; do
  "$MFA_PYTHON" "$stage/prepare.py" --arm "$arm" --output "$stage/models/$arm"
done
qsub "$stage/validate_8gpu.pbs"
# After validation passes, submit each arm with its validation dependency:
qsub -v SPOTTC_ARM=jsd_only -W depend=afterok:VALIDATION_JOB "$stage/run_8gpu.pbs"
qsub -v SPOTTC_ARM=ce_fs -W depend=afterok:VALIDATION_JOB "$stage/run_8gpu.pbs"
qsub -v SPOTTC_ARM=ce_only -W depend=afterok:VALIDATION_JOB "$stage/run_8gpu.pbs"
```

Each formal job uses eight nodes, one GPU per node, one true DDP model,
regular-g and four hours per segment. Normal wall-time exits requeue only
that arm; failures stop continuation. `suite_status.json` records job identities.
`accepted_source.tar.gz` and its manifest freeze the accepted new-arm sources.
The reused JSD + forward-sum run retains its own immutable source archive.

## Tracking and evaluation protocol

Each arm gets a separate W&B run, linked to the epoch-10 parent, and resumes its
saved run ID. Defaults come from `../wandb_config.json`; local logs persist and
offline W&B logs are retained if networking fails. Ancestor raw logs are copied
under `ancestor_metrics` without interpolating or relogging them as new updates.
Track raw/weighted components, optimizer steps, epoch, LR, full configuration,
recognition and boundary metrics. JSD recognition, segmentation and averaging
results have reporting priority; CE jobs remain part of this authorized ablation.

Monitoring uses the frozen IF-MDD raw-PHN scorer and R-value formula, with their
source hashes recorded. TIMIT dev400 and core-test192 text-free and supplied-phone
forced-alignment metrics remain separate. Forced alignment retains the same
S3T2** diagnostic graph across arms; it is distinct from Charsiu's position-CTC
training graph. Report P/R/F1/R-value, full-set boundaries and equal-phone-count
TSE subsets separately, with common subsets for direct TSE comparisons.
These monitoring results are not a LibriSpeech ASR benchmark. Archived MFA CTMs
retain unverified model identity and are not evidence of exceeding the paper's
MFA topline.
