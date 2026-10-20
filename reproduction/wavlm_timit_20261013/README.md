# Original single-head TIMIT OTTC JSD with WavLM Large

User-requested backbone replacement for the original Wav2Vec2-Large-LV60
single-head TIMIT experiment. This is separate from four-head, IF-MDD, and
Zapformer experiments. The original trained Wav2Vec2 checkpoints remain intact.

Matched recipe: TIMIT fused40, train4620, original README Kaldi test240 evaluation
each epoch, seed42, batch64, 40 epochs / 2920 optimizer updates, LR2e-4,
weight decay .005, warmup500, FP16, frozen convolutional frontend,
`regfrozen` with alignment freeze threshold2165 updates, linear phone/mass heads,
output/final dropout .1, original upstream `bins_metric=jsd`. No new divergence,
substates, forward-sum, or framewise alignment supervision are introduced.
The released WavLM backbone configuration is retained, including its native
internal dropout/SpecAugment settings; this is a backbone replacement under a
matched downstream recipe, not a controlled comparison of pretraining algorithms.

Official pretrained source: `microsoft/wavlm-large`, revision
`c1423ed94bb01d80a3f5ce5bc39f6026a0f4828c`, read from the existing IF-MDD Hugging
Face **pretrained cache** (not any IF-MDD trained acoustic model). The local
`reproduction/models/wavlm-large` symlink references that immutable snapshot.
Its 315,453,120-parameter WavLM encoder emits a native20ms grid. The legacy
`wav2vec2` attribute name in the reused OTTC code is a storage slot; the live
encoder and regfrozen snapshot are both actual `WavLMModel` instances.

PyTorch2.6 exposes weight-normalized positional convolution weights with
`parametrizations.weight.original0/original1`; the official old checkpoint names
are `weight_g/weight_v`. Loading maps these names without changing tensors.
Missing, unexpected, or mismatched weights reject initialization. GPU profiling
checks selected pretrained tensors, including both positional-convolution tensors.

Submit the complete workflow:

```bash
cd /work/gm64/m64000/OTTC
qsub reproduction/wavlm_timit_20261013/run.pbs
```

Single-GPU interactive entry (after obtaining a GPU allocation):

```bash
WAVLM_MAX_WALL_SECONDS=3600 \
bash /work/gm64/m64000/OTTC/reproduction/wavlm_timit_20261013/interactive.sh
```

This sets a one-hour **invocation** budget, including profiling/initialization,
with five minutes reserved for saving and post-training work. Choose a budget
below the allocation's remaining time. It first reuses a current successful
profile or runs the exact recipe probe, then trains toward total40 epochs.
If the budget expires, it saves at an optimizer boundary and marks `PAUSED.json`;
run the same entry again in a new allocation to continue. The full40-epoch LR
schedule and W&B run identity are preserved. Already evaluated runs exit.
The interactive and PBS entries share `.job.lock` to prevent concurrent writes.
The new time-budget callback and shell syntax were checked on CPU; the full
GPU workflow remains subject to the profile gate.

The job profiles before starting production: three real batch64 updates before
alignment freezing, three at/after the original freeze transition, and a batch
of the64 longest training utterances. It records loading, transfer,
forward/backward and optimizer timings, peak allocated/reserved memory,
single-rank/DDP status, and one-second GPU telemetry. `profile/PASS.json` is
required for training; source changes invalidate it. Profiling changes only its
own disposable model. Production initializes again from the official pretrained
WavLM weights. Profile and production have independent W&B runs.

Each epoch additionally records diagnostic development400 audio-only PER,
per-utterance raw OT loss and text-free P/R/F1/R-value, preserving training RNG.
These diagnostics do not change the original final-epoch checkpoint selection.
The final evaluator reuses the original single-head boundary extraction and
matching protocol with the frozen IF-MDD R-value function. It reports README240,
core-test192, dev400, Buckeye and L2-ARCTIC separately. This workflow reports
text-free segmentation; it does not claim forced alignment results.

Outputs:

- `job.log`, `telemetry.csv`, `profile/batches.jsonl`, `profile/PASS.json` here.
- Model/checkpoints: `../models/large_models_results/wavlm-large/runexp_ottc_fuse_phoneme_jsd_wavlm_20261013/`.
- Final metrics and per-utterance predictions: `../results/runexp_ottc_fuse_phoneme_jsd_wavlm_20261013/`.
- W&B defaults come from `../wandb_config.json`; explicit environment overrides
  are honored. Offline records are retained if network initialization fails.
  Matching checkpoint resumes reuse the saved production run ID.

Original comparison: `../results/runexp_ottc_fuse_phoneme_jsd/metrics.json`.
Its test240 and core192 results must not be mixed with each other or with dev400.
The user also has a distinct ongoing/finished TIMIT Zapformer scratch experiment.

Validation so far: syntax and full official WavLM CPU initialization passed,
including strict pretrained key loading and native gradient checkpointing.
GPU profiling/training is pending until the submitted allocation runs. A queued
job or CPU initialization is not a trained-model result.
