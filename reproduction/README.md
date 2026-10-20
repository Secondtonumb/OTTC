# Official README TIMIT fused-phoneme reproduction

当前四头扩展实验的 CTC、OTTC CE、OTTC 上游 jsd 均已完成 40 轮，
K2-FSA forced alignment / text-free inference 评估已完成。
最新 loss 曲线、完成状态和结果请打开
[四头实验报告](multihead_timit/notion_update/README.md)。本文记录此前的原始单头复现。

Source: `idiap/OTTC`, commit `0d97e7e`. The official README's two TIMIT
commands run via `launch.py` and the original `run_exp.py` entry point.
No Haopeng/IF-MDD checkpoint is loaded. The encoder is the original
`facebook/wav2vec2-large-lv60` checkpoint at revision
`0cde644b64dac88d8416bec1c92a4099b850ba0b`; tensor hashes and the two
Torch weight-norm key conversions are in `provenance/upstream.json`.

## Training protocol

| Setting | OTTC | CTC |
|---|---|---|
| Dataset | `timit_fused_phoneme`, original loader | same |
| Train / test | train 4,620 / kaldi_test 240, including SA | same |
| Encoder | Wav2Vec2-large-LV60, CNN frozen | same |
| Heads | original linear logits + linear OT weights | original linear logits |
| Optimizer | AdamW, weight decay 0.005 | same |
| Learning rate | 2e-4 | 5e-5 |
| Batch / epochs | 64 / 40 | same |
| Warmup / scheduler | 500 steps / linear | same |
| Forward | regfrozen, CLI frozen epochs 30 | CTC |
| Prepare function | `prepare_dataset_for_timit_phoneme` | same |
| Seed | 42 (HF default, also before creating heads) | same |
| Eval / save | every epoch; keep last 5 checkpoints | same |

The official code converts 30 frozen epochs to `int(30 * 4620 / 64) = 2165`
steps, while each epoch actually has `ceil(4620 / 64) = 73` batches. This
original approximation is retained. The checkpoint at epoch 40 is evaluated;
the test set is never used to select a checkpoint.

The official mapping keeps `q`, yielding 40 phone classes and 42 model
outputs including pad/unk. Vocabulary and mappings are not replaced with a
39-phone benchmark during training. SA sentences are retained as in the loader.

## Runtime fixes

`provenance/compatibility.patch` records changes against upstream:

- Enable the documented TIMIT path override and trust the local dataset loader.
- Parse CLI True/False correctly; handle an output directory with no checkpoints.
- Fix OTTC metric tuple indexing and stop mutating padded reference labels.
- Remove dropped/padded frames before OTTC decoding, so `<ignored>` is not
  counted as a falsely recognized phoneme.
- Allow both OTTC and CTC audio-only inference without reference labels.
- Preserve the batch dimension of OT weights when batch size is one; upstream
  `squeeze()` otherwise fails on single-utterance inference.
- Compute WER through the same jiwer scorer without remote metric downloads.
- Delay notebook-only IPython imports until the display helper is called.
- Save the processor, vocabulary, arguments, split sizes, Trainer state and metrics.

For the two original README runs, training architecture, OT loss, mapping,
collator, learning rates, and original freeze-step formula are unchanged.
An additional `jsd_standard` cost is available for the separate JS comparison
below; the existing CE and upstream `jsd` branches remain unchanged.
The dependency overlay in `vendor/` pins
Transformers 4.39.3, Datasets 2.18.0 and Accelerate 0.28.0, along with compatible
NumPy/PyArrow/fsspec. PyTorch is 2.6.0+cu126 for the installed GH200 environment
(README allows a CUDA-enabled PyTorch >= 2.3); Python is 3.10.14. No shared
environment package is changed. Optional PEFT/bitsandbytes are disabled in the
launcher because this recipe does not use them.

## Run and artifacts

```bash
qsub -o /work/gm64/m64000/OTTC/reproduction/results/train.log \
  /work/gm64/m64000/OTTC/reproduction/train.pbs
```

The wrapper uses one GH200 GPU, first OTTC then CTC. Actual training is on
interactive allocation `3504867.opbs`, mg0022, 2h, running `bash reproduction/train.pbs`
with output `results/train_interactive.log`. Both queued batch jobs `3504841.opbs`
and `3504860.opbs` were cancelled before execution after long scheduler estimates.
Successful two-step smoke jobs: `3504820.opbs` and
`3504846.opbs`. The initial regular-g job `3504841.opbs` was cancelled while
queued (scheduler estimated a 13h wait); no weights were trained by that job.
The short-g resubmission also received an estimated next-day start. The
interactive allocation immediately provided a free GPU and preserves all
training arguments. Audio-only evaluator validation completed in job `3504864.opbs`;
earlier evaluator probes exposed and fixed batch-one and missing-label bugs.
Resubmission resumes an existing run; use a new suffix for an independent seed.

Checkpoints:
`models/large_models_results/wav2vec2-large-lv60/runexp_{ottc,ctc}_fuse_phoneme/checkpoint-*`.
Each model directory also contains `processor/`, `vocab_used.json`,
`run_config.json`, `dataset_sizes.json`, `trainer_state.json`, epoch transcript
and alignment dumps, and `eval_results.json`.

Results:
`results/runexp_{ottc,ctc}_fuse_phoneme/` contains exact runtime/source hashes,
completion markers, per-utterance JSONL predictions and `metrics.json`.

`evaluate.py` supplies only audio to the trained model. It reports raw phoneme
sequence error rate, shared normalized no-sil phone error rate, and one-to-one
boundary P/R/F1/R-value at 10/20/30ms. Raw segmentation uses frame argmax class
transitions, including blank transitions, with frame index * 20ms timestamps
and no fitted time shift. Reference endpoints within 20ms of an utterance edge
are excluded. Correctly recognized tokens are DP-matched for interval overlap
and start+end timestamp error; unmatched reference durations stay in the
overlap denominator. These metrics do not implement the paper's unspecified
starting-frame F1 contract and must not be claimed as its exact table numbers.

Both the README's 240-item fused reference and the existing 192-item core-test
benchmark are evaluated. Optional `--cross-domain` adds Buckeye and L2-ARCTIC
from the existing local manifests as supplementary evaluation only. The
Trainer's padded-batch `eval_wer` is retained separately from trimmed audio-only
recognition scores, because upstream CTC decoding does not mask padded frames.
The training wrapper enables these supplemental cross-domain evaluations for
both models. The TIMIT result is in-domain (TIMIT-trained), unlike the earlier
LibriSpeech-trained zero-shot models; that training difference must be kept
explicit in any comparison table.

## CE and JS comparison

The original README OTTC checkpoint uses `bins_metric=crossentropy` (CE cost).
`train_js.pbs` adds two independently initialized OTTC runs with the same
TIMIT recipe and seed 42: upstream `jsd` and `jsd_standard`. The original
`jensen_shannon_divergence` uses the reversed KL direction in PyTorch:
`.5 KL(m || p) + .5 KL(m || q_epsilon)`. This is preserved as `jsd` for a
faithful comparison. The new `jsd_standard` implements
`.5 KL(p || m) + .5 KL(q || m)`, with `m=(p+q)/2`, natural logs, exact
zero-target terms and stable log-space prediction arithmetic. The cost alone
changes; no loss rescaling or learning-rate tuning is applied.

```bash
qsub -o /work/gm64/m64000/OTTC/reproduction/results/train_js.log \
  /work/gm64/m64000/OTTC/reproduction/train_js.pbs
```

Each new run has a separate suffix `runexp_ottc_fuse_phoneme_{jsd,jsd_standard}`.
`check_js_cost.py` verifies standard JS against SciPy, symmetry, finite extreme
logits, numerical gradients and batched GPU transport gradients. Two-step
smoke training and checkpoint reload precede each full 40-epoch run.
`summarize_js.py` checks all completion markers, epoch/step/configuration,
checkpoint loading and every prediction JSONL before writing `RESULTS_JS_ZH.md`.
The original CE/CTC report and training-source archive remain separate.

Both JS runs completed on the existing interactive allocation `3504867.opbs`,
mg0022, with log `results/train_js_interactive.log`; the allocation was released
after completion. Training runtimes were 13.85 minutes (`jsd`) and 14.27 minutes
(`jsd_standard`). Source snapshots and changes are archived separately under
`provenance/js_source/` and `provenance/js_changes.patch`. The comparison plots
and convergence curves are exported by `plot_js_comparison.py` to
`results/js_comparison/`. These are single-seed, unchanged-hyperparameter cost
comparisons, not a search for the best tuned performance of each loss.

## Soft-DTW baseline

`train_softdtw.pbs` runs the original `loss_type=softdtw` implementation with
CE local costs, gamma 0.1 and normalize=False. It uses the same official
LV60 encoder, seed 42, fused-phone dataset/vocabulary, CNN freeze, batch 64,
40 epochs, AdamW weight decay .005, 500-step warmup and linear scheduler.
Learning rate 5e-5 is inherited from the README CTC baseline: the README
lists Soft-DTW as available but does not give a separate complete TIMIT
Soft-DTW command. Soft-DTW uses a single linear phone head and does not use
the OTTC weight head or regfrozen training schedule. Final epoch 40 is used
for all evaluations without test-based checkpoint selection.

The upstream `soft_dtw_loss.py` is unchanged, including CPU/Numba dynamic
programming, per-utterance loss computation, GPU/CPU copies and batch-mean
reduction. `check_softdtw.py` verifies CE costs and logits gradients against
an independent differentiable PyTorch DP at gamma .01/.1/1 and checks
finite GPU loss/gradients. Necessary integration fixes are archived in
`provenance/softdtw_changes.patch`: route Soft-DTW to existing recognition
metrics, avoid in-place target mutation, and trim logits by each input
length instead of a potentially mismatched batch mask.

```bash
qsub -o /work/gm64/m64000/OTTC/reproduction/results/train_softdtw.log \
  /work/gm64/m64000/OTTC/reproduction/train_softdtw.pbs
```

Actual run uses interactive allocation `3506088.opbs`, mg0033, with log
`results/train_softdtw_interactive.log`. The wrapper performs two-step smoke
training/reload, full training, then the same 3,172-utterance audio-only
recognition and segmentation evaluation as the CTC/OTTC models.
`summarize_softdtw.py` validates final checkpoint/configuration/loading and
every prediction JSONL, records the checkpoint SHA256, and writes
`RESULTS_SOFTDTW_ZH.md`. `plot_js_comparison.py --include-softdtw` exports
benchmark, convergence and shared-utterance segmentation PNG/PDF files to
`results/softdtw_comparison/` while preserving the prior JS comparison.
Training sources and hashes are in `provenance/softdtw_source/` and
`provenance/softdtw_source.json`.

The 40-epoch run and all 3,172 evaluations completed successfully. Training
took 17.35 minutes, and allocation `3506088.opbs` was released afterward.
The raw TIMIT phone error rate is 74.61%; the current gamma/lr configuration
has severe deletions and long phone segments. This is a baseline result,
not evidence of best achievable Soft-DTW performance. The final-checkpoint
weight check confirms the frozen CNN is unchanged and early/late Transformer
weights have updated; the evidence and phone output counts are saved with
the results.

## W&B tracking

All five completed experiments have verified historical W&B curves; links and
logging contracts are in `WANDB_ZH.md`. `run_exp.py` now enables W&B together
with TensorBoard by default, uses the suffix as the run name, records train
loss every 50 steps and eval loss/PER every epoch, and preserves the W&B run
identity across checkpoint resumes. Existing 40-epoch histories retain their
original 200-step recording interval. Tracking changes do not retrain or alter
the completed checkpoints. Project/entity defaults are in `wandb_config.json`;
CLI/env overrides and explicit offline/disabled modes are available.
The final audio-only evaluator publishes recognition/segmentation summaries
to each live run. GPU nodes without connectivity retain offline W&B records
for later synchronization. A three-step CPU Trainer check verifies the
actual live logging integration independently of the speech models.
