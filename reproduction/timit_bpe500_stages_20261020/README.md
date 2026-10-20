# TIMIT BPE500 staged validation

Order: `ctc` -> `ottc_ce` -> `spottc_ce` -> `spottc_jsd`.
This is validation order, not weight transfer. Each phase and stage starts from
official Wav2Vec2-large-LV60 SSL weights with a new prediction head, unless
resuming its own completed checkpoint. Old experiments, locks and snapshots
are untouched. Existing 200-step OTTC JSD results are not a passed stage.

## Parallel OTTC-CE physical64 run (latest user update)

The user requested an OTTC version while retaining running CTC job 3617026.
`ottc_ce_physical64_10ep/` trains original upstream OTTC `crossentropy` on all
4620 TIMIT training utterances for ten epochs with one physical batch of 64,
one forward/backward per step and no gradient accumulation or OOM fallback.
The final batch has 12 utterances; evaluation batch size remains two.

```bash
source reproduction/libri960_continue5_20261017/environment.sh
"$LIBRI_PYTHON" reproduction/timit_bpe500_stages_20261020/ottc_ce_physical64_10ep/train.py --check
"$LIBRI_PYTHON" reproduction/timit_bpe500_stages_20261020/ottc_ce_physical64_10ep/train.py --submit
```

Initialization is independent SSL, not CTC weights. Seed 42, learning rate
2e-4, 500-step warmup, ten epochs/730 steps and the tokenizer are unchanged.
The original regfrozen switch stays at epoch 30 (step 2191), beyond this
ten-epoch observation. The user explicitly waived waiting for CTC recognition
convergence and the small-set overfit gate. CTC smoke evidence covers shared
data/I-O only, not OTTC loss; CPU tests exercise trainer scaling and gradient
contracts with a synthetic fixture. Actual OTTC finite loss/head gradients
and physical64 memory usage are checked on the first GPU training steps.
All ten epoch weights, best-dev, resumable checkpoints and final last5/last10
dev/test results are retained as above. Evaluation artifacts stay local and
formal training scalar tracking remains enabled.

No existing job is cancelled. The separately prepared `ctc_physical64_10ep/`
was not submitted. Current CTC uses microbatch2 while this OTTC uses physical64,
so the comparison includes a batching implementation difference; raw CTC and
OTTC loss scales must not be compared as equivalent recognition performance.

## Full-data batch64 ten-epoch run (2026-10-20 user update)

After the 32-utterance diagnostic failed with all-blank predictions, the user
requested full TIMIT training with effective batch 64 for ten epochs before
review. The independent `ctc_batch64_10ep/` recipe implements that request:

```bash
source reproduction/libri960_continue5_20261017/environment.sh
"$LIBRI_PYTHON" reproduction/timit_bpe500_stages_20261020/ctc_batch64_10ep/train.py --check
"$LIBRI_PYTHON" reproduction/timit_bpe500_stages_20261020/ctc_batch64_10ep/train.py --submit
```

It uses all 4620 training utterances, batch 64 split into microbatches of two,
73 steps per epoch and 730 steps total. Learning rate 2e-4, 500 warmup steps,
seed 42 and SpecAugment are retained. The scheduler covers exactly ten epochs;
warmup ends during epoch seven. The final partial batch contains 12 examples.
This is fresh SSL initialization, not continuation of the failed small-set
weights. The existing successful CTC GPU smoke is reused with source/data
checks, but the overfit recognition gate is explicitly bypassed. Development
WER/CER is recorded every epoch and never causes early stopping. At ten epochs,
best-dev and final-five/final-ten averages receive local dev/test evaluation.
All ten epoch weights, best-dev and the latest two complete resumable states
are retained. Training scalar tracking remains enabled; evaluation uploads
remain disabled. A two-hour job reserves twenty minutes for saving/evaluation;
budget exhaustion saves state and requires resubmitting this same recipe.
No later stage is submitted automatically; old results and contracts remain
unchanged.

## Submit one stage

```bash
source reproduction/libri960_continue5_20261017/environment.sh
"$LIBRI_PYTHON" reproduction/timit_bpe500_stages_20261020/train.py --check
"$LIBRI_PYTHON" reproduction/timit_bpe500_stages_20261020/train.py --submit --stage ctc
"$LIBRI_PYTHON" reproduction/timit_bpe500_stages_20261020/train.py --status
```

Submission uses `regular-g`, one node, GPU 0, a two-hour allocation and a
6000-second shared training budget. Twenty minutes remain for checkpoint and
evaluation work. `--submit --dry-run` prints the command without submitting.
Only one recorded pipeline job may be active at a time. No existing job is
cancelled; later stages and continuations are never submitted automatically.
Submit the same stage again after a budget pause to resume. For the next stage,
replace `ctc` with its exact name; submission and execution both require the
previous stage's source/data-matched full-development learning gate.

## Gates and phases

Each job runs these phases in order, skipping only validated completion gates:

1. `smoke`: longest 12 training utterances and four development utterances,
   three optimizer steps, including actual step-1 save/restart. OTTC variants
   exercise joint and frozen alignment phases. Numerical checks are not
   recognition-convergence claims.
2. `overfit`: 32 unique training utterances chosen with seed 42; the same
   utterances are evaluated with audio-only decoding. Up to 40 epochs, batch 4,
   zero warmup and SpecAugment disabled. Stop when WER <= 10% and CER <= 5%.
   These are diagnostic settings and weights, never imported into full runs.
   Exhausting 40 epochs without passing blocks further phases.
3. `full`: unchanged TIMIT 4620/400/192 manifests and BPE500 tokenizer, batch 64
   (microbatch 2), learning rate 2e-4, 500-step warmup, 40 epochs and seed 42.
   Every epoch evaluates all 400 development utterances. The learning gate
   requires three consecutive evaluations at steps >= 500 with WER < 100% and
   both WER and CER below initialization. Gate success is not benchmark
   completion and does not require beating the preceding method.

Code 75 means paused with resumable state; code 2 means a completed diagnostic
failed its recognition gate. Inspect local results before changing a failed
recipe. CPU checks do not attest to GPU convergence.

CTC and OTTC have 500 outputs. SPOTTC uses the existing ordered S3T2 target
expansion/state-reset decoder: 1 + 3 * 499 = 1498 outputs, including three
states for each nonblank reserved ID; references contain only IDs 4..499.
This is state-uniform OTTC, not an FSA forward loss or graph decoder. JSD is
the original upstream OTTC cost, not a replacement divergence implementation.
Decoding is audio-only greedy without a language model or supplied text.
Raw CTC target-length-mean and OTTC per-utterance loss scales are not comparable.

## Results, checkpoints and provenance

`smoke/STAGE`, `overfit/STAGE` and `full/STAGE` have separate contracts,
checkpoints, logs and predictions. Each completed epoch atomically saves named
model weights, epoch, step and model/source/tokenizer identity. All epoch
archives and the best evaluation weights are retained. The trainer retains
the latest two complete resumable states; unfinished checkpoint directories
are preserved but not selected. Full-state completion markers follow the
model, optimizer, scheduler and RNG writes; model hashes are verified on resume.

Full runs evaluate best-dev, last-five and last-ten averages on dev and test
after training. Selection uses dev WER only; member hashes and predictions
stay local. No BPE boundary scores are claimed without reference boundaries.
Full training scalar curves use separate persistent W&B run IDs; network
failure falls back to offline training logs. Smoke, overfit and all evaluation
artifacts remain local via `wandb_policy.py` and `OTTC_EVAL_UPLOAD=0`.
Submitted source/data hashes are recorded in `SUBMISSIONS.json`; modifications
block execution/resume rather than silently changing an active experiment.