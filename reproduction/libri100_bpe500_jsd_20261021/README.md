# Libri100 BPE500: OTTC-JSD and SPOTTC-JSD

Two independent runs use the original upstream OTTC JSD objective. The user
requested LibriSpeech train-clean-100 after the completed TIMIT comparison,
with two GPUs and approximately four hours per job. With no further batch
clarification available, this recipe conservatively retains global batch64:
32 utterances per GPU, one forward/backward, no gradient accumulation or
microbatch splitting. Both start from official Wav2Vec2-large-LV60 SSL weights,
not previously trained TIMIT or phone models.

## Submit and resume

```bash
source reproduction/libri960_continue5_20261017/environment.sh
"$LIBRI_PYTHON" reproduction/libri100_bpe500_jsd_20261021/prepare.py
"$LIBRI_PYTHON" reproduction/libri100_bpe500_jsd_20261021/train.py --check
"$LIBRI_PYTHON" reproduction/libri100_bpe500_jsd_20261021/train.py --submit --recipe ottc_jsd
"$LIBRI_PYTHON" reproduction/libri100_bpe500_jsd_20261021/train.py --submit --recipe spottc_jsd
```

`--submit --dry-run` previews a command. Runs queue independently, with no
afterok dependency. Duplicate active submissions and completed matching runs
are reused. Each job requests two GPU nodes, each with 32 CPUs and one GPU,
for four hours. The cluster disallows `ngpus` in `select`; the established
MPI/torchrun launcher starts one rank per allocated node using NCCL Socket.
Three hours cover preflight/smoke/training; one hour is reserved for checkpoint
and evaluation work. PBS exit 75 is a checkpointed budget pause, not convergence.
Distributed workers exit normally after a planned pause; the PBS wrapper checks
completion markers so torchrun does not mistake a checkpointed pause for a crash.
Resubmit the same recipe to continue. No automatic continuation or other
experiment is submitted. Four hours is not a claim that all 50 epochs fit.

GPU smoke uses the longest 192 training utterances, physical32 on each rank,
three optimizer steps, actual step-1 checkpoint resume, and joint/frozen-head
gradient checks. Smoke failure blocks training. CPU DDP tests use a synthetic
loss fixture and do not attest to real GPU JSD performance.

## Data and training contract

The existing manifest eligibility is retained: train-clean-100 28538,
dev-clean 2703, dev-other 2864, test-clean 2620, test-other 2938. Original
excluded-manifest paths/hashes are recorded; these are not claimed as all
official 28539/2939 training/test-other files. Text is normalized with the
existing lowercase/apostrophe-preserving BPE500 normalization. The tokenizer
is the verified local SentencePiece BPE trained on Libri960 training text,
not the separately stored public Unigram500 asset. No dev/test text trained
the tokenizer. IDs 0..3 are reserved; targets use IDs 4..499.

Formal training: 50 epochs, 446 optimizer steps per epoch (22300 total), seed
42, learning rate 2e-4, 500 warmup steps, weight decay 0.005 and BF16. The final
batch is 29 utterances per rank. The original regfrozen switch is after epoch30.
Non-reentrant Wav2Vec2 activation checkpointing reduces memory without splitting
batches. DDP unused-parameter handling supports the frozen alignment phase.

Each epoch writes atomic named model weights. Retention keeps dev-best3,
last10 and detected evaluation dependencies, plus best-dev and latest2 full
optimizer/scheduler states with both rank RNGs. Mid-epoch states save every
200 steps or at the budget limit. Rank0 alone writes shared model metadata;
completion markers are written after both rank RNG saves and a barrier.
Original experiments and source snapshots are not modified.

## Recognition, topology and segmentation

Epoch validation reports greedy WER/CER for dev-clean, dev-other and pooled
dev data. Best-dev selection uses pooled dev WER. Final best-dev, last5 and
last10 candidates receive full dev/test recognition evaluation, with all dev
selection finished before test inference. Candidate evaluation restores its
own joint/frozen phase.

SPOTTC training uses the existing S3T2 state-uniform target expansion and 1498
outputs; no FSA or substitute divergence is added to the training loss.
Evaluation separately reports greedy decoding and the existing S3T2** K2
token-loop decoder: S0 and S2 of the same BPE token are mandatory, S1 is optional,
and outer-state repetitions are allowed. Neither decoder receives reference
transcripts or timings. Graph search restricts token outputs to ordinary IDs.

Libri's existing phone alignments are automatic, not human word-boundary
ground truth. Segmentation is therefore separately reported on the unchanged
TIMIT dev/test audio and human WRD boundaries as a zero-shot cross-domain
evaluation. These are text-free word boundaries, not phoneme segmentation or
forced alignment, and do not determine the Libri model selection.

Training scalar curves use persistent W&B IDs with offline fallback. Smoke,
recognition/segmentation predictions, code snapshots and evaluation reports
remain local via `wandb_policy.py` and `OTTC_EVAL_UPLOAD=0`.