# Blank-free OTTC transducer on TIMIT

This is a research model that retains the RNN-T Encoder / Predictor / Joint
structure and replaces the RNN-T alignment likelihood with the original
upstream-JSD OTTC objective. It is **not standard RNN-T**, and it makes no
claim of equivalent sequence likelihood or streaming behavior.

The existing data and experiments are unchanged. The earlier
`reproduction/rnnt_timit_20261017` directory is an unvalidated standard-RNN-T
draft from before the loss discussion; this directory is the blank-free recipe.

## Training and inference

For each target occurrence u, the predictor sees BOS and phones strictly before
u. Repeated phones occupy distinct target bins: `[t,t,aa]` has three bins,
not two. A private BOS exists only in the predictor's input embedding.

The original external TIMIT vocabulary remains unchanged on disk: pad0,
phones1..40, unk41. The model classifier has exactly **40 phone outputs**;
its internal IDs0..39 map to external1..40. It has no blank, pad, unk, EOS,
CTC collapse, or RNN-T forward-backward loss.

Let a=softmax_t(mass_head(h_t)), b_u=1/U, and Gamma(a,b) be the original
one-dimensional monotone CDF transport. The main loss is

`L_OT = sum_(t,u) Gamma_(t,u) * D_upstream_jsd(softmax(Joint(h_t,g_(u-1))), onehot(y_u))`.

Only positive transport pairs are evaluated, avoiding a dense T*U*hidden
lattice. CUDA geometry directly calls the frozen `batched_ottc_loss_bucketized`
with dummy one-dimensional features and discards that dummy cost. The actual
history-conditioned classification cost directly calls its original
`jensen_shannon_divergence` function. Neither function nor formula is modified.
The CPU reference uses the same sorted-CDF geometry for correctness checks.

The source is
`reproduction/zapformer_s3t2_native_20261012/archived_ot_loss.py`.
Its `jsd=True` is the historical upstream reverse-KL-to-mixture cost, **not
standard JSD**. Reducing the inventory from42 to40 outputs is explicit here
and in the run contract; its raw values are not directly comparable to the
existing42-output frame-only models. This extends the original frame-only
cost to predictor-conditioned frame/occurrence pairs.

Normalized temporal mass cannot predict the output count. A separate head
predicts `log_count = log(valid_encoder_frames) + Linear(masked_mean(h))`.
Its bias initially predicts .2 phones per encoder frame. The total objective is

`L = (sum_utterance L_OT + length_weight * sum Huber(log_count, log(U))) / utterances`.

The default length weight is1. This count loss is a deliberate added design
choice; it is not the CIF quantity loss and is not supplied by the Joint module.
No auxiliary frame classifier or standard RNN-T objective is used.

Recognition accepts only audio and its padding mask. It predicts U by rounding
the count head, constructs uniform bins using that predicted U, and greedily
chooses each phone to minimize its transport-weighted original upstream-JSD
cost, conditioned on previous generated phones. Every occurrence updates the
predictor, including repeats. The fixed safety cap512 and minimum1 are recorded
and clipping is reported. This first recipe assumes nonempty speech utterances;
it does not support empty-transcript detection or streaming inference.

The internal mass quantiles k/predicted_U determine boundaries. Recognition and
boundaries never receive reference labels, true counts, or forced alignments.
Count prediction errors can cause deletions/insertions; identical neighboring
phones need not have identifiable internal boundaries under classification loss.
Representation of repeats does not guarantee accurate repeat counts or timing.

## Data, evaluation and checkpoints

Reuse the frozen TIMIT40 manifests: train4620 / dev400 / core-test192.
Speaker/data inventory checks come from the existing data loader. Do not remove
adjacent equal labels. CNN is frozen; the Wav2Vec2 transformer is trainable.
`DATA_AUDIT.json` records that these existing manifests have **zero adjacent
equal target pairs in all three splits**. Accordingly, dedicated synthetic
tests validate repeat handling; this data cannot establish real adjacent-repeat
recognition performance. No new deduplication is applied by this recipe.
The initial recipe uses LV60, seed42,40 epochs, AdamW2e-4, weight decay.005,
warmup500, BF16 encoder/joint, FP32 transport/count/JSD, microbatch8 with8-way
accumulation (effective64). These defaults are a starting recipe, not tuned
or established performance.

Every epoch evaluates full development data. Minimum dev raw PER selects the
single best checkpoint; test is never used for selection. Report raw PER,
normalized no-sil PER, count MAE, and repeated-target utterance PER alongside
text-free boundary P/R/F1/R-value at10/20/30ms. Boundaries use fractional mass
quantiles, duration/valid_frames, excluded utterance endpoints, and the frozen
IF-MDD directional greedy scorer. Source hashes and full predictions are saved.
There is no forced alignment or MFA comparison in this recipe.

An independently named atomic `epoch_NNN.pt` is saved at every completed epoch,
with epoch/step/model configuration/contract identity. All epoch weights are
retained; `checkpoint_manifest.json` records their hashes. `resume.pt` also
retains optimizer, scheduler, RNG and batch cursor; `best_dev.pt` is separate.
After completion, evaluate best-dev and available final5/final10 averages on
development before selecting a configuration. Test is then reported separately
for all candidates. Average members include paths, epochs, steps and hashes.
Tiny or limited-data runs produce `SMOKE_COMPLETE.json`, never a production
training completion marker.

Each run uses W&B by default, with the repository's project/entity defaults and
environment overrides. Resumes retain the saved ID, and connectivity failure
falls back to offline logging. Local JSONL records retain raw loss components,
steps/epochs/LR, metrics, timing and memory. Final JSON summaries and averaging
provenance are uploaded as a W&B artifact. Sync retained offline directories
with `wandb sync` when networking is available.

## Commands

Run from the repository root:

```bash
source reproduction/ottc_transducer_timit_20261017/environment.sh
"$OTTC_TRANS_PY" -m unittest reproduction.ottc_transducer_timit_20261017.test_model -v
"$OTTC_TRANS_PY" -m reproduction.ottc_transducer_timit_20261017.validate \
  --output reproduction/ottc_transducer_timit_20261017/validation/new_cpu_check
qsub reproduction/ottc_transducer_timit_20261017/profile.pbs
```

The validation runner uses real audio with a random tiny encoder, verifies
exact uninterrupted/resumed weights, optimizer/scheduler/loss curves, W&B ID,
ten independent epoch snapshots, and actual final5/final10 arithmetic and
development-only selection. Its metrics are smoke diagnostics.

Before long GPU training, inspect the exact-recipe profile's source/data
contract, `PROFILE_PASS.json`, native CUDA equivalence test, longest-input
safety, loading/compute/optimizer times, throughput and GPU telemetry. A
submitted or running PBS job is not a profile pass. The profile does not save
production weights or automatically launch long training.

On an allocated GPU, a new production run or exact matching resume uses:

```bash
bash reproduction/ottc_transducer_timit_20261017/run.sh \
  --output reproduction/ottc_transducer_timit_20261017/runs/jsd_blankfree_seed42
```

`--max-wall-seconds` bounds each invocation; rerun the same command/output to
continue a paused run. Source/data/configuration changes are rejected on resume.
Do not use existing CE/CIF/OTTC checkpoints as matching transducer resumes.

References: [SpeechBrain Joint](https://speechbrain.readthedocs.io/en/latest/API/speechbrain.nnet.transducer.transducer_joint.html),
[original OTTC framework](https://arxiv.org/abs/2502.01588),
[RNN-T](https://arxiv.org/abs/1211.3711).
