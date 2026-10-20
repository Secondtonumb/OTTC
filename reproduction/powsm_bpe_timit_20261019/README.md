# POWSM-token TIMIT text recognition

## Four independent experiments

| Output directory under models/ | Target head | OTTC bin cost |
| --- | --- | --- |
| ottc__jsd | Native POWSM tokens | Original upstream `jsd`, not standard JSD |
| ottc__ce | Native POWSM tokens | Original `crossentropy` |
| spottc_s3t2__jsd | Three states per POWSM token | Original upstream `jsd` |
| spottc_s3t2__ce | Three states per POWSM token | Original `crossentropy` |

All four start from the same local official Wav2Vec2-large-LV60 SSL weights,
with a random output head, seed 42 and native 20 ms stride. No POWSM acoustic
weights or previously trained phone model is used. Train/dev/test sizes are
4620/400/192, preserving the existing disjoint-speaker manifests (train includes
SA utterances). Each full run uses 40 epochs, effective batch 64, microbatch 2,
learning rate 0.0002, 500 warmup steps, weight decay 0.005 and bf16. The feature
encoder is frozen; the alignment encoder freezes after 30 completed epochs.

## Tokenizer and topology

POWSM revision `21ffa410432ace159f3c1fdeed304c70eddf7d34` is pinned. Although the
official config calls it BPE, this SentencePiece model is UNIGRAM with 40000
pieces and a 40002-entry ESPnet output inventory. Use native output IDs, not
raw SentencePiece IDs. The whole output inventory is retained; blank is 0.
See data/metadata.json for hashes and the exact source-manifest provenance.

Text is NFKC-normalized, lowercased, apostrophes deleted and other nonletters
treated as word separators. This is an explicit adaptation to the original
POWSM tokenizer's lack of apostrophe pieces: `don't` becomes `dont`. Every
prepared utterance has zero unknown pieces and an exact normalized roundtrip.

The SPOTTC adaptation uses the frozen IF-MDD `make_subphone_labels(..., 's3_t2')`
and state-reset decoder. Each nonblank output ID receives three states in
order; stable numeric surrogate names avoid interpreting a text token as a
phone, silence or a control symbol. No token is pruned. There are 120004 output
classes including blank, versus 40002 for plain OTTC. State labels and the
token-to-state mapping are saved with each run.

As in the existing SPOTTC OTTC branch, training expands every target into all
three states and uses uniform state mass. It is NOT an FSA forward loss and
does not enforce the S3T2 optional-middle-state graph during training or greedy
decoding. The label expansion alone does not distinguish S3T2 from its starred
variants. This limitation must accompany any interpretation of topology results.
Both costs call the existing upstream model/loss unchanged.

## Validation, scheduling and results

`train.py --check` verifies all text/state roundtrips, repeated tokens, blank
handling, state padding, WER/CER and atomic epoch checkpoint averaging on CPU.
Each recipe's local-only GPU smoke uses the 12 longest training utterances,
tests three optimizer steps including joint and frozen phases, and deliberately
stops after step one before resuming optimizer/scheduler/RNG state. A matching
PASS.json is required before production; a failed smoke stops its job chain.

`dispatch.py --submit` submits two four-hour regular-g chains, one per cost.
Each chain validates both heads, trains OTTC first, then SPOTTC. After about
three hours of training budget it saves full resumable state and submits a
continuation. Source/data drift blocks continuation. DISPATCH.json and jobs/
record the contracts, submission IDs, parent jobs, logs and GPU telemetry.

Production runs keep separate W&B IDs and scalar logs plus complete local
history, config and source snapshots. Smoke/evaluation data and artifacts are
local-only; training does not upload model or evaluation artifacts.

Every completed epoch gets its own atomic model checkpoint and metadata with
epoch, optimizer step, model identity and hashes. All epochs are retained, in
addition to best-dev and the latest two full Trainer checkpoints. Final local
evaluation compares best-dev, last-five and last-ten averages with full member
provenance. It reports dev/test WER and CER, selects only by dev WER, and keeps
per-utterance reference/hypothesis/frame IDs. CER includes spaces. Decoding is
audio-only argmax, adjacent-ID collapse, blank removal, state-reset decoding
for SPOTTC, then native SentencePiece decode; no language model. No BPE boundary
segmentation score is claimed because no reference BPE boundaries are available.