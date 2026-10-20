# TIMIT BPE500 OTTC JSD / CE

The default `run_interactive.sh` entry runs full TIMIT training for 40 epochs
per cost through `full/run_interactive.sh`. Matching probe checkpoints are
imported without resetting optimizer state. Full-run outputs live in full/;
see full/README.md for epoch archives, final averages and training tracking.
Use `--probe` only when explicitly requesting the legacy 200-step test.

## Legacy probe protocol

Two independent, local-only 200-optimizer-step probes reuse the original OTTC
`jsd` and `crossentropy` implementations and the phase-aware gradient checks.
No CTC, AED or SPOTTC run is included. Existing 40k experiments are unchanged.

Inputs are the existing speaker-disjoint TIMIT 4620/400/192 manifests. Targets
use the locally trained LibriSpeech-960 BPE500, not the public Icefall
Unigram500. Text is lowercase and internal apostrophes are retained; the old
40k recipe deleted apostrophes, so this is not a tokenizer-only controlled
comparison. No supplied transcript is used for audio-only greedy decoding.

Both costs use seed 42, official Wav2Vec2-large-LV60 SSL initialization, a new
500-class output head, native 20ms frames, effective batch 64, microbatch 2,
learning rate 2e-4 and the original 500-step warmup/40-epoch schedule. The
probe stops after 200 steps (about 2.74 epochs, still within warmup), or its
1500-second training budget. It does not establish final convergence.

Each GPU job first tests three smoke steps with actual step-1 checkpoint
resume, joint/frozen alignment phases and finite gradients. Failure blocks
the learning probe. The original trainer atomically retains per-epoch weights,
best-dev weights and full latest resume checkpoints. These partial probes do
not run final-five/final-ten averaging or test-set acoustic inference.

Full-dev WER/CER and raw training loss are in models/ottc__COST/history.jsonl.
decode/before and decode/after preserve first-8-train/first-20-dev predictions
and frame IDs. PROBE_RESULTS.json summarizes these diagnostic subsets only.
All logs, predictions and checkpoints stay local; W&B uploads are disabled
through reproduction/wandb_policy.py. No automatic continuation is submitted.

Prepare once with `train.py --prepare`, then validate with `train.py --check`.
Submit run.pbs independently with BPE_COST=jsd and BPE_COST=ce.

## Interactive GPU allocation

Inside an existing PBS GPU interactive session, run:

```bash
bash reproduction/timit_bpe500_20261020/run_interactive.sh jsd 7200
bash reproduction/timit_bpe500_20261020/run_interactive.sh ce 7200
# Or run both sequentially within one shared total budget:
bash reproduction/timit_bpe500_20261020/run_interactive.sh both 7200
# Only for the original 200-step diagnostic:
bash reproduction/timit_bpe500_20261020/run_interactive.sh --probe both 2400
```

The budget is capped by remaining PBS walltime minus 900 seconds for
checkpoint/evaluation work. Both costs share the total budget, not 7200 seconds
each. Full training saves and exits on budget exhaustion; rerun the same
command to continue toward 40 epochs. It skips a cost only after its full
training and benchmark evaluations complete, not after 200 steps.
The shared interactive lock refuses concurrent probe/full runs. No job is
submitted or cancelled by this entry. Full-run logs live in full/jobs/;
legacy probe logs remain under jobs/. The unchanged legacy script is retained
as run_probe_interactive.sh, including its duplicate queued-job checks.