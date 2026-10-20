# OTTC upstream JSD with configurable Charsiu forward-sum

User-confirmed 2026-10-13: retain the original OTTC/JSD computation; replace only
the companion forward-sum. Reuse the existing 313-state output rather than add
a phone text encoder or cross-attention. Resume the completed Libri100 S3T2**
epoch-10 checkpoint, including Rubik optimizer, cumulative model average,
scaler and eight ranks' sampler/RNG states. Extend the native schedule to total
50 epochs. Actual epochs 1–10 retain their original loss and 10-epoch schedule;
this is a changed-loss continuation for epochs 11–50, not 50 epochs trained
with Charsiu loss from initialization. Earlier proposed CE stages are superseded.

The scalar objective is `ottc_weight * original_jsd_sum + forward_sum_weight *
forward_sum_raw`. The OTTC statement block is unchanged (AST checked); original
bucketized transport, mass head, three-state target expansion, exclusion of
control columns 1–6 and upstream `jsd=True` remain intact. The forward-sum
criterion is extracted unchanged from Charsiu's MIT-licensed `src/models.py`
with full-source/class hashes and license retained. Its score source is an
explicit adaptation: logsumexp over S0/S1/S2 phone probabilities, gather each
target occurrence separately, then log-softmax over valid target positions.
Repeated identical phones have separate target-position IDs.

Edit `loss_config.json` before preparing and validating a new experiment:

| Parameter | Default | Meaning |
| --- | --- | --- |
| `ottc_weight` | 1.0 | Multiply the unchanged OTTC sum |
| `forward_sum_weight` | 1.0 | Multiply the companion loss; zero disables it |
| `forward_sum_kind` | `charsiu_positions` | Charsiu position CTC; `native_topology` retains the old graph for comparison |
| `forward_sum_normalization` | `target` | Divide each utterance by target phones; also supports `input` or `none` for the Charsiu branch |
| `forward_sum_reduction` | `mean` | Mean or sum over the local batch |
| `forward_sum_blank_logprob` | -1.0 | Fixed blank score added before final normalization |
| `forward_sum_temperature` | 1.0 | Temperature of phone scores before position log-softmax |
| `forward_sum_zero_infinity` | true | Match Charsiu; log impossible utterance counts separately |
| `forward_sum_ctc_device` | `cpu` | Deterministic CPU CTC with differentiable transfers; optional `cuda` is subject to PyTorch deterministic-kernel availability |
| `forward_sum_output_beam` | 10.0 | Used only by the old native graph branch |

The native graph comparison requires `normalization=input`, `reduction=sum`.
OTTC retains its local batch-sum reduction even when Charsiu uses local mean;
the two weights explicitly control their relative contribution. Losses and
their weighted values are logged separately; values with different reductions
are not directly comparable. Blank -1 is a score, not a probability.

`continue_jsd.py --help` exposes corresponding hyphenated CLI options. Overrides
must match the output's saved specification; changed settings require a new
experiment and validation instead of silently altering a resumed run. Existing
checkpoints/source snapshots are preserved. A separate W&B run records this
changed-loss branch, links its parent run, and resumes its own saved ID.
Ancestor logs are retained under `ancestor_metrics`, never relogged as new loss.

Mode B: eight nodes, one GPU per node, true DDP, regular-g, 4h per segment;
duration 640s/rank, base LR .02, save/evaluate every 50 steps, native float64
average every 200 steps. Normal wall-time exits requeue; failed validation or
training prevents continuation. NCCL uses the accepted socket fallback.

Reproduction entry points, from repository root:

```bash
source reproduction/zapformer_s3t2ss_native_20261012/environment.sh
"$MFA_PYTHON" reproduction/zapformer_jsd_charsiu_fs_20261013/prepare.py --output /work/gm64/m64000/OTTC/reproduction/zapformer_jsd_charsiu_fs_20261013/training
qsub reproduction/zapformer_jsd_charsiu_fs_20261013/validate_8gpu.pbs
# Submit run_8gpu.pbs with afterok:<validation_job> after preparing the output.
```

Validation replays original Charsiu loss and gradients, checks unchanged OTTC
against the frozen model, and compares full-model eight-rank updates 1061–1062
with a process restart after 1061. Model/average/optimizer/scheduler/all-rank
states/raw losses must match bitwise. Native reporting accumulators are now
preserved across same-epoch resumes; a partial epoch does not update the best
completed-epoch training loss. This is a reporting-only persistence adaptation,
with native training-loop bytecode unchanged.

TIMIT dev400/core-test192 monitoring retains the frozen text-free scorer and
supplied-phone S3T2** forced-alignment graph, including P/R/F1 and R-value.
The latter is explicitly labelled a diagnostic graph that differs from the
new Charsiu auxiliary training graph. No inference decoder or scoring formula
is changed by replacing the training loss.
