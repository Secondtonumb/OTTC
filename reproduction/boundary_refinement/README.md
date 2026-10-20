# OTTC / K2 continuous boundary refinement

The default experiment **reuses a trained OTTC checkpoint without training**.
It requires the supplied phone sequence, as does the existing K2 forced aligner.
No reference boundary timestamps enter the refinement function.

## Frozen-mass path (start here)

`ottc.models.alignment.refine_from_ottc_mass` reads the existing OTTC alignment
mass alpha. For N phones, original single-head OTTC has target mass beta=1/N.
At each threshold j/N it finds the crossing cell and interpolates its time:

```
A[t-1] < j/N <= A[t]
r = (j/N - A[t-1]) / alpha[t]
candidate[j] = frame_edge[t] + r * (frame_edge[t+1] - frame_edge[t])
```

This uses the cumulative-mass splitting idea from
[CIF-PyTorch](https://github.com/MingLunHan/CIF-PyTorch/blob/main/modules/cif_middleware.py).
The implementation is original, device independent and supports multiple
thresholds in the same frame and zero-mass frames. It does not copy the reference
code's hardcoded CUDA allocations, threshold=1 assumptions or single-fire loop.
There is no quantity loss or tail token: the supplied phone count is already known.

The global candidate equals the existing `native_ot_spans` interpolation under
the same masses, target masses and clock. **The new part is its K2 consistency
gate**: candidate j is accepted only inside a window around K2 boundary j.
Otherwise the original K2 time is retained. We do not paste a remote crossing's
fraction into a different K2 frame or clamp a remote crossing onto a window edge.

The window radius is min(20 ms, half the preceding coarse-boundary gap, half the
following gap), using 0 and duration at the ends. This makes adjacent windows
nonoverlapping and preserves token order even with mixed acceptance. Repeated
phone positions are kept. `accepted` measures consistency with this window,
not confidence or accuracy. The 20 ms radius is an explicit untuned default.

```python
from ottc.models.alignment import refine_from_ottc_mass

# alpha: [T], only valid frames from the original OTTC alignment branch
# k2_times: [N-1], internal K2 boundaries in seconds
out = refine_from_ottc_mass(alpha, k2_times, duration, max_shift_ms=20)
refined_times = out.times
global_ot_times = out.global_times
accepted_mask = out.accepted
```

Default frame edges use **duration/T**, matching native-k2. A different clock
must be supplied explicitly as `frame_edges` (T+1 edges). Fractions assume mass
is uniform within each frame cell; they are continuous estimates, not new
sub-frame acoustic measurements. Optional `target_mass` supports explicit beta;
do not silently apply uniform beta to SPOTTC substate topologies.

## Original single-head TIMIT checkpoints

The exporter reuses step2920 (40 epochs, TIMIT train4620 including SA):

| Selector | Existing checkpoint directory below `reproduction/models/large_models_results/wav2vec2-large-lv60/` |
|---|---|
| `jsd_upstream` (primary) | `runexp_ottc_fuse_phoneme_jsd/checkpoint-2920` |
| `ce` (control) | `runexp_ottc_fuse_phoneme/checkpoint-2920` |

The original upstream JSD loss is unchanged. CE is the existing OT-weighted CE
model, not a newly trained framewise-CE model. These are not the four-head runs.
Phone strings are mapped through each original processor vocabulary (40 phones,
42 output labels including special tokens; q and dx retained).

**Extraction:** call the original model forward with `model.step=2920`, eval
and no-grad. For these regfrozen models, alpha comes from
`wav2vec2_freeze -> lm_weight -> softmax`, not the current prediction encoder.
The prediction hidden states are captured in that same pass only to support the
optional learned-head experiment. All base parameters remain frozen. Cache
metadata includes checkpoint/manifest hashes and the K2 source snapshot/hash.

## Run

Use the existing Python3.10 CUDA environment, without modifying its packages:

```bash
cd /work/gm64/m64000/OTTC
PY=/work/gm64/m64000/miniforge3/envs/sb_k2_cuda/bin/python

# Run on a GPU allocation. --limit 0 exports the full named manifest.
$PY reproduction/boundary_refinement/run.py export \
  --model jsd_upstream --split dev --limit 0 --output /path/to/cache/jsd/dev
$PY reproduction/boundary_refinement/run.py evaluate-frozen \
  --cache /path/to/cache/jsd/dev --output /path/to/results/jsd/dev
```

Repeat for `--split test` and `--model ce` using distinct output directories.
Dev is 400 utterances; test is core192. The exporter fails on decoding/mapping
errors rather than silently removing failed utterances. Nonempty output
directories are protected from overwriting. A complete cache has `index.json`;
partial caches cannot be evaluated. Cache size includes float32 hidden states.

For a short GPU validation (4 dev + 4 test per model, **no training**):

```bash
/opt/pbs/bin/qsub reproduction/boundary_refinement/smoke.pbs
```

Use the PBS file's environment setup if invoking the CLI inside another GPU job.
It contains the required shared-library paths and the existing HF offline cache
settings. The single-node debug job follows the Miyabi dispatch convention.

Each evaluation reports four readouts with the **same duration/T clock**:

1. `k2`: untouched native-k2, blank policy previous.
2. `k2_refined`: K2 plus the local gate on fractional OT crossings.
3. `ot_global`: global fractional OT crossings, diagnostic replacement decoder.
4. `ot_global_frame_left`: the left edge of those same OT crossing cells, to
   isolate fractional interpolation from global mass placement.

Metrics: internal token-boundary MAE/P90; unlabeled P/R/F1/R-value at 5/10/20 ms;
accepted fraction; number of reference boundaries outside the movable windows.
All internal phone transitions (including silence, q, dx) are included; utterance
endpoints are excluded. Indexed MAE includes every boundary, not just accepted
or well-aligned cases. P/R uses the frozen IF-MDD directional-matching scorer.
These are **supplied-phone forced-alignment** results, not text-free segmentation
or PER, and not numerically interchangeable with older start/end MAE protocols.

`predictions.jsonl` preserves both K2 and refined times, global candidates,
acceptance masks and reference times for analysis. References are used only in
scoring, never in candidate/gate computation. Cache entries also retain the
original K2 segments with their token identities and frame indices.

Timing measures only the postprocessor after warmup, including its validation
and CUDA synchronization overhead. It excludes encoder, K2, disk and host/device
copies. Reusing alpha may require the extra frozen alignment encoder versus a
K2-only optimized prediction path; **do not interpret readout timing as total
alignment speed**. Cached model-forward/K2 timings include startup effects and
are not a controlled end-to-end speed benchmark.

## Optional learned local head (separate future experiment)

`CIFBoundaryRefiner` predicts positive mass over 8 virtual local bins from acoustic
features, neighboring phone embeddings and relative position. The median mass
crossing determines a continuous position in the same constrained window.
Uniform initialization retains K2 boundaries. Virtual bins parameterize density;
they do not increase encoder resolution. Only this new head requires training.

```python
from ottc.models.alignment import CIFBoundaryRefiner, boundary_loss

head = CIFBoundaryRefiner(feature_dim=1024, vocab_size=42)
out = head(features, k2_times, phone_ids, duration)  # no reference timestamps
loss = boundary_loss(out, gold_internal_times)     # SmoothL1, beta=5 ms
```

For this optional experiment, export `train` and `dev` caches, then:

```bash
$PY reproduction/boundary_refinement/run.py train \
  --train-cache /path/to/cache/jsd/train --dev-cache /path/to/cache/jsd/dev \
  --epochs 10 --output /path/to/head/jsd
$PY reproduction/boundary_refinement/run.py evaluate \
  --cache /path/to/cache/jsd/test --checkpoint /path/to/head/jsd/best.pt \
  --output /path/to/results/jsd/learned_test
```

The loss is boundary SmoothL1 in ms, with no CTC loss or changes to the old
CE/JSD objective. Gold outside a window is not clipped or filtered. Train uses
train timestamps; selection uses dev MAE, including the identity epoch0 candidate.
Test never selects checkpoints. W&B is on by default alongside local raw logs,
with offline fallback. This initial driver only starts fresh runs; it rejects
nonempty outputs and does not implement training resume. The learned-head
optimization/checkpoint path has unit coverage, not a full TIMIT training result.

## Tests

Full frozen-model evaluation is available via `qsub reproduction/boundary_refinement/full.pbs`.
The completed run `results/full_3544607` covers dev400/test192 for both models
without exclusions. Generate its audited Markdown/CSV tables with:

```bash
python reproduction/boundary_refinement/summarize.py \
  --root reproduction/boundary_refinement/results/full_3544607
```

All four full-set MAEs slightly worsen after local refinement; keep K2 as the
default. See `results/full_3544607/summary/report.md` and the updated Chinese
record at `seamless_communication/docs/m4t/timit_ottc_cif_inventory.md`.

The standalone modules require only PyTorch and work on CPU:

```bash
OMP_NUM_THREADS=1 python -m unittest discover -s reproduction/boundary_refinement -v
```

Tests cover the CIF example, multi-crossing and zero-mass cells, analytical
gradients, native-OT equivalence, accepted/fallback behavior, noncrossing output,
repeated/degenerate boundaries, identity initialization, explicit clocks,
fractional-time learning and checkpoint reload.
