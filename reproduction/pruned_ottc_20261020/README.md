# Batch-local, chunked OTTC

This is an opt-in implementation for a **new training recipe**, not a modification
of any active training/resume contract. No jobs are started by this module.

The design follows [PrunedCTC at e604792](https://github.com/yfyeung/PrunedCTC/blob/e60479249598274a0ccb7b93074d40ab95090a11/pruned_ctc.py).
It adapts vocabulary reduction and chunked recomputation to the existing OTTC
objective; it does not replace OTTC with CTC or depend on k2.

## Preserved and adapted behavior

| Component | This implementation |
| --- | --- |
| Batch vocabulary | Sorted union of valid target IDs in one forward/microbatch |
| Local IDs | Contiguous from zero; local zero is a normal target, not blank |
| Global blank | Excluded from targets; retained in full normalization if the model has that output class |
| Padding and repeats | Ignore -100 padding; retain every target occurrence and its order |
| Output head | Keep the original full weight/bias parameters; no new per-batch classifier |
| Normalization | Full-vocabulary logsumexp, accumulated in FP64 in vocabulary chunks |
| Retained scores | Selected log probabilities and, for JSD, one background scalar per valid frame |
| Backward | Recompute vocabulary chunks; update all output classes and encoder states |
| Alignment | Original OTTC CDF transport, not a CTC graph or beam-pruned lattice |
| Inference | Unchanged full-vocabulary base forward, independent of training targets |

For global targets `[17,42,17]` and `[42,9,-100]`, the batch vocabulary is
`[9,17,42]`, and local targets are `[1,2,1]` and `[2,0,-100]`. The next batch gets
its own mapping. Gradients still go to the original global output-head rows.

The current trainer accumulates several microbatches per optimizer step. The
mapping is computed per actual forward, not over the entire accumulated batch.
This changes neither the full normalization nor the summed objective.

## Original JSD, not a replacement divergence

The original function is imported from
[archived_ot_loss.py](../zapformer_s3t2_native_20261012/archived_ot_loss.py).
Its reverse-KL convention, epsilon handling, and per-class contributions remain
unchanged. No standard-JSD substitution or subset softmax is performed.

Let `F(probability, target_value)` denote that function's original per-class
output, including its own target smoothing. For a one-hot target at global ID
`token`, the same cost can be grouped as:

```text
background[frame] = sum_over_full_vocabulary F(probability[frame, class], 0)
cost[frame, token] = background[frame]
                   + F(probability[frame, token], 1)
                   - F(probability[frame, token], 0)
```

The all-zero entries are internal per-coordinate background contributions, not
a new target distribution. Background values and their derivatives are computed
in chunks using the original function and PyTorch autograd. Their gradients
remain in the final loss, including gradients through the learned transport
weights. Simply dropping the background would change the original objective.

Only one-hot occurrence targets are supported. Oracle-generated targets,
Euclidean costs, and the separate standard-JSD branch are rejected by the model
adapter rather than silently approximated.

## Training integration

Use the mixin when constructing the model in an independent recipe, before
creating the optimizer. Keep that recipe's existing model configuration and
training schedule:

```python
from ottc.models.sequence.wav2vec2_ottc import wvForOTTC
from reproduction.pruned_ottc_20261020.model import PrunedOTTCMixin


class PrunedOTTC(PrunedOTTCMixin, wvForOTTC):
    pass


model = PrunedOTTC(config, pruned_chunk_size=128)
```

Here `config` is the new recipe's existing `wvOTTCConfig`, with a linear head,
`regular` or `regfrozen` forward, and `crossentropy` or original `jsd` cost.
The adapter preserves the base forward's encoder, mass head, dropout, and
alignment-freezing behavior. During training it defers the head projection and
returns `.loss` with `.logits=None`; the repository's `QuietTrainer` uses this
interface. Eval with labels and label-free inference use the unchanged dense
base forward. Parameter names and state dictionaries remain compatible.

The model configuration records the chunk size and vocabulary/cost policy. Add
this directory's Python sources and the archived loss source to the **new**
recipe's source hashes and snapshots. Recreate the same subclass when resuming.
Do not edit old recipes, bypass their hash checks, or restart completed models.
Keep existing per-epoch weights, best-dev weights, resumable checkpoints and
training scalar logging policies.

The lower-level [pruned_ottc_loss](loss.py) accepts encoder states, full head
parameters, padded global target IDs, and normalized frame masses directly.
Disable autocast around this loss call; the mixin does so automatically.

## Memory and numerical limits

- `chunk_size=4096` matches PrunedCTC's default. For BPE500, use a smaller chunk
  such as 128 to actually split the vocabulary. A chunk at least as large as the
  vocabulary still materializes one complete frame-by-vocabulary block.
- Working projection storage scales with valid frames times chunk width, plus
  valid frames times selected vocabulary size. Dense parameters, their gradients,
  optimizer states, encoder activations and transport metadata remain.
- The JSD background needs an additional forward vocabulary pass and temporary
  per-chunk derivative computation. Memory reduction is not a speed guarantee.
- FP32 and BF16 inputs are supported; FP64 is available for reference checks.
  FP16 and higher-order derivatives are unsupported. Projection arithmetic uses
  at least FP32, as in PrunedCTC, so legacy BF16-rounded head outputs need not be
  bit-identical. The original JSD's extreme-probability limitations remain.
- The deferred head assumes a non-reentrant model forward and a fixed linear
  output head. Do not resize or replace the classifier after adaptation.
- No GPU peak-memory, throughput, real-data training, or recognition claim is
  made by the CPU checks. No evaluation data or metrics are uploaded.

## Local verification

```bash
python reproduction/pruned_ottc_20261020/check.py
```

The check only needs CPU PyTorch. It loads the actual archived functions through
AST, changing only hardcoded CUDA device literals to CPU in the reference
transport. It also executes the original model's regular/frozen forward methods
with a tiny encoder, without requiring pretrained assets or a GPU.

Coverage includes 22 dense-reference cases (CE/JSD, FP64/FP32/BF16, chunk sizes,
optional bias, frozen head, padding, repeated targets, encoder/head/frame-mass/
target-mass gradients), numerical gradcheck, and four training-adapter cases.
The latter check changing batch vocabularies, three optimizer steps across the
joint/frozen phases, bounded projection widths, compatible state dictionaries,
and unchanged inference. Values, errors, software version and source hashes are
saved locally in [checks/cpu.json](checks/cpu.json).

## CUDA and real-audio validation (2026-10-20)

PBS job `3617181` completed successfully on NVIDIA GH200 (95 GiB visible),
PyTorch 2.6.0+cu126. The same 22 numerical cases and four model-adapter cases
passed on CUDA. The [CUDA check report](checks/gpu_3617181_20261020T022608Z/cuda_checks.json)
retains the full errors and source hashes.

The [real-audio results](checks/gpu_3617181_20261020T022608Z/RESULTS.json) cover
the first two and longest two TIMIT training utterances, CE and original JSD,
and FP32 and BF16 encoder arithmetic: eight cases in total. Every loss matched
at the recorded FP32 precision, all parameter-gradient comparisons passed the
predefined tolerances, and repeated losses were stable. Maximum absolute
parameter-gradient errors were about 1.1e-6 for FP32 and 1.25e-4 for BF16.
The selected batch vocabularies contained 34 and 50 tokens, without blank.

Both comparison heads projected in FP32 to match the pruned arithmetic. These
are not comparisons against a legacy BF16-rounded dense head. Encoder dropout,
scalar attention dropout, layerdrop and SpecAugment were disabled; no optimizer
updates or recognition evaluation were performed. Each case used one warmup
and only two timed forward/backward samples. Optimizer-state allocation and
audio-loading time are not included.

| BPE500 JSD, chunk 128 | Dense peak GiB | Pruned peak GiB | Dense ms | Pruned ms |
| --- | ---: | ---: | ---: | ---: |
| First two, FP32 | 2.4430 | 2.4420 | 108.5 | 121.2 |
| Longest two, FP32 | 2.4982 | 2.4985 | 110.1 | 124.0 |
| First two, BF16 encoder | 2.4347 | 2.4350 | 120.2 | 133.0 |

There was no material total peak-memory reduction for BPE500 (differences below
0.05%). The FP32 CE samples were about 4-5% slower; the FP32 JSD samples were
about 12% slower. These small-sample timings are indicative, not a throughput
benchmark. The longest BF16 JSD dense samples were 413 ms and 119 ms, versus
130 ms and 131 ms for pruned, so their aggregate ratio must not be interpreted
as a speedup. Large-vocabulary scaling remains unmeasured.

The first attempt, job `3617058`, passed the CUDA unit checks but failed the
real-audio comparison because the probe had not disabled functional attention
dropout. Its [failure record](checks/gpu_3617058_20261020T022005Z/FAIL.json) and
source snapshot are preserved. Only the probe's stochasticity controls were
fixed; no model/loss changes or tolerance relaxation were needed.

To repeat in an allocated GPU environment:

```bash
python reproduction/pruned_ottc_20261020/check.py --device cuda
python reproduction/pruned_ottc_20261020/gpu_probe.py --output /path/to/new/local/results
```

The provided [probe.pbs](probe.pbs) uses the established environment and a
20-minute limit. It writes immutable per-run source snapshots, data/audio/model
hashes, full local metrics, and PASS/FAIL records. It does not touch existing
training jobs or upload evaluations. The implementation is validated for this
bounded setting, but these results do not justify making it the BPE500 default
on performance grounds.