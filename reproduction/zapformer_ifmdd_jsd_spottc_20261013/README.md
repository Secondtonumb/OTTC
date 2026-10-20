# IF-MDD standard-JSD SPOTTC: interactive single-GPU test

This is the explicitly requested IF-MDD **standard-JSD** experiment. Original
upstream JSD, existing IF-MDD CE experiments, checkpoints and frozen snapshots
remain unchanged. The original IF-MDD repository is not modified.

## Run

On Miyabi, request an interactive GPU allocation:

```bash
qsub -I -q regular-g -W group_list=gm64 \
  -l select=1:ncpus=32:mpiprocs=1 -l walltime=01:00:00
```

Inside the allocation:

```bash
cd /work/gm64/m64000/OTTC
bash reproduction/zapformer_ifmdd_jsd_spottc_20261013/interactive_onegpu.sh
```

The script uses the scheduler's first assigned GPU, sources the existing
validated environment, and writes to a new timestamped `interactive/` directory.
It does not submit additional jobs. W&B is online by default, alongside local
logs; a networking failure in training falls back to offline logging.

Default `TEST_MODE=full` runs:

1. GPU numerical parity against live IF-MDD transport JSD and a separate dense
   interval-overlap oracle, including loss, logits/mass gradients, CE regression,
   logit clamp and JSD uniform fallback.
2. Native Zapformer tiny training: two updates continuously vs one update plus
   resume to two, comparing model/average/optimizer/scheduler and RNG states.
3. One full native Zapformer training update; four-utterance dev/test monitoring
   hooks include recognition and segmentation P/R/F1/R-value. This subset is
   an integration test, not a full held-out benchmark.
4. Four real Libri100 training utterances, native full model and BatchedRubik,
   200 updates, no augmentation, encoder auxiliary scale zero. Records raw JSD
   transport loss, LR, step/epoch, greedy and exact S3T2** FSA PER.
5. Decode/reference comparison, checkpoint and W&B diagnostic artifacts.

For only parity plus the fixed training utterance pilot:

```bash
TEST_MODE=pilot OVERFIT_STEPS=200 \
  bash reproduction/zapformer_ifmdd_jsd_spottc_20261013/interactive_onegpu.sh
```

For CPU numerical parity (also works on the login node):

```bash
TEST_MODE=parity \
  bash reproduction/zapformer_ifmdd_jsd_spottc_20261013/interactive_onegpu.sh
```

Overrides: `TEST_OUTPUT`, `TEST_MODE=full|pilot|parity`, `OVERFIT_STEPS`,
`GPU_DEVICE`, `WANDB_MODE`, `WANDB_PROJECT`, `WANDB_ENTITY`.
Use `WANDB_MODE=offline` to retain local logs; sync those completed directories
later with `wandb sync`. Each independent test should use a new output
directory. Native training stages reuse their run ID and checkpoint on a
restart; interrupted fixed4 pilots are retained and require a new output path.

## Implementation and provenance

`vendor/utils/losses/ot_loss.py` is byte-identical to current IF-MDD. Its standard
JSD is .5 KL(p || m) + .5 KL(q || m), m=(p+q)/2, with the original 1e-12
clamp and renormalization. No replacement runtime divergence is written.

The isolated copy of `SubphoneticTopologyLoss` adds `ottc_cost` with options
`ce` (class default, previous behavior) and `ifmdd_standard_jsd`. Both normal
and uniform-fallback transport calls select JSD explicitly. The Zapformer
adapter always requests `ifmdd_standard_jsd`. The exact change is in
`topology_cost.patch`; source and runtime hashes are in `source_manifest.json`.

State expansion, mass floor, clamping, fallback, reduction, MFA labels and
decoders otherwise follow the earlier migration. The inventory is 303 acoustic
classes including blank: 100 ordinary MFA tokens × 3 states, sil/spn × 1 state,
plus blank. No blank is inserted in targets. Greedy decoding follows IF-MDD;
separate FSA shortest-path recognition enforces S3T2**. Training has no CTC or
FSA forward-sum auxiliary objective.

The live reference topology has CE hardcoded; parity therefore changes only
the transport `jsd` argument to True in a reference-only wrapper, then calls
the original live IF-MDD loss. An independent dense transport/JSD oracle also
checks values and gradients. It exists solely in validation, not training.

Completed matching CE fixed4 outputs are reused as the comparison. Raw CE
and JSD losses have different definitions and should not be compared by size.

## Outputs

- `console.log`: interactive output
- `PARITY_PASS.json`: numerical checks
- `INTEGRATION_PASS.json`: training/resume checks, pilot losses and reused CE record
- `SUMMARY.json`: final diagnostic greedy/FSA PER
- `DECODE_COMPARISON.md`: utterance reference vs both decoders
- `fixed4/reference_decode.json`: every saved decode snapshot, including FSA states
- `fixed4/step-000000200.pt`: default pilot model and optimizer
- `*/wandb_run.json`, `*/updates.jsonl`: run identity and raw logging
- `PUBLISHED.json`: W&B artifact logging receipts

CPU parity and shell/Python syntax checks were completed during preparation.
GPU training/learnability results must be read from the interactive run; the
earlier CE pilot's 0% PER is not a JSD result.
