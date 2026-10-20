# Interactive TIMIT training: Zapformer 10 ms + monophone OTTC

This is a **new TIMIT-trained experiment**, separate from the Libri100/MFA-trained
Zapformer models evaluated on TIMIT. No previous checkpoint is overwritten or
loaded. The shell entry defaults to upstream OTTC JSD, with an OTTC CE control.
The long training is started by the user inside their interactive GPU allocation.

## Start interactively

On the login node, request one GPU node:

```bash
/opt/pbs/bin/qsub -I -q regular-g -W group_list=gm64 \
  -l select=1:ncpus=32:mpiprocs=1 -l walltime=03:00:00
```

Wait until PBS gives you the compute-node shell, then run:

```bash
bash /work/gm64/m64000/OTTC/reproduction/zapformer_timit_20261013/interactive.sh
```

The default JSD run is saved under `runs/jsd_seed42`. To run the independent CE
control, inside a GPU allocation:

```bash
TIMIT_COST=ce bash /work/gm64/m64000/OTTC/reproduction/zapformer_timit_20261013/interactive.sh
```

`regular-g` is routed by the cluster to the appropriate queue (currently
`small-g` for this node count). An interactive request can still queue; its
main convenience is direct access to the GPU shell and live training output.
TIMIT's smaller data volume reduces epoch time, not the cost of each backbone
forward pass. No promise is made that a from-scratch model will converge as
quickly as a pretrained Wav2Vec2 model.

## Defaults

| Item | Setting |
|---|---|
| Backbone | Full existing native Zapformer, random seed42 initialization |
| Features | 80-bin Kaldi Fbank; waveform×32768, 25 ms window, 10 ms hop, snip_edges=False, dither=0, energy_floor=0 |
| Output | One output per input Fbank frame, 10 ms; no final time downsampling |
| Training | TIMIT train4,620, including SA1/SA2, 3.9425 hours |
| Dev / test | dev400 / core-test192; disjoint utterance IDs, no test training |
| Inventory | Original fused TIMIT40, including sil/q/dx; 42 logits including blank0/unk41 |
| Targets | One target per phone occurrence, uniform target mass; **no substates** |
| Objective | Original archived upstream JSD by default; `TIMIT_COST=ce` uses the same archived OTTC routine with jsd=False |
| Auxiliary task loss | No CTC loss and no FSA/Charsiu forward-sum loss |
| Encoder auxiliary gradients | Native Zapformer aux_loss_scale=.05, ×.25 after step2000, native GradScaler coupling |
| Optimizer | Native BatchedRubik beta1=.99 and native parameter LR groups |
| Schedule | Native InterpCosine, base LR .02, min_factor=.025, linear_scale=.5 |
| Precision / augmentation | BF16 + native GradScaler; native AlternatingSpecAugment |
| Batch budget | Maximum padded batch duration100 seconds; length buckets; every utterance once per epoch, no drop_last or speed perturbation |
| Budget | 40 epochs by default; save/stop after10,200 seconds per invocation |
| Evaluation | Dev loss/PER on native periodic hook and every epoch; final dev/test recognition + K2 FA using best dev-PER checkpoint |
| Selection | Minimum dev raw TIMIT40 PER; no test checkpoint selection |

Upstream JSD here means the preserved original cost
`0.5 KL(m || p) + 0.5 KL(m || q_epsilon)`, not standard Jensen–Shannon divergence.
The implementation imports `archived_ot_loss.py` directly; there is no replacement
formula. CE is **OT-weighted CE**, not framewise CE supervised by gold timestamps.
Training reads phone strings/IDs; reference timestamps are used only by evaluation.
The encoder uses BF16 autocast; OT mass generation, normalization and the archived
transport criterion run in FP32, matching the existing native OTTC adapter.

The native backbone, optimizer, schedule, augmentation and `train_one_epoch` are
reused. Task adaptations are explicit: 42-way monophone head, frame-preserving
projection, TIMIT inputs, deterministic padded-duration sampling, metric/logging
hooks and resumable single-GPU dispatch. All encoder parameters are trainable;
this does not inherit the Wav2Vec2 CNN-freezing/regfrozen recipe.

## Change budget or resume

Example: separate 20-epoch run with a smaller padded-duration budget:

```bash
TIMIT_OUTPUT=/work/gm64/m64000/OTTC/reproduction/zapformer_timit_20261013/runs/jsd_20ep_d60 \
TIMIT_EPOCHS=20 TIMIT_MAX_DURATION=60 \
bash /work/gm64/m64000/OTTC/reproduction/zapformer_timit_20261013/interactive.sh
```

Other settings: `TIMIT_LR`, `TIMIT_MAX_WALL_SECONDS`, `WANDB_MODE` (online by
default, offline fallback). Additional trainer CLI flags may be appended.
Choose a new `TIMIT_OUTPUT` when changing training settings. Matching reruns
resume automatically from `resume.pt`, including optimizer, scheduler, scaler,
native cumulative average, sampler cursor and all augmentation/RNG states.
The output lock prevents two processes writing the same run. A changed data,
model, objective, schedule or source contract is rejected on resume.

Ctrl-C / TERM requests a graceful save after a complete optimizer update.
Keep the wall-clock limit below the allocation limit (default leaves10 minutes
in a3-hour allocation). Forced termination/SIGKILL cannot perform a final save;
resume uses the last completed checkpoint. Use a new allocation and rerun the
same command to continue. `TRAIN_COMPLETE.json` is written only after final
dev/test evaluation succeeds. Already-complete runs exit without retraining.

## Outputs and scoring

- `resume.pt`: latest atomic full training state, saved every100 updates,
  every completed epoch and at requested/time-limit stop.
- `best.pt`: weights selected by dev PER, separate from the resume state.
- `history.jsonl`, `interactive.log`, `wandb_run.json`: raw transport loss sum,
  per-utterance loss, LR, step/epoch, dev PER and boundary metrics; same W&B run
  ID is reused on resume. Final metrics are uploaded as an artifact.
- `dev_epochNNN.json`: complete dev recognition and text-free boundaries.
- `final_dev.json`, `final_test.json`: best-checkpoint PER plus separate
  text-free and supplied-phone K2 P/R/F1/R-value@10/20/30 ms, and internal-boundary
  FA MAE. Predictions are stored alongside metrics.

Raw PER uses TIMIT40 reference phones from the same preparation used to train.
The optional normalized no-sil PER removes sil/q and maps dx→d. Earlier historical
Wav2Vec2 reports use their own reference preparation; do not assume those PERs
are directly comparable without matching the scoring protocol. Boundary metrics
use all internal token transitions, including silence; no utterance endpoints;
duration/T clock, frozen IF-MDD directional matching and R-value. K2 receives the
phone sequence for FA only; recognition receives audio alone.

The cached corpus under `data/` is already prepared. `prepare.py` validates it on
startup and records manifest/feature hashes. It refuses to overwrite incompatible
or partial caches. The native vendor hashes are checked before training and the
new task sources are recorded in the resume contract.

## Validation

`validate.pbs` runs short real TIMIT GPU checks: JSD/CE continuous versus resumed
updates on a tiny backbone, plus a full-backbone update and final dev/test/FA
smoke. These validation outputs use separate `validation_JOBID/` directories and
do not start the40-epoch production run or establish a held-out accuracy result.
