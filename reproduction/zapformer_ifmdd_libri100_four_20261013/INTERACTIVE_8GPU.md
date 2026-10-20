# Resumable JSD-only: two completed epochs per10minute allocation

Request eight nodes, one GPU/node, true DDP:

```bash
qsub -I -q interact-g -W group_list=gm64 \
  -l select=8:ncpus=32:mpiprocs=1 -l walltime=00:10:00
```

When the allocation starts, immediately run:

```bash
cd /work/gm64/m64000/OTTC
bash reproduction/zapformer_ifmdd_libri100_four_20261013/interactive_jsd_only_8gpu.sh
```

Repeat the same two commands for each new allocation. If interact-g is full, change `-q interact-g` to `-q regular-g`; an interactive request still waits for the selected resource pool.

Every call restores the latest `models/jsd_only/step-XXXXXXXXX.pt`: model, native cumulative float64 average, BatchedRubik optimizer, native50epoch LR scheduler, AMP scaler and all eight ranks' RNG/sampler/native statistics. The saved W&B ID is reused. This is the existing formal JSD-only experiment; the total training horizon remains50epochs.

`interactive_session.json` stores the next target, normally current completed epochs+2, capped at50. The absolute optimizer stop step is computed from the actual native Libri100 sampler plans. At the target epoch boundary the trainer saves and exits normally. If the allocation time budget arrives first, it saves a partial epoch; the next invocation keeps the unfinished target instead of advancing it by another2. No checkpoints or training histories are cleared.

Current saved checkpoint: step156, epoch2 batch50;1epoch is fully complete. The next target is3completed epochs, about162moreupdates to step318 if the native sampler counts stay106per epoch. From an epoch boundary, twoepochs are about212updates. Actual batch plans are computed before each segment rather than assuming106.

A10minute allocation includes model/resume setup, W&B, native evaluation, checkpoint writing and exit. The launcher reads PBS start_time/walltime and reserves90seconds before the allocation ends. Its wrapper subtracts startup/resume/W&B time from the training deadline; it doesn't allow another600seconds after startup. The first real5minute training segment reached156updates (~1.47epochs); step median was1.27s and mean1.63s including timed training/validation overhead. Twoepochs in10minutes are plausible, not guaranteed. All formal full TIMIT evaluation/FA calls remain enabled; startup/saving/network variation can reduce progress.

Defaults: EPOCHS_PER_SESSION=2, INTERACTIVE_MAX_SECONDS=600, INTERACTIVE_SAVE_RESERVE=90, batch640s/rank,8GPUs,BF16,original IF-MDD standard-JSD SPOTTC,S3T2**,FS=0,encoder aux.05,baseLR.02,save/eval50steps,average200steps,last8checkpoints retained. Example oneepoch target: `EPOCHS_PER_SESSION=1 bash .../interactive_jsd_only_8gpu.sh`. Do not set old SP4_STOP_AFTER; targets are managed automatically.

The matching queued formal JSD-only job (resolved from jobs.json) is held while you repeat manual interactive sessions. The default KEEP_BATCH_HELD=1 retains that hold between allocations so the batch trainer does not start in between. To return to queued batch training after a session, run `/opt/pbs/bin/qrls -h u 3543235.opbs` (or the matching job ID printed by the launcher). Set KEEP_BATCH_HELD=0 to release a hold created by this invocation on exit. If already running, the launcher exits without starting a competing trainer; existing user holds remain unchanged. The same `.training.lock` protects the experiment. After the interactive segment, the queued formal job can continue from its checkpoint. Other arms are untouched; no new PBS continuation jobs are created by this launcher.

Outputs: `models/jsd_only/interactive_<jobid>.log`, `interactive_session.json`, `progress.json`, `updates.jsonl`, `throughput.jsonl`, `step-XXXXXXXXX.pt`, evaluation metrics, `wandb_run.json`. W&B online by default with repository defaults/offline fallback. CPU planning and wrapper checks passed; original full eightGPU worker/resume/memory acceptance remains unchanged. The new deadline/target wrapper must run in an allocation for its GPU segment measurement.

Absolute target example:

```bash
EPOCH_TARGET=20 bash reproduction/zapformer_ifmdd_libri100_four_20261013/interactive_jsd_only_8gpu.sh
```

This sets the cumulative completed-epoch target to20, retaining the existing native50epoch LR/optimizer recipe for checkpoint compatibility. Repeat the command in each new10minute allocation; the absolute target is also persisted if the next call omits EPOCH_TARGET. Time-limited partial epochs resume toward20. Once20epochs are completed, further calls return without training until a higher EPOCH_TARGET is explicitly supplied. This is a stop target, not a fresh20epoch training schedule. Keep the queued batch job held while working toward the target.
