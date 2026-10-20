# TIMIT Zapformer JSD continuation

This entry extends the completed original 40-epoch single-head TIMIT run to a
larger **total** epoch count (default 80). Finish the original command first.
It reads the parent checkpoint without modifying the parent's code, checkpoints,
logs, W&B run, or final benchmark. The extension has its own output and W&B run,
with the parent's run ID, step, and configuration hashes recorded as ancestry.
Repeated invocations resume the extension's own checkpoint and W&B ID.

Inside an interactive GPU allocation:

```bash
TIMIT_EPOCHS=80 TIMIT_MAX_WALL_SECONDS=3600 \
bash /work/gm64/m64000/OTTC/reproduction/zapformer_timit_extension_20261013/interactive.sh
```

Choose a per-invocation time budget below the allocation's **remaining** time,
leaving at least ten minutes for initialization, saving, and final evaluation.
At roughly 3.3 minutes per epoch, epochs 41–80 take about 2.2 hours before final
evaluation. A one-hour invocation saves its checkpoint and can be resumed in
another allocation using the same command; do not change `TIMIT_EPOCHS` for an
already initialized extension output. A different total needs a separate output.

Only training duration and the explicit continuation LR policy change. The
original archived upstream OTTC JSD, features, vocabulary, native encoder,
optimizer, auxiliary loss, augmentation, sampler, and metric protocol are reused.
The full model, cumulative native average, optimizer moments, scaler, sampler,
and RNG states are loaded. Initial parent recipe fields and source hashes must
match; the parent must have its final completion marker. The original selected
best checkpoint is inherited as the dev-PER comparison candidate.

The first 40 epochs retain the native 40-epoch InterpCosine horizon. Later epochs
hold the native final LR: base LR .02 × min_factor .025 = .0005 before native
parameter-group scaling. There is **no learning-rate restart** and no retrospective
80-epoch cosine schedule. This is a continuation of a 40-epoch recipe, not an
80-epoch recipe trained from scratch. Any later LR restart is a separate experiment.

The extension logs actual training loss, steps, epochs, LR, full development
PER/loss and text-free P/R/F1/R-value locally and to W&B (offline fallback).
Historical parent losses are not copied or interpolated. Final dev/test text-free
and supplied-phone K2 FA summaries are uploaded. Select checkpoints with dev PER;
test results are reported at completion and do not control continuation decisions.

Validation: syntax, native scheduler continuity, constant-tail behavior, and
scheduler save/load checked on CPU. GPU checkpoint continuation has not yet been
executed for this new entry; the original recipe's GPU resume evidence remains
in `../zapformer_timit_20261013/VALIDATION.md` and is not claimed as an extension test.
