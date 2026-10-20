# Current tracking

The original eightGPU validation3542236 exited1 after2m08s: JSD+FS completed six full960-second DDP updates and3+3 resume with bitwise model/average/optimizer/scheduler/scaler/RNG/sampler/native statistics and raw loss replay. Validation then rejected insufficient reserved-memory headroom: uninterrupted88.88GiB, resumed94.00GiB of95GiB. Actual peak allocated was66.11GiB. No CUDA OOM occurred; the memory acceptance guard failed. Other three arms were not validated in that job. Formal dependency jobs3542237–3542240 finished without starting after the failed dependency.

All960 validation checkpoints, logs and W&B identities are preserved in `validation_960_job3542236`. See `validation_960_job3542236/RESULT.json` for actual DDP throughput and failure evidence. SingleGPU960 throughput remains a successful measurement, but cannot establish safe eightGPU resume memory usage.

All four arms now use one common640-second sampler setting, with unchanged model/objective/native loop,50epochs,8GPUs and seed42 initialization. The successful320/640/960 singleGPU measurements and criterion acceptance remain valid because no Python runtime files changed.

Replacement validation:3543233, regular-g routed to small-g. It verifies all four640-second full-model DDP updates, throughput, memory and six versus3+3 restart. Log: `validation_640_recovery.log`.

Replacement formal jobs (each8GPUs,50epochs; afterok3543233):
- JSD+FS:3543234
- JSD-only:3543235
- CE+FS:3543236
- CE-only:3543237

Authoritative job history is in `jobs.json`; PBS status must be rechecked when reporting. Formal checkpoints and per-update raw/throughput logs will appear under `models/<arm>`. W&B uses online mode by default with offline fallback. This recovery does not alter other historical models or snapshots.

Update:3543233 passed all four full8GPU640-second validation chains. Formal jobs3543234–3543237 have been released from dependencies and are queued. See `ALL_ARMS_PASS.json` and `THROUGHPUT_REPORT.json` for exact resume and measured memory/throughput evidence.

Update 2026-10-13T08:37:06.867764+00:00: The user cancelled JSD-only and requested deletion before later retraining. PBS3543235 was deleted. Interactive checkpoints (latest step850,8completed epochs), local training/evaluation logs, optimizer/resume state and saved W&B ID were removed from models/jsd_only. The static experiment spec and shared lock file remain. The remote W&B run y1t9xl5k remains historical; future training must initialize a new run. Cancellation provenance: reports/jsd_only_deleted_20261013.json. No replacement job was submitted. Other arms continue.

Update 2026-10-13 17:47 JST: queue confirms JSD+FS3543234, CE+FS3543236 and CE-only3543237 still running. No stop action taken. JSD-only remains cancelled. Status/PER/FA report and protected checkpoint snapshot: [STATUS.md](reports/status_20261013T084606Z/STATUS.md). JSD+FS still improves on dev; stopping scope is awaiting user clarification. This is IF-MDD standard JSD, distinct from upstream OTTC JSD.

Update 2026-10-14T09:06:36.677940+00:00: User requested stopping all three remaining Libri100 arms. PBS shows no unfinished jobs; 3543234/3543236/3543237 are FINISH. All three TRAIN_COMPLETE.json records confirm epoch50/step5300 completed on October13, before this request. No qdel was needed and no continuation was submitted. All existing checkpoints/logs remain; final checkpoints additionally hard-linked in reports/stop_request_20261014. Current request is to leave all three stopped, not to resume. This records completion of the configured budget, not proof of convergence. Audit: [action.json](reports/stop_request_20261014/action.json).
