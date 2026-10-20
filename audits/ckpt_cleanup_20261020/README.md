# Checkpoint cleanup, 2026-10-20

The user requested checkpoint cleanup after the gm64 Lustre quota was exhausted.
74 weight/optimizer files were deleted from the explicitly named POWSM runs.
Their uniquely linked allocated blocks totaled 190.0635 GiB. Final observed
group-quota headroom was 189.8826 GiB; quota accounting lagged the unlinks.

Affected runs: gradient_check_fix/models/ottc__jsd, ottc__ce and
spottc_s3t2__jsd under reproduction/powsm_bpe_timit_20261019.
spottc_s3t2__ce was skipped entirely because its training lock remained held.

Preserved for each affected run: first epoch, best-dev epoch and best weights,
newest ten epoch weights, every weight referenced by the run's RESULTS.json
or local diagnostic JSON, and all logs/configs/metrics/predictions/source
snapshots. All full resume checkpoints of unfinished runs were preserved.
For benchmark-completed runs only, redundant older model/optimizer payloads
were removed while the latest complete resume checkpoint was retained.
All epoch metadata sidecars were retained, including hashes of removed weights.

All BPE500 checkpoints (including steps 146 and 200), CTC runs, smoke runs,
datasets and unrelated experiments were untouched. No job was cancelled or
started, and no persistent training lock was removed.

489 retained regular files were checked for unchanged stat identity, size,
allocated blocks and mtime; files at most 5 MiB also had SHA-256 checks.
Candidate epoch hashes come from their existing metadata, while redundant
full-checkpoint payload hashes were computed before deletion. The exact
candidate list and retained-file inventory are in approved_manifest.json;
unlink_journal.jsonl records durable before/after events. execution_result.json
and quota_final.json record completion and observed quota state.

Deletion was permanent, not a move to trash. Independent backups were not
verified; deleted early epoch weights cannot be exactly reconstructed from
averaged weights. This cleanup does not repair the earlier interrupted
checkpoint or stale interactive lock; training recovery remains separate.