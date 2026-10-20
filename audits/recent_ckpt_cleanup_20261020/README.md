# Recent experiment checkpoint cleanup

The user authorized inspection and cleanup of redundant checkpoints from
recent OTTC experiments. This audit discovered repository recipes dated
2026-10-13 through 2026-10-20 and only modified completed, compatible,
unlocked Trainer runs with a validated latest full resume checkpoint.

Deleted: 364 model/optimizer payloads from 26 runs, totaling 575.1706 GiB of
uniquely linked allocated blocks. Affected recipe families:

- reproduction/native_stride_ottc_20261018
- reproduction/timit_ssl_native10_20261019
- reproduction/crottc_timit_20ms_20261019

Retained: first epoch, best-dev epoch/weights, latest ten epoch weights,
final-five/final-ten averages and their members, validated latest model plus
optimizer/scheduler/RNG, and detected cross-experiment JSON references to
weights or checkpoint directories. No candidate had another hardlink.
All noncandidate files, including logs, configurations, predictions, metrics,
source snapshots and epoch metadata, were kept. 3,111 retained regular files
passed unchanged stat/size/mtime checks; those at most 5 MiB also passed
SHA-256 checks. Latest safetensors payload lengths and PyTorch archive
structures were validated before selecting older redundant payloads.

The reference index scans experiment JSON outside data, vendor, snapshots,
checkpoint internals and tracking-cache directories. Detected referenced
checkpoint directories protect every contained candidate. Incomplete runs,
held locks, unrecognized formats, missing best-state records, and absent
trainer locking contracts are skipped, with exact reasons in the manifest.
Libri960 continuation, pretrained base models, BPE500, CTC/smoke data and the
previously cleaned POWSM family were not changed by this execution.

plan.json records the dry run; approved_manifest.json is the fresh executed
inventory, including references, candidates, preserved signatures and skips.
unlink_journal.jsonl records fsynced before/after events. Candidate hashes use
existing epoch metadata when available and otherwise were computed before
deletion. execution_result.json records completion; quota_final.json records
758.5132 GiB of observed group-quota headroom. Lustre accounting and other
project writes can change this value independently of the cleanup.

Deletion was permanent, not a move to trash. Independent backups were not
verified; deleted early weights cannot be reconstructed exactly from averages.
No job was started or cancelled, no persistent lock was removed, and no
artifact was uploaded. The earlier BPE500 lock/recovery issue is separate.