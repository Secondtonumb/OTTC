# Accepted ordinary S3T2 native migration

The user approved the verified k2 deterministic accumulation adapter on
2026-10-12. All 135 frozen Zapformer Python sources remain unchanged. The
original archived OTTC `jsd=True` objective is reused; no divergence formula
was replaced. Native graph/forward/posterior construction is retained, with
deterministic CUDA scatter and explicit terminal-row broadcast as recorded
dependency adaptations.

| Check | PBS job | Wall time | Evidence |
|---|---|---|---|
| Single GPU, multi-utterance JSD/CE resume, both full-size model updates | 3536077.opbs | 125 s | ../zapformer_s3t2_native_20261012/validation/SMOKE_PASS.json |
| Four-node true DDP, both objectives and every rank's saved state | 3536124.opbs | 94 s | ../zapformer_s3t2_native_20261012/validation_ddp/DDP_PASS.json |
| Two complete epochs, epoch-boundary restart, final current/average evaluation | 3536162.opbs | 160 s | results/EPOCH_PASS.json |

All recorded current/averaged-model and optimizer differences are zero. The
two-epoch test compares all checkpoint states and each raw loss **bitwise**:
each objective completes 58 updates, with restart after update 29. These are
integration fixtures, not full Libri100 convergence experiments. S3T2** and
future 960/MFA stages require their own configuration and validation.

`accepted_source.tar.gz` contains 165 source/script/documentation files,
including the frozen vendor. `accepted_environment.json` records archive/source
SHA256, exact versions and import locations. Accepted versions: torch
2.6.0+cu126, CUDA 12.6, k2 1.24.4.dev20260601+cuda12.6.torch2.6.0, Lhotse 1.33.0,
NumPy 2.2.6 and W&B 0.21.0. Hardware is GH200 120GB. Data paths refer to existing
immutable shared caches; Libri100 preparation records every reused shard hash.
No existing trained model or historical source snapshot was changed.

To repeat single-GPU checks inside a fresh GPU allocation, source
`../zapformer_s3t2_native_20261012/environment.sh`, then run its `validate.py
--output /absolute/new/output` with `$MFA_PYTHON`. Use a fresh directory to avoid
continuing an existing test. `validate_epochs.py --output /absolute/new/output`
repeats the extended checks. The multi-node PBS/worker files preserve native
rank-per-node torchrun dispatch. Acceptance checks bind results to actual source
hashes; changing training source invalidates the old formal-launch markers.

Repeatability is scoped to the tested software/hardware, configuration and
world size. Different world sizes change batches and gradient reduction; no
cross-world-size or cross-version bitwise guarantee is asserted.
