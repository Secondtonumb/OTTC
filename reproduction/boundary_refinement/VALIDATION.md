# Validation on 2026-10-13

The default path was validated **without training**. No existing OTTC checkpoint
or CE/JSD loss was modified. The earlier queued smoke script containing new-head
training (PBS3544558) was cancelled before execution. Replacement PBS3544578
completed successfully in 49 seconds; it only exported and evaluated frozen models.

13 CPU unit tests pass, both in the normal CPU environment (torch2.7.1) and in
the existing GPU-job environment (torch2.6.0, Python3.10). The real-data export
uses CUDA/K2 with the original step2920 CE and upstream-JSD models. It verifies
that cached prediction hidden states reproduce the original model logits.

## Smoke results only — not a TIMIT benchmark

Each model used the **first 4 dev and first 4 core-test manifest entries**. Dev
has 152 internal boundaries; test has 154. These samples were chosen before
scoring, without tuning the 20 ms window. They are not speaker balanced.

| Frozen model | Subset | K2 MAE ms | Refined MAE ms | K2 F1@20ms % | Refined F1@20ms % | K2 R % | Refined R % | Accepted / total |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| upstream JSD | dev4 | 19.643 | 18.347 | 59.21 | 63.16 | 65.18 | 68.55 | 104 / 152 |
| upstream JSD | test4 | 19.745 | 19.274 | 63.64 | 63.64 | 68.96 | 68.96 | 64 / 154 |
| CE | dev4 | 21.104 | 21.264 | 57.24 | 59.21 | 63.50 | 65.18 | 75 / 152 |
| CE | test4 | 20.496 | 20.599 | 63.64 | 60.39 | 68.96 | 66.19 | 67 / 154 |

This validates reuse and continuous outputs, **not a general accuracy gain**.
JSD improves MAE on these small subsets; CE slightly worsens. No hyperparameters
were changed after seeing test results. The subsequent full dev400/test192 evaluation is now complete; see the update below.
This smoke table remains a historical workflow check only.

Readout-only synchronized GPU latency was 1.33–1.62 ms/utterance on these short
samples. This includes Python/tensor validation and synchronization and excludes
both encoders and K2. It is not an end-to-end speed comparison.

Raw paired predictions, all 5/10/20 ms P/R/F1/R-value metrics, global OT and its
frame-left ablation, cache/checkpoint provenance:

`results/smoke_3544578/{jsd_upstream,ce}/eval_{dev,test}/`

The optional learned head has synthetic optimization and checkpoint-reload unit
coverage only. It has not been trained on TIMIT in this task.

## Subsequent full-set result (same day)

PBS3544607 completed both frozen models on dev400 and core-test192 (1,184
utterance/model pairs; no failures). All four K2-to-refined MAEs worsened slightly
(+0.178 to +0.281 ms). The smoke trend did not generalize. Keep the original K2
readout as default for now. Full audited results and timing components are in
`results/full_3544607/summary/report.md`; the Chinese experiment record is updated
at `/work/gm64/m64000/seamless_communication/docs/m4t/timit_ottc_cif_inventory.md`.
