# IF-MDD SPOTTC on native Zapformer base: Libri100 four arms

User explicitly requested IF-MDD standard JSD / CE × graph forward-sum on 2026-10-13. This suite uses IF-MDD standard JSD, distinct from the unchanged upstream OTTC JSD used by the morning suite. CE means IF-MDD OTTC transport cross-entropy, not frame CE or ordinary CTC.

| Arm | OTTC cost | S3T2** graph FS weight |
|---|---|---:|
| jsd_fs | unchanged IF-MDD `utils/losses/ot_loss.py` standard JSD | 1 |
| jsd_only | same standard JSD | 0 |
| ce_fs | unchanged IF-MDD CE transport | 1 |
| ce_only | same CE transport | 0 |

Native MFA IPA105 sequence labels are retained; sorted IF-MDD inventory has 303 output states (blank, 100 three-state phones, sil/spn single-state). All arms use identical seed42 initialization. Morning313-output checkpoints are incompatible and are preserved. This is a new 50epoch comparison, without the morning10epoch warm start. Earlier30epoch oneGPU CE has different world size/auxiliary configuration and remains independently authorized.

Encoder: frozen native Zapformer base source, 80-dimensional cached Fbank at10ms, frame-preserving frontend, BF16, native SpecAugment, BatchedRubik .02 beta1 .99, original relative learning rates/scheduler and encoder auxiliary scale .05 (.25 multiplier after step2000). 8GPU true DDP per arm; one process/GPU/node. Full model and cumulative float64 average checkpoints; retain last8. Independent local/W&B runs, saved IDs reused across4hour continuations; online default with offline fallback.

OTTC reproduces original IF-MDD target expansion, repeated phones, state-uniform target mass, learned frame mass, epsilon1e-6, logit sanitization/clamp and float-relative length floor. Only the vendor topology cost selector is exposed; original sources are unchanged and hashed in `source_manifest.json`.

FS calls original IF-MDD transcript `_forward_k2` with S3T2** and output_beam10. It uses raw per-utterance graph NLL divided by effective input frames, summed across the batch, weight1. This explicit reduction adaptation matches the morning input-frame-normalized FS scale; original IF-MDD default phone-normalized mean is not used. Blank appears at phone boundaries; no content/fused loss heads. Recognition uses audio-only S3T2** phone-loop shortest path, recorded separately from original greedy PER. Supplied-phone alignment is separate from text-free results.

Before formal training, `benchmark_onegpu.pbs` checks CPU/GPU criterion parity and measures short/median/longest/largest-padded real Libri100 batches at320/640 seconds per rank, with full encoder backward, BF16, SpecAugment and native optimizer. Two warmup and three measured updates per shape. `select_batch.py` requires15% reserved-memory headroom for every arm and chooses one common duration based on worst-arm throughput. An OOM excludes that candidate. If neither candidate fits, training remains blocked. SingleGPU benchmark excludes DDP communication, checkpoint/evaluation and parameter plotting.

Then `validate_8gpu.pbs` tests each full-model arm for6 actual DDP updates, compared bitwise to a3+3 restart: model, average, optimizer, scheduler, all-rank scaler/RNG/sampler, native statistics and raw loss logs. Actual DDP per-step throughput and maximum-rank peak memory are recorded in `throughput.jsonl`; require10% headroom. `ALL_ARMS_PASS.json` requires common full-model initialization. Formal jobs have PBS afterok dependencies and startup checks; no training starts before both GPU stages pass.

Data: `../zapformer_libri100_data_20261012/training.json` (28538 utterances,100.5865h). Current evaluation monitor is historical TIMIT development400/core-test192, explicitly recorded; no Libri held-out PER/WER claim. Evaluation reports greedy/FSA PER and boundary P/R/F1/R-value, with supplied-phone FA separate, retaining scorer/R-value hashes. Different raw OTTC costs are not directly comparable.

Outputs/checkpoints: `models/<arm>/step-XXXXXXXXX.pt`; logs/config/W&B IDs alongside. GPU benchmark outputs: `throughput/<arm>/{320,640}/SUMMARY.json`. DDP acceptance: `validation/<arm>/{uninterrupted,resumed}`. Submission IDs/dependencies are stored in `jobs.json`.
