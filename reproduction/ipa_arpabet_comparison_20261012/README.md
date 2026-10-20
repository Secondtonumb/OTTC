# TIMIT phone editing errors: IPA → canonical ARPAbet / TIMIT-39

Saved-prediction backfill, 2026-10-12. Completed models use their final epoch-10 checkpoints. S3T2** uses a fixed epoch-5 checkpoint (step 34436), evaluated independently while training continues. No retraining, test-based checkpoint selection, time correction or original-file modification.

S = substitutions; D = deletions; I = insertions. Percentages divide by reference phone count. PER = 100 × (S+D+I)/N. These are sequence errors; boundary timing scores are not recomputed here.

## Mapping-only comparison: original speech-only reference and scorer

Both predictions and original closure-merged reference are projected to the same canonical 39 inventory. The original jiwer edit scorer is reused. Adjacent equal phone tokens are retained; no extra CTC collapse is applied after mapping. Standard q/ʔ deletion changes the denominator, which is recorded in the CSV.

| Model | Epoch | Dev400 mapped S / D / I % | Dev mapped PER % | Test192 mapped S / D / I % | Test mapped PER % | Test original IPA PER % |
|---|---:|---:|---:|---:|---:|---:|
| CTC | 10 | 20.00 / 3.39 / 4.03 | 27.42 | 21.67 / 3.27 / 4.14 | 29.08 | 41.44 |
| OTTC CE-cost | 10 | 19.34 / 1.76 / 21.80 | 42.90 | 20.40 / 1.69 / 20.89 | 42.98 | 54.96 |
| SP-OTTC CE-cost S3T2 | 10 | 19.29 / 8.72 / 3.20 | 31.22 | 21.09 / 9.51 / 3.24 | 33.83 | 45.30 |
| FiLM K4 | 10 | 18.17 / 19.02 / 1.54 | 38.73 | 19.94 / 17.98 / 1.38 | 39.31 | 48.82 |
| OTTC upstream JSD | 10 | 20.54 / 2.14 / 16.13 | 38.81 | 22.17 / 2.10 / 17.25 | 41.52 | 52.65 |
| SP-OTTC upstream JSD S3T2 | 10 | 19.00 / 19.02 / 1.51 | 39.53 | 19.99 / 19.12 / 1.58 | 40.69 | 50.49 |
| SP-OTTC upstream JSD S3T2** | 5 | 22.18 / 10.52 / 2.66 | 35.37 | 23.91 / 11.06 / 2.45 | 37.42 | 48.02 |

## FA-Bench raw-PHN editing protocol: common inventory and scorer

Official dev400/core-test192; original TIMIT PHN uses the frozen FA-Bench TIMIT61_TO_39 mapping. Closure and pause labels become SIL and are retained. Glottal stop q is deleted. The exact FA-Bench edit_counts / Needleman-Wunsch scorer is reused, including its tie-break rules. Our saved audio-only native decoder removes silence/non-speech IDs, so their absence is penalized in this table. The public MFA numbers use word-transcript input; they are source references, not a local MFA reproduction. Shared inventory and scorer do not imply equal-input ranking.

| Model / input | Epoch | Dev400 S / D / I % | Dev PER % | Test192 S / D / I % | Test PER % |
|---|---:|---:|---:|---:|---:|
| CTC / audio only | 10 | 19.16 / 19.24 / 2.76 | 41.16 | 20.57 / 19.22 / 3.02 | 42.81 |
| OTTC CE-cost / audio only | 10 | 21.03 / 15.58 / 15.20 | 51.82 | 22.38 / 15.48 / 14.51 | 52.38 |
| SP-OTTC CE-cost S3T2 / audio only | 10 | 18.10 / 23.86 / 2.26 | 44.22 | 19.58 / 24.60 / 2.47 | 46.65 |
| FiLM K4 / audio only | 10 | 16.53 / 32.77 / 1.25 | 50.55 | 17.96 / 32.04 / 1.33 | 51.34 |
| OTTC upstream JSD / audio only | 10 | 21.48 / 16.36 / 10.97 | 48.81 | 22.72 / 16.47 / 12.13 | 51.31 |
| SP-OTTC upstream JSD S3T2 / audio only | 10 | 17.06 / 32.78 / 1.24 | 51.08 | 18.18 / 32.82 / 1.33 | 52.34 |
| SP-OTTC upstream JSD S3T2** / audio only | 5 | 19.80 / 25.58 / 2.05 | 47.43 | 21.15 / 26.13 / 2.05 | 49.33 |
| MFA 3.4 published / words | — | 11.50 / 17.60 / 3.60 | 32.7 | 12.90 / 17.50 / 4.40 | 34.8 |

[Public MFA source](https://github.com/olewave/fa-bench/blob/main/records/202609/en/gold/phone/timit/Details.md#timit-phone-editing-errors-by-condition). Public values are rounded to one decimal; their sums may differ by 0.1.

## Mapping sensitivity and unsupported predictions

Primary mapping reuses FA-Bench IPA_TO_39 and declares MFA aliases/representatives (e.g. aj→AY, ej→EY, ʉː→UW), then collapses listed allophonic variants (e.g. aspirated/palatalized stops) to English ARPAbet categories. The alternate aliases-only mapping retains those variants as unsupported tokens. Neither mapping is fitted to errors or selected using test performance. Unsupported phones (e.g. c, ç, ɟ, ɲ, ʎ) remain never-matching symbols and are counted; none are silently discarded.

| Model | Native-reference test PER: aliases only % | Native-reference test PER: English collapsed % |
|---|---:|---:|
| CTC | 33.62 | 29.08 |
| OTTC CE-cost | 47.68 | 42.98 |
| SP-OTTC CE-cost S3T2 | 38.37 | 33.83 |
| FiLM K4 | 42.39 | 39.31 |
| OTTC upstream JSD | 45.93 | 41.52 |
| SP-OTTC upstream JSD S3T2 | 44.93 | 40.69 |
| SP-OTTC upstream JSD S3T2** | 41.64 | 37.42 |

## Verification and files

- Native IPA S/D/I counts reproduce every saved evaluation exactly before remapping.
- Exact ID coverage is asserted for dev4096/test192 and official dev400/core-test192.
- Per-utterance accounting verifies matches+substitutions+deletions=Nref and matches+substitutions+insertions=Nhyp.
- Source SHA256 values include the original PHN files, prediction JSONL files, source metrics, mapping/scorer code and split lists.
- Full dev4096/official dev400/test192 matrices and both mapping variants are preserved; no interpolation.

[Matrix CSV](matrix.csv) · [Result JSON](results.json) · [Mapping](mapping.json) · [Per-utterance JSONL](per_utterance.jsonl) · [Mapping usage](mapping_usage.csv) · [Source hashes](source_manifest.json)
