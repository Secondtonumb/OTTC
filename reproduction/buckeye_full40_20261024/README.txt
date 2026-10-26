Buckeye full40: timestamp and inventory compatibility, 2026-10-24

User-selected source:
  /work/gm64/m64000/dataset/buckeye_miyabi_2026-10-22.tar.gz
Prepared data:
  /work/gm64/m64000/OTTC/reproduction/data/buckeye_full40_20261022/
Raw speaker folders (also usable by a separately configured FA-Bench ingest):
  reproduction/data/buckeye_full40_20261022/raw/data/

For NEW OTTC evaluations:
  from reproduction.buckeye_dataset import load_buckeye_rows
  rows, vocab, contract = load_buckeye_rows('test', model_vocab=model_vocab)
  # Pass training_speakers=[...] for a known Buckeye-trained model.

Data files:
  train.jsonl / dev.jsonl / test.jsonl  historical OTTC evaluation row schema
  all.jsonl                          all accepted utterances
  json/*.json                        historical IF-MDD dictionary schema
  vocab.json / processor/             original OTTC 42-ID inventory and processor
  raw/data/                          untouched original WAV/.phones/.words/etc.
  raw/analysis/*_timestamps.tsv       complete native annotation rows
  inventory_mapping.json             every raw symbol, action, count and example
  annotation_issues.jsonl             source annotation ambiguity locations
  excluded_utterances.jsonl           every excluded candidate with reason
  historical_compatibility.json       overlap audit against the old partial19 rows
  audit.json / PREPARE_COMPLETE.json  protocol, hashes and full validation evidence

Each JSONL row contains wav, duration, samples, speaker, source_utt, reference,
training_target_ids and training_phones, as before. New provenance fields record
source_wav, integer source_start_sample/source_end_sample, reference_source_rows,
reference_native and dataset_id. The reference is a list of [phone,start,end]
in seconds relative to the exact start sample of its PCM WAV slice.

The original tier endpoint is the current interval end; its start is the previous
raw endpoint. A missing-label row still advances that endpoint. Parse LF records
without splitting embedded CR. Preserve native time precision; use floor/ceil
integer sample crop bounds and subtract that exact sample origin. No model-frame
quantization or time offset is applied. Clip intervals only where the audio
utterance begins/ends. Reference phones separated by pauses retain their gaps.
Real silence/interviewer/noise events remain distinct in the original tiers;
the default reference contains speech phones only. Artificial edge sil is only
in training targets. Adjacent equal gold phones remain separate intervals.

The compatibility inventory uses exactly the historical vocabulary IDs and the
frozen IF-MDD mapping. This includes dx->d, zh->sh, ao->aa, tq->t, nx->n,
syllabic/nasal vowel variants, and q exclusion. Full40 adds explicit j->jh and
ah+1n->ah rules. reference_native preserves each selected original raw label;
the complete native inventory, including excluded/event labels, is in the
untouched source tiers and inventory_mapping.json. The unknown label 'id' is
not guessed. Composite labels such as 'ah n' have no annotated internal split;
affected utterances are enumerated and excluded, with no equal-duration splits.
The existing 1-ms minimum reference overlap is retained and counted in audit.json.

Split membership is the existing fixed FA-Bench 24/8/8 SPEAKER split. Utterance
segmentation reuses the frozen IF-MDD word-based algorithm: >=0.35-s pauses,
0.2-s minimum, 20-s maximum target duration (a single annotated long word can
exceed it, as in the historical code). These are not the FA-Bench paper-protocol
utterance IDs or counts. All 255 tracks and 40 speakers contribute accepted
utterances. No speaker or source track crosses splits. Old Buckeye-trained
models may overlap these new evaluation speakers; pass their training speakers
to the loader. legacy_test_speakers.jsonl offers the original s21/s22 cohort
from this NEW source for explicitly labelled historical-split analyses.

Historical source snapshots, data and resume contracts were preserved. Future
evaluation selection is recorded in reproduction/dataset_defaults.json and
AGENTS.md. No new training or model inference is performed by this preparation;
no evaluation artifact is uploaded.

Reproduce from the OTTC repository root:
  python reproduction/buckeye_full40_20261024/extract.py
  python -m unittest discover -s reproduction/buckeye_full40_20261024 -p 'test_*.py' -v
  python reproduction/buckeye_full40_20261024/prepare.py
  python reproduction/buckeye_full40_20261024/validate.py
  python reproduction/buckeye_full40_20261024/activate.py

Extraction verifies the source SHA-256/gzip CRC and staged file hashes. Validation
independently checks every accepted phone against the source timestamp TSV and
every PCM slice byte-for-byte against its original WAV. A completion marker is
written only after all checks pass; the default loader checks its pinned hash.
