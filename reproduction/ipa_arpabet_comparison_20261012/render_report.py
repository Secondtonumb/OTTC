"""Render the saved rescore, retaining protocol and input distinctions."""
import json
from pathlib import Path

OUT = Path(__file__).resolve().parent
x = json.loads((OUT / 'results.json').read_text())
rows = x['results']
models = list(dict.fromkeys(r['model'] for r in rows))


def selected(model, split, protocol, subset='official'):
    return next(r for r in rows if r['model'] == model and r['split'] == split
                and r['protocol'] == protocol and r['subset'] == subset)


def edits(r):
    return ' / '.join(f'{r[k + "_percent"]:.2f}' for k in ['substitutions', 'deletions', 'insertions'])


lines = [
    '# TIMIT phone editing errors: IPA → canonical ARPAbet / TIMIT-39',
    '',
    'Saved-prediction backfill, 2026-10-12. Completed models use their final epoch-10 checkpoints. '
    'S3T2** uses a fixed epoch-5 checkpoint (step 34436), evaluated independently while training continues. '
    'No retraining, test-based checkpoint selection, time correction or original-file modification.',
    '',
    'S = substitutions; D = deletions; I = insertions. Percentages divide by reference phone count. '
    'PER = 100 × (S+D+I)/N. These are sequence errors; boundary timing scores are not recomputed here.',
    '',
    '## Mapping-only comparison: original speech-only reference and scorer',
    '',
    'Both predictions and original closure-merged reference are projected to the same canonical 39 inventory. '
    'The original jiwer edit scorer is reused. Adjacent equal phone tokens are retained; no extra CTC collapse '
    'is applied after mapping. Standard q/ʔ deletion changes the denominator, which is recorded in the CSV.',
    '',
    '| Model | Epoch | Dev400 mapped S / D / I % | Dev mapped PER % | Test192 mapped S / D / I % | Test mapped PER % | Test original IPA PER % |',
    '|---|---:|---:|---:|---:|---:|---:|',
]
for model in models:
    a = selected(model, 'test', 'native_ids')
    b = selected(model, 'test', 'mapped_native_jiwer_english_collapsed')
    d = selected(model, 'development', 'mapped_native_jiwer_english_collapsed')
    lines.append(f'| {model} | {b["epoch"]} | {edits(d)} | {d["PER_percent"]:.2f} | {edits(b)} | {b["PER_percent"]:.2f} | {a["PER_percent"]:.2f} |')
lines += ['', '## FA-Bench raw-PHN editing protocol: common inventory and scorer', '',
          'Official dev400/core-test192; original TIMIT PHN uses the frozen FA-Bench TIMIT61_TO_39 mapping. '
          'Closure and pause labels become SIL and are retained. Glottal stop q is deleted. '
          'The exact FA-Bench edit_counts / Needleman-Wunsch scorer is reused, including its tie-break rules. '
          'Our saved audio-only native decoder removes silence/non-speech IDs, so their absence is penalized '
          'in this table. The public MFA numbers use word-transcript input; they are source references, '
          'not a local MFA reproduction. Shared inventory and scorer do not imply equal-input ranking.', '',
          '| Model / input | Epoch | Dev400 S / D / I % | Dev PER % | Test192 S / D / I % | Test PER % |',
          '|---|---:|---:|---:|---:|---:|']
for model in models:
    a = selected(model, 'development', 'raw_phn_fabench_english_collapsed')
    b = selected(model, 'test', 'raw_phn_fabench_english_collapsed')
    lines.append(f'| {model} / audio only | {b["epoch"]} | {edits(a)} | {a["PER_percent"]:.2f} | {edits(b)} | {b["PER_percent"]:.2f} |')
a, b = x['published_mfa_reference']
lines.append(f'| MFA 3.4 published / words | — | {edits(a)} | {a["PER_percent"]:.1f} | {edits(b)} | {b["PER_percent"]:.1f} |')
lines += ['', '[Public MFA source](https://github.com/olewave/fa-bench/blob/main/records/202609/en/gold/phone/timit/Details.md#timit-phone-editing-errors-by-condition). '
          'Public values are rounded to one decimal; their sums may differ by 0.1.', '',
          '## Mapping sensitivity and unsupported predictions', '',
          'Primary mapping reuses FA-Bench IPA_TO_39 and declares MFA aliases/representatives '
          '(e.g. aj→AY, ej→EY, ʉː→UW), then collapses listed allophonic variants '
          '(e.g. aspirated/palatalized stops) to English ARPAbet categories. '
          'The alternate aliases-only mapping retains those variants as unsupported tokens. '
          'Neither mapping is fitted to errors or selected using test performance. Unsupported phones '
          '(e.g. c, ç, ɟ, ɲ, ʎ) remain never-matching symbols and are counted; none are silently discarded.', '',
          '| Model | Native-reference test PER: aliases only % | Native-reference test PER: English collapsed % |',
          '|---|---:|---:|']
for model in models:
    a = selected(model, 'test', 'mapped_native_jiwer_aliases_only')
    b = selected(model, 'test', 'mapped_native_jiwer_english_collapsed')
    lines.append(f'| {model} | {a["PER_percent"]:.2f} | {b["PER_percent"]:.2f} |')
lines += ['', '## Verification and files', '',
          '- Native IPA S/D/I counts reproduce every saved evaluation exactly before remapping.',
          '- Exact ID coverage is asserted for dev4096/test192 and official dev400/core-test192.',
          '- Per-utterance accounting verifies matches+substitutions+deletions=Nref and matches+substitutions+insertions=Nhyp.',
          '- Source SHA256 values include the original PHN files, prediction JSONL files, source metrics, mapping/scorer code and split lists.',
          '- Full dev4096/official dev400/test192 matrices and both mapping variants are preserved; no interpolation.',
          '', '[Matrix CSV](matrix.csv) · [Result JSON](results.json) · [Mapping](mapping.json) · '
          '[Per-utterance JSONL](per_utterance.jsonl) · [Mapping usage](mapping_usage.csv) · [Source hashes](source_manifest.json)', '']
(OUT / 'README.md').write_text('\n'.join(lines))
print(OUT / 'README.md')
