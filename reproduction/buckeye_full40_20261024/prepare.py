"""Convert the new full Buckeye source to the historical OTTC/IF-MDD schemas.

The original tiers remain byte-for-byte intact. Ambiguous/invalid gold regions
are enumerated and excluded from the default scored utterances, never repaired.
"""
from __future__ import annotations

import csv
import hashlib
import itertools
import json
import math
import re
import shutil
import wave
from collections import Counter, defaultdict
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path

from vendor import legacy_prepare as legacy

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DATA = ROOT / 'reproduction/data/buckeye_full40_20261022'
RAW = DATA / 'raw'
OLD = ROOT / 'reproduction/buckeye_singlehead_20261014/data'
SPLIT_SOURCE = ROOT.parent / 'fa-bench/datasets/languages/en/buckeye/split'
SR = 16000
MIN_PHONE = Decimal('0.001')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        while block := f.read(8 * 1024 * 1024):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    temp.replace(path)


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    with temp.open('w') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
    temp.replace(path)


def read_tier(path):
    with Path(path).open(encoding='utf-8', newline='') as f:
        return f.read()


@dataclass
class TierRow:
    index: int
    line: int
    start: Decimal
    end: Decimal
    label: str


def parse_tier(text, track, tier):
    """LF is the record delimiter; even a missing label advances the endpoint."""
    result, issues = [], []
    previous = highwater = Decimal(0)
    in_data = False
    for line_no, line in enumerate(text.split('\n'), 1):
        if line.strip() == '#':
            in_data = True
            continue
        if not in_data or not line.strip():
            continue
        fields = line.split(None, 2)
        if len(fields) < 2:
            raise ValueError(f'Malformed endpoint: {track}:{tier}:{line_no}')
        end = Decimal(fields[0])
        label = fields[2].split(';', 1)[0].strip() if len(fields) > 2 else ''
        row = TierRow(len(result) + 1, line_no, previous, end, label)
        result.append(row)
        reasons = []
        if not label:
            reasons.append('missing_label')
        if end < previous and row.index > 1:
            reasons.append('time_reversal')
        elif previous < highwater and previous >= 0:
            reasons.append('interval_after_reversal')
        if end < 0 or previous < 0:
            reasons.append('negative_endpoint')
        for reason in reasons:
            issues.append(dict(track=track, tier=tier, row=row.index, line=line_no,
                               label=label, reason=reason,
                               start=str(min(previous, end)), end=str(max(highwater, previous, end))))
        previous = end
        highwater = max(highwater, end)
    return result, issues


MAPPING = legacy.build_phone_map(HERE / 'vendor/phones.60-48-39.map')


def classify_phone(raw_label):
    """Explicit historical inventory, with two documented full-corpus variants."""
    normalized = legacy.normalize_raw_label(raw_label)
    # The full corpus adds a nasalized stress variant absent in partial19.
    if raw_label.strip().lower() == 'ah+1n':
        normalized = 'ahn'
    if not normalized:
        return None, 'missing_label', normalized
    if len(normalized.split()) != 1:
        return None, 'composite_without_internal_gold_boundary', normalized
    if normalized == 'j':
        return 'jh', 'speech', normalized
    mapped = MAPPING.get(normalized, normalized)
    if mapped is None:
        return None, 'historical_q_exclusion', normalized
    if mapped not in legacy.TARGET_PHONES:
        return None, 'unmapped_phone', normalized
    return ('aa' if mapped == 'ao' else mapped), ('event' if mapped == 'sil' else 'speech'), normalized


def overlaps(start, end, issue):
    a, b = Decimal(issue['start']), Decimal(issue['end'])
    if a == b:
        return start < a < end
    return max(start, a) < min(end, b)


def frame_span(start, end, frame_count):
    start_frame = max(0, min(frame_count, math.floor(Decimal(str(start)) * SR)))
    end_frame = max(start_frame, min(frame_count, math.ceil(Decimal(str(end)) * SR)))
    return start_frame, end_frame


def copy_clip(source, destination, first, last):
    destination.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(source), 'rb') as f:
        params = f.getparams()
        f.setpos(first)
        pcm = f.readframes(last - first)
    assert len(pcm) == (last - first) * params.nchannels * params.sampwidth
    if destination.exists():
        with wave.open(str(destination), 'rb') as f:
            if f.getparams()[:4] != (params.nchannels, params.sampwidth, params.framerate, last - first):
                raise FileExistsError(destination)
            assert f.readframes(last - first) == pcm, destination
        return
    temp = destination.with_name(destination.name + '.tmp')
    with wave.open(str(temp), 'wb') as f:
        f.setparams(params)
        f.writeframes(pcm)
    temp.replace(destination)


def split_speakers():
    destination = DATA / 'split_sources'
    destination.mkdir(exist_ok=True)
    speaker_to_split, provenance = {}, {}
    for split in ['train', 'dev', 'test']:
        src = SPLIT_SOURCE / (split + '.list')
        shutil.copyfile(src, destination / src.name)
        speakers = {line.split()[0] for line in src.read_text().splitlines()
                    if line.strip() and not line.startswith('#')}
        assert len(speakers) == (24 if split == 'train' else 8)
        for speaker in speakers:
            assert speaker not in speaker_to_split
            speaker_to_split[speaker] = split
        provenance[split] = dict(source=str(src), sha256=sha(src), speakers=sorted(speakers))
    assert len(speaker_to_split) == 40
    return speaker_to_split, provenance


def compatibility_audit(rows):
    current = {Path(r['wav']).stem: r for r in rows}
    counts = Counter()
    changed = []
    for split in ['train', 'dev', 'test']:
        with (OLD / (split + '.jsonl')).open() as f:
            for line in f:
                old = json.loads(line)
                key = Path(old['wav']).stem
                counts['old_utterances'] += 1
                new = current.get(key)
                if new is None:
                    counts['not_in_new_scored_manifest'] += 1
                    continue
                same_phones = [x[0] for x in new['reference']] == [x[0] for x in old['reference']]
                same_samples = new['samples'] == old['samples']
                same_times = same_phones and all(abs(a[i] - b[i]) <= 1.1e-6
                    for a, b in zip(new['reference'], old['reference']) for i in [1, 2])
                counts['same_phone_sequence'] += same_phones
                counts['same_audio_sample_count'] += same_samples
                counts['same_timestamps_within_1_1us'] += same_times
                counts['same_training_target_ids'] += new['training_target_ids'] == old['training_target_ids']
                if not (same_phones and same_samples and same_times):
                    counts['changed_utterances'] += 1
                    changed.append(dict(utterance=key, same_phones=same_phones,
                                        same_samples=same_samples, same_times=same_times))
    return dict(counts=counts, changed=changed)


def main():
    extraction = json.loads((DATA / 'EXTRACT_COMPLETE.json').read_text())
    assert extraction['gzip_crc_passed'] and not extraction['missing_files']
    # Check staged annotation/analysis content before using it; WAV hashes were
    # checked during extraction and every generated PCM slice is checked later.
    for name, info in extraction['verified_files'].items():
        if Path(name).suffix in {'.phones', '.words', '.tsv'}:
            assert sha(RAW / name) == info['sha256'], name
    vocab_path = OLD / 'vocab.json'
    vocab = json.loads(vocab_path.read_text())
    shutil.copyfile(vocab_path, DATA / 'vocab.json')
    shutil.copytree(OLD / 'processor', DATA / 'processor', dirs_exist_ok=True)
    assert set(vocab.values()) == set(range(42)) and vocab['<pad>'] == 0 and vocab['<unk>'] == 41
    speaker_to_split, split_provenance = split_speakers()
    all_rows, legacy_rows, excluded, issues = [], {}, [], []
    mapping_counts, stats = Counter(), Counter()
    mapping_examples = {}
    records = sorted((RAW / 'data').glob('s*/*.wav'))
    assert len(records) == 255
    for track_number, wav in enumerate(records, 1):
        track, speaker = wav.stem, wav.parent.name
        phone_tier, phone_issues = parse_tier(read_tier(wav.with_suffix('.phones')), track, 'phones')
        word_tier, word_issues = parse_tier(read_tier(wav.with_suffix('.words')), track, 'words')
        issues.extend(phone_issues + word_issues)
        hazards = phone_issues + word_issues
        mapped = []
        for phone in phone_tier:
            target, kind, normalized = classify_phone(phone.label)
            mapping_counts[phone.label] += 1
            mapping_examples.setdefault(phone.label, dict(track=track, row=phone.index, line=phone.line))
            if kind in ['composite_without_internal_gold_boundary', 'unmapped_phone']:
                issue = dict(track=track, tier='phones', row=phone.index, line=phone.line,
                             label=phone.label, reason=kind, start=str(phone.start), end=str(phone.end))
                hazards.append(issue)
                issues.append(issue)
            if kind == 'speech' and phone.end == phone.start:
                issue = dict(track=track, tier='phones', row=phone.index, line=phone.line,
                             label=phone.label, reason='zero_duration_speech',
                             start=str(phone.start), end=str(phone.end))
                hazards.append(issue)
                issues.append(issue)
            if kind == 'speech' and phone.end > phone.start:
                mapped.append((phone, target))
        words = [legacy.Interval(row.label, float(row.start), float(row.end), row.label)
                 for row in word_tier if row.end > row.start]
        segments = legacy.build_word_segments(words, min_duration=.2, max_duration=20., silence_split_sec=.35)
        with wave.open(str(wav), 'rb') as f:
            assert f.getframerate() == SR and f.getnchannels() == 1 and f.getsampwidth() == 2
            frame_count = f.getnframes()
        for index, segment in enumerate(segments):
            stats['candidate_utterances'] += 1
            first, last = frame_span(segment.start, segment.end, frame_count)
            start, end = Decimal(first) / SR, Decimal(last) / SR
            bad = [h for h in hazards if overlaps(start, end, h)]
            if bad:
                excluded.append(dict(utterance=f'{track}_{index}', source_utt=track,
                                     speaker=speaker, split=speaker_to_split[speaker],
                                     start_sample=first, end_sample=last, reasons=bad))
                continue
            reference, native, source_indices = [], [], []
            for phone, target in mapped:
                a, b = max(phone.start, start), min(phone.end, end)
                if b - a < MIN_PHONE:
                    if b > a:
                        stats['speech_overlaps_below_historical_1ms_threshold'] += 1
                    continue
                reference.append([target, float(a - start), float(b - start)])
                native.append([phone.label, float(a - start), float(b - start)])
                source_indices.append(phone.index)
            if not reference or last - first < 400:
                excluded.append(dict(utterance=f'{track}_{index}', source_utt=track,
                                     speaker=speaker, split=speaker_to_split[speaker],
                                     start_sample=first, end_sample=last,
                                     reasons=[dict(reason='empty_or_too_short_speech_reference')]))
                continue
            assert all(p in vocab and p not in ('sil', '<pad>', '<unk>') and 0 <= a < b <= float(end-start)
                       for p, a, b in reference)
            assert all(a[2] <= b[1] + 1e-12 for a, b in zip(reference, reference[1:])), track
            path = DATA / 'audio/data_clean' / speaker / f'{track}_{index}.wav'
            copy_clip(wav, path, first, last)
            targets = ['sil'] + [p for p, _, _ in reference] + ['sil']
            # Only training targets collapse runs; gold intervals stay distinct.
            targets = [key for key, _ in itertools.groupby(targets)]
            word_rows = legacy.relative_words(segment, start=float(start), end=float(end))
            word_rows = legacy.clamp_timestamps(word_rows, duration=(last-first)/SR, min_duration=0.)
            row = dict(utt=str(path), utt_id=f'buckeye_{track}_{index:04d}', wav=str(path),
                       speaker=speaker, source_utt=track, duration=(last-first)/SR, samples=last-first,
                       reference=reference, reference_native=native, reference_source_rows=source_indices,
                       training_target_ids=[vocab[p] for p in targets], training_phones=targets,
                       words=word_rows, wrd=' '.join(w['word'] for w in word_rows),
                       source_wav=str(wav), source_wav_start=float(start), source_wav_end=float(end),
                       source_start_sample=first, source_end_sample=last,
                       dataset_id='buckeye_full40_20261022', split=speaker_to_split[speaker],
                       inventory='ottc_legacy42', artificial_edge_silence_in_reference=False)
            all_rows.append(row)
            aligned = ' '.join(['sil'] + [p for p, _, _ in reference] + ['sil'])
            legacy_rows[str(path)] = dict(wav=str(path), duration=row['duration'], spk_id=speaker,
                canonical_aligned=aligned, perceived_aligned=aligned, perceived_train_target=aligned,
                wrd=row['wrd'], subset='raw_chunk', sex='U', words=word_rows,
                phonemes=[dict(phoneme=p, start=a, end=b) for p, a, b in reference],
                source_utt=track, source_wav_start=float(start), source_wav_end=float(end))
        if track_number % 10 == 0 or track_number == len(records):
            print('prepared tracks', track_number, '/', len(records), 'utterances', len(all_rows), flush=True)
    assert len({r['speaker'] for r in all_rows}) == 40
    assert len({r['source_utt'] for r in all_rows}) == 255
    mapping_report = {}
    for raw, count in sorted(mapping_counts.items()):
        mapped, kind, normalized = classify_phone(raw)
        mapping_report[raw] = dict(count=count, normalized=normalized, target=mapped,
                                   action=kind, example=mapping_examples[raw])
    write_json(DATA / 'inventory_mapping.json', dict(
        target='Byte-identical historical 42-ID OTTC vocabulary; 40 non-special IDs including sil/q/dx',
        vocab_sha256=sha(DATA / 'vocab.json'),
        historical_rules='Frozen IF-MDD map plus overrides, then ao->aa. dx->d, zh->sh, '
                         'nx->n, tq->t, syllabic/nasal variants collapsed; q excluded.',
        full_corpus_additions={'j': 'jh', 'ah+1n': 'ah (nasal ahn with +1 annotation)'},
        ambiguity_policy='No equal-duration subdivision of composite labels; quarantine affected chunks. '
                         'Unmapped id is preserved and quarantined, with no guessed replacement.',
        raw_labels=mapping_report))
    write_jsonl(DATA / 'annotation_issues.jsonl', issues)
    write_jsonl(DATA / 'excluded_utterances.jsonl', excluded)
    splits = {}
    for split in ['train', 'dev', 'test']:
        rows = [r for r in all_rows if r['split'] == split]
        write_jsonl(DATA / f'{split}.jsonl', rows)
        write_json(DATA / 'json' / f'{split}.json', {r['wav']: legacy_rows[r['wav']] for r in rows})
        splits[split] = dict(utterances=len(rows), speakers=sorted({r['speaker'] for r in rows}),
                            tracks=len({r['source_utt'] for r in rows}),
                            hours=sum(r['duration'] for r in rows)/3600,
                            reference_intervals=sum(len(r['reference']) for r in rows),
                            prepared_manifest_sha256=sha(DATA / f'{split}.jsonl'))
    write_json(DATA / 'json/data_clean.json', legacy_rows)
    write_json(DATA / 'json/train_dev_test.json', legacy_rows)
    write_jsonl(DATA / 'all.jsonl', all_rows)
    write_jsonl(DATA / 'legacy_test_speakers.jsonl', [r for r in all_rows if r['speaker'] in {'s21', 's22'}])
    compatibility = compatibility_audit(all_rows)
    write_json(DATA / 'historical_compatibility.json', compatibility)
    report = dict(dataset_id='buckeye_full40_20261022', full_40_speaker_buckeye=True,
        source=extraction['source'], source_sha256=extraction['source_sha256'], raw_root=str(RAW / 'data'),
        source_utterances=255, speaker_count=40, utterances=len(all_rows), splits=splits,
        split_protocol='FA-Bench fixed 24/8/8 SPEAKER membership; IF-MDD legacy word segmentation. '
                       'These utterance IDs/counts are not FA-Bench paper-protocol utterances.',
        split_sources=split_provenance,
        segmentation=dict(min_duration=.2, max_duration=20., silence_split_sec=.35, min_phone_duration=.001),
        timestamp_protocol='Original endpoint seconds; start from immediately previous raw endpoint, '
                           'including missing-label rows. Slice floor(start*16000):ceil(end*16000), '
                           'then subtract actual integer sample start; clip only at slice edges. '
                           'No 10/20ms quantization, time shift, fabricated phone subdivisions or interval merging.',
        inventory='ottc_legacy42; exact old vocabulary IDs; gold reference retains separate adjacent intervals',
        references='Human speech phones; real annotation gaps retained; no artificial silence in reference',
        training_targets='Historical mapped phones + artificial edge sil; consecutive duplicates collapsed for targets only',
        canonical_field='Compatibility alias for perceived phone sequence; not dictionary canonical pronunciation',
        stats=stats, excluded_utterances=len(excluded),
        exclusion_reason_counts=dict(Counter(h['reason'] for r in excluded for h in r['reasons'])),
        annotation_issue_counts=dict(Counter(i['reason'] for i in issues)),
        unknown_inventory=dict((k,v) for k,v in mapping_report.items() if v['action']=='unmapped_phone'),
        speaker_utterance_source_disjointness_verified=True,
        historical_compatibility=compatibility['counts'],
        evaluation_uploads=False,
        source_sha256s={str(p):sha(p) for p in [Path(__file__), HERE/'extract.py',
                        HERE/'vendor/legacy_prepare.py', HERE/'vendor/phones.60-48-39.map']},
        completion='PREPARE_COMPLETE.json is written only after independent full validation')
    write_json(DATA / 'audit.json', report)
    print('PREPARED (validation pending)', json.dumps(splits), flush=True)


if __name__ == '__main__':
    main()
