"""Exhaustive source-TSV timestamp and PCM validation, independent of the converter."""
import csv
import hashlib
import itertools
import json
import math
import wave
from collections import Counter, defaultdict
from decimal import Decimal
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DATA = ROOT / 'reproduction/data/buckeye_full40_20261022'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        while b := f.read(8*1024*1024):
            h.update(b)
    return h.hexdigest()


def main():
    audit = json.loads((DATA / 'audit.json').read_text())
    for path, expected in audit['source_sha256s'].items():
        assert sha(path) == expected, ('Converter source changed since preparation', path)
    vocab = json.loads((DATA / 'vocab.json').read_text())
    assert vocab == json.loads((ROOT / 'reproduction/buckeye_singlehead_20261014/data/vocab.json').read_text())
    assert (DATA/'vocab.json').read_bytes() == (ROOT/'reproduction/buckeye_singlehead_20261014/data/vocab.json').read_bytes()
    mapping = json.loads((DATA / 'inventory_mapping.json').read_text())['raw_labels']
    tracks = defaultdict(list)
    speakers, sources, seen = {}, {}, set()
    ctc = Counter()
    intervals = 0
    manifest_hashes = {}
    for split in ['train', 'dev', 'test']:
        path = DATA / f'{split}.jsonl'
        manifest_hashes[split] = sha(path)
        assert manifest_hashes[split] == audit['splits'][split]['prepared_manifest_sha256']
        speakers[split], sources[split] = set(), set()
        legacy = json.loads((DATA/'json'/f'{split}.json').read_text())
        rows = 0
        with path.open() as f:
            for line in f:
                r = json.loads(line)
                assert r['utt'] not in seen
                seen.add(r['utt'])
                speakers[split].add(r['speaker'])
                sources[split].add(r['source_utt'])
                assert r['split'] == split and r['dataset_id'] == audit['dataset_id']
                assert r['samples'] == r['source_end_sample'] - r['source_start_sample']
                assert r['duration'] == r['samples']/16000
                assert math.isfinite(r['duration'])
                reference = r['reference']
                assert reference and len(reference) == len(r['reference_native']) == len(r['reference_source_rows'])
                assert all(p in vocab and p not in ['sil','<unk>','<pad>'] and 0 <= a < b <= r['duration']
                           for p,a,b in reference)
                assert all(a[2] <= b[1]+1e-12 for a,b in zip(reference,reference[1:]))
                targets = ['sil'] + [p for p,_,_ in reference] + ['sil']
                targets = [k for k,_ in itertools.groupby(targets)]
                assert r['training_phones'] == targets
                assert r['training_target_ids'] == [vocab[p] for p in targets]
                old = legacy[r['wav']]
                assert old['phonemes'] == [dict(phoneme=p,start=a,end=b) for p,a,b in reference]
                assert old['source_utt'] == r['source_utt'] and old['duration'] == r['duration']
                repeated = sum(a[0] == b[0] for a,b in zip(reference,reference[1:]))
                for stride in [160,320]:
                    frames = (r['samples'] - 400)//stride + 1
                    ctc[f'infeasible_{stride//16}ms'] += len(reference)+repeated > frames
                intervals += len(reference)
                tracks[r['source_utt']].append(r)
                rows += 1
        assert rows == audit['splits'][split]['utterances']
    for a,b in itertools.combinations(speakers,2):
        assert not speakers[a]&speakers[b] and not sources[a]&sources[b]
    assert len(set.union(*speakers.values())) == 40 and len(tracks) == 255
    pcm_fingerprint = hashlib.sha256()
    max_error = Decimal(0)
    source_rows_seen = 0
    source_labels = Counter()
    with (DATA/'raw/analysis/phones_timestamps.tsv').open(newline='') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for track, group in itertools.groupby(reader, lambda r:r['recording_id']):
            tokens = {}
            for p in group:
                source_rows_seen += 1
                source_labels[p['token']] += 1
                tokens[int(p['token_index'])] = p
            rows = tracks[track]
            with wave.open(rows[0]['source_wav'], 'rb') as source:
                assert source.getframerate() == 16000 and source.getnchannels() == 1 and source.getsampwidth() == 2
                for r in rows:
                    start = Decimal(r['source_start_sample'])/16000
                    end = Decimal(r['source_end_sample'])/16000
                    assert float(start) == r['source_wav_start'] and float(end) == r['source_wav_end']
                    for (phone,a,b),(raw,na,nb),index in zip(r['reference'],r['reference_native'],r['reference_source_rows']):
                        p = tokens[index]
                        assert raw == p['token'] and (a,b) == (na,nb)
                        assert mapping[raw]['action'] == 'speech' and mapping[raw]['target'] == phone
                        expected_start = max(Decimal(p['start_s']), start)
                        expected_end = min(Decimal(p['end_s']), end)
                        error = max(abs(Decimal(str(a))+start-expected_start),
                                    abs(Decimal(str(b))+start-expected_end))
                        max_error = max(max_error,error)
                        assert error <= Decimal('0.0000000001'), (r['utt'], index, error)
                    source.setpos(r['source_start_sample'])
                    expected_pcm = source.readframes(r['samples'])
                    with wave.open(r['wav'],'rb') as clip:
                        assert clip.getnframes() == r['samples'] and clip.getframerate() == 16000
                        assert clip.getnchannels() == 1 and clip.getsampwidth() == 2
                        pcm = clip.readframes(r['samples'])
                    assert pcm == expected_pcm, r['wav']
                    pcm_fingerprint.update(r['utt_id'].encode())
                    pcm_fingerprint.update(hashlib.sha256(pcm).digest())
            if len(tracks) and int(track[1:3])%5 == 0:
                print('validated',track,flush=True)
    assert source_rows_seen == 927302
    assert dict(source_labels) == {k:v['count'] for k,v in mapping.items()}
    hashes = {str(p.relative_to(DATA)):sha(p) for p in [DATA/'audit.json',DATA/'vocab.json',
            DATA/'inventory_mapping.json',DATA/'annotation_issues.jsonl',DATA/'excluded_utterances.jsonl',
            DATA/'historical_compatibility.json',DATA/'all.jsonl',DATA/'legacy_test_speakers.jsonl']}
    report = dict(dataset_id=audit['dataset_id'], validated=True,
        source_sha256=audit['source_sha256'], source_tracks=255, source_speakers=40,
        validated_utterances=len(seen), validated_reference_intervals=intervals,
        source_phone_rows_checked=source_rows_seen,
        max_timestamp_reconstruction_error_seconds=str(max_error),
        all_pcm_slices_byte_identical_to_source=True, pcm_collection_sha256=pcm_fingerprint.hexdigest(),
        manifests_sha256=manifest_hashes, metadata_sha256=hashes,
        vocab_byte_identical_to_historical=True, split_speaker_and_track_disjoint=True,
        ctc_reference_diagnostics=dict(ctc),
        validation_source_sha256=sha(Path(__file__)), evaluation_uploads=False)
    path = DATA/'PREPARE_COMPLETE.json'
    temp = path.with_suffix('.json.tmp')
    temp.write_text(json.dumps(report,indent=2)+'\n')
    temp.replace(path)
    print('PREPARE_COMPLETE',json.dumps(report),flush=True)


if __name__ == '__main__':
    main()
