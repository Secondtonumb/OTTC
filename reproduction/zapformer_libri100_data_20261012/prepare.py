"""Select verified native MFA-IPA Libri100 rows without copying cached features.

Selection uses the exact historical Libri100 utterance manifest. This candidate
keeps native word-transcript IPA phones; it does not adopt mapped frame-CE labels.
Preparing this artifact does not submit training or bypass migration acceptance.
"""
import collections
import gzip
import hashlib
import json
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
SOURCE = Path('/work/gm64/m64000/dataset/english-mfa31-10ms-all-train-v2/training.json')
SELECTION = HERE.parent/'spottc_jsd_s3t2_framece_libri100_20261012/data/training.jsonl'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def main():
    meta = json.loads(SOURCE.read_text())
    entry = meta['split_manifests'][meta['train_split']]
    source_rows = SOURCE.parent/entry['path']
    assert sha(source_rows) == entry['sha256']
    with SELECTION.open() as stream:
        wanted = {json.loads(line)['id'] for line in stream}
    assert len(wanted) == 28538
    feature_meta = {str((SOURCE.parent/item['path']).resolve()): item for item in meta['features']}
    seen = collections.Counter(); phones = collections.Counter(); speakers = set(); used = {}
    seconds = 0.; frames = 0; scanned = 0
    output = HERE/'train.jsonl.gz'; temporary = output.with_name(output.name+'.partial')
    with temporary.open('wb') as raw, gzip.GzipFile(filename='',mode='wb',fileobj=raw,mtime=0) as encoded:
        with gzip.open(source_rows,'rt') as stream:
            for line in stream:
                row = json.loads(line); scanned += 1
                ident = row['recording_id'].removeprefix('librispeech:')
                if row['corpus'] != 'librispeech' or ident not in wanted:
                    continue
                assert row['previous_internal_split'] == 'train'
                assert row['frames'] > 0 and row['tokens'] and all(3 <= t < 105 for t in row['tokens'])
                path = Path(row['feature_file'])
                if not path.is_absolute(): path = SOURCE.parent/path
                path = path.resolve(); info = feature_meta[str(path)]
                assert 0 <= row['feature_offset'] and (row['feature_offset']+row['frames']*80)*4 <= info['bytes']
                used[str(path)] = info
                row['feature_file'] = str(path)
                row['official_libri100_utterance_id'] = ident
                encoded.write((json.dumps(row,ensure_ascii=False,separators=(',',':'))+'\n').encode())
                seen[ident] += 1; speakers.add(row['speaker']); phones.update(row['tokens'])
                seconds += row['duration']; frames += row['frames']
    assert scanned == entry['count'] and seen.keys() == wanted and all(n == 1 for n in seen.values())
    features = []
    for path, info in sorted(used.items()):
        assert Path(path).stat().st_size == info['bytes'] and sha(path) == info['sha256']
        features.append(dict(path=path,sha256=info['sha256'],bytes=info['bytes']))
    os.replace(temporary,output)
    payload = dict(format=meta['format'],training_ready=True,train_split='train',
                   split_manifests=dict(train=dict(path=output.name,sha256=sha(output),count=len(seen),seconds=seconds)),
                   feature_config=meta['feature_config'],features=features,
                   dataset='LibriSpeech train-clean-100, exact historical accepted subset',
                   speaker_count=len(speakers),frames=frames,phone_usage=dict(phones),
                   inventory=dict(path=str(SOURCE.parent/'tokens.txt'),sha256=sha(SOURCE.parent/'tokens.txt'),size=105),
                   provenance=dict(source_manifest=str(SOURCE),source_manifest_sha256=sha(SOURCE),
                       source_rows_manifest=str(source_rows),source_rows_sha256=entry['sha256'],
                       selection_manifest=str(SELECTION.resolve()),selection_sha256=sha(SELECTION),
                       preparation_source_sha256=sha(Path(__file__)),
                       target_definition='native MFA word-transcript IPA phones; not historical mapped ARPAbet frame-CE targets',
                       feature_policy='reuse immutable cached Fbank shards; no copied/recomputed features',
                       speed_perturbation=False),
                   formal_training_submitted=False,migration_acceptance_required=True)
    (HERE/'training.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(count=len(seen),hours=seconds/3600.,speaker_count=len(speakers),
                         verified_feature_shards=len(features),manifest=str(HERE/'training.json')),ensure_ascii=False),flush=True)


if __name__ == '__main__':main()
