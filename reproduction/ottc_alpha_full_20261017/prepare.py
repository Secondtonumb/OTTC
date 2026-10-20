"""Freeze full test-set audio-only inputs; keep all scoring references separate."""
import hashlib
import json
from pathlib import Path

import soundfile as sf

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    tmp.replace(path)


def main():
    specs = [
        ('timit', 'timit_core_test_192', 192,
         'reproduction/results/runexp_ottc_fuse_phoneme_jsd/timit_core_test_192.jsonl',
         'reproduction/timit_singlehead_k2_fa_20261013/full/jsd_upstream/metrics.json'),
        ('buckeye', 'buckeye_test', 1840,
         'reproduction/buckeye_singlehead_20261014/evaluation/jsd_upstream/test_predictions.jsonl',
         'reproduction/buckeye_singlehead_20261014/evaluation/jsd_upstream/metrics.json'),
        ('libri100', 'libri_test_clean', 2620,
         'reproduction/libri100_singlehead_20261014/evaluation/jsd_upstream/test_clean_predictions.jsonl',
         'reproduction/libri100_singlehead_20261014/evaluation/jsd_upstream/metrics.json'),
        ('libri100', 'libri_test_other', 2938,
         'reproduction/libri100_singlehead_20261014/evaluation/jsd_upstream/test_other_predictions.jsonl',
         'reproduction/libri100_singlehead_20261014/evaluation/jsd_upstream/metrics.json'),
    ]
    jobs = {}
    sources = {}
    references = {}
    for dataset, split, expected, source_name, meta_name in specs:
        source, meta = ROOT / source_name, ROOT / meta_name
        metadata = json.loads(meta.read_text())
        sources[str(source)] = digest(source)
        sources[str(meta)] = digest(meta)
        job = jobs.setdefault(dataset, dict(dataset=dataset, checkpoint=metadata['checkpoint'],
            checkpoint_sha256=metadata['checkpoint_sha256'], cases=[]))
        rows = [json.loads(line) for line in source.open()]
        assert len(rows) == expected
        for row in rows:
            utt = Path(row['utt']).stem
            rid = split + '__' + utt
            assert rid not in references
            duration = row.get('duration')
            if duration is None:
                duration = sf.info(row['wav']).duration
            if dataset == 'timit':
                prior = dict(hypothesis=row['hypothesis'], spans=row['predicted_spans'],
                             boundaries=row['predicted_boundaries'])
            else:
                prior = row['text_free']
            job['cases'].append(dict(recording_id=rid, utt=utt, split=split,
                wav=row['wav'], duration=duration, prior_audio_only_prediction=prior))
            references[rid] = dict(split=split, duration=duration, reference=row['reference'],
                saved_reference_boundaries=row.get('reference_boundaries'),
                reference_kind='existing automatic alignment; model identity unverified' if dataset == 'libri100' else 'human phones')
        print('PREPARED', split, len(rows), 'max seconds', max(r['duration'] for r in references.values() if r['split'] == split), flush=True)
    assert len(references) == 7590
    # Longest, shortest and median inputs are processed first as a bounded exact-recipe probe.
    for job in jobs.values():
        ordered = sorted(job['cases'], key=lambda c: c['duration'])
        probe = [ordered[-1], ordered[0], ordered[len(ordered) // 2]]
        probe_ids = {c['recording_id'] for c in probe}
        job['cases'] = probe + [c for c in job['cases'] if c['recording_id'] not in probe_ids]
    inputs = dict(scope='complete fixed test sets, 7590 utterances, three native single-head upstream-JSD checkpoints',
        expected_utterances=7590, expected_by_split={s: n for _, s, n, _, _ in specs}, jobs=list(jobs.values()))
    if (HERE / 'DECODE_CONTRACT.json').exists():
        assert json.loads((HERE / 'INPUTS.json').read_text()) == inputs, 'Cannot alter frozen active inputs'
        return
    write(HERE / 'INPUTS.json', inputs)
    write(HERE / 'SCORING_REFERENCES.json', references)
    write(HERE / 'PREPARE_COMPLETE.json', dict(utterances=len(references), source_sha256=sources,
        input_sha256=digest(HERE / 'INPUTS.json'), references_sha256=digest(HERE / 'SCORING_REFERENCES.json'),
        no_reference_fields_in_decoder_inputs=True))


if __name__ == '__main__':
    main()
