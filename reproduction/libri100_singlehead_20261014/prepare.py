"""Reuse audited Libri100 targets; cache audio/labels without timestamps."""
import bootstrap
from bootstrap import ROOT, HERE
from collections import Counter
import hashlib
import itertools
import json
from pathlib import Path

import datasets
import numpy as np
import soundfile as sf
from transformers import Wav2Vec2Processor

SOURCE = ROOT/'reproduction/multihead_libri100/data'
BUCKEYE = ROOT/'reproduction/buckeye_singlehead_20261014/data'


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def generate_audio(manifest, processor_directory):
    processor = Wav2Vec2Processor.from_pretrained(processor_directory)
    with Path(manifest).open() as f:
        for line in f:
            row = json.loads(line)
            audio, sr = sf.read(row['wav'], dtype='float32')
            assert sr == 16000 and audio.ndim == 1
            values = processor(audio, sampling_rate=sr).input_values[0]
            assert len(values) == row['samples'] and np.isfinite(values).all()
            yield {'input_values': values, 'labels': row['training_target_ids'], 'length': len(values)}


def main():
    output = HERE/'data'
    output.mkdir(parents=True, exist_ok=True)
    if (output/'PREPARE_COMPLETE.json').exists():
        report = json.loads((output/'PREPARE_COMPLETE.json').read_text())
        assert report['source_metadata_sha256'] == digest(SOURCE/'metadata.json')
        for split, info in report['splits'].items():
            assert info['source_manifest_sha256'] == digest(info['source_manifest'])
            assert info['prepared_manifest_sha256'] == digest(output/(split+'.jsonl'))
        print('Existing audited data preparation reused', flush=True)
        return
    processor = Wav2Vec2Processor.from_pretrained(BUCKEYE/'processor')
    processor.save_pretrained(output/'processor')
    vocab = json.loads((BUCKEYE/'vocab.json').read_text())
    assert vocab == json.loads((SOURCE/'vocabs.json').read_text())['ottc_arpabet']
    (output/'vocab.json').write_text(json.dumps(vocab, indent=2)+'\n')
    mapping = {'ao':'aa', 'zh':'sh'}
    inverse = {v:k for k,v in vocab.items()}
    report = dict(corpus='LibriSpeech train-clean-100, complete existing 28,538-utterance manifest',
        source=str(SOURCE), source_metadata_sha256=digest(SOURCE/'metadata.json'),
        split_protocol='official speaker-disjoint train-clean-100 / dev-clean+dev-other / test-clean+test-other',
        phone_mapping=mapping, target_protocol='existing ottc_arpabet canonical targets including sil; mapped consecutive duplicates collapsed; audio and phone sequences only',
        references='Libri automatic aligned raw speech-phone intervals mapped without merging; alignment model unverified, not human ground truth. Buckeye original human partial19 unchanged.',
        splits={})
    speakers, utterances, sources = {}, {}, {}
    subset_rows = {s:[] for s in ['dev-clean','dev-other','test-clean','test-other']}
    for split, native, expected in [('train','train-train',28538), ('dev','train-dev',5567), ('test','test',5558)]:
        path = SOURCE/(native+'.jsonl')
        speakers[split], utterances[split], sources[split] = set(), set(), set()
        counts = Counter(); durations = []; subsets = Counter()
        manifest = output/(split+'.jsonl')
        with manifest.open('w') as f, path.open() as original:
            for line in original:
                row = json.loads(line)
                wav = Path(row['wav']); info = sf.info(wav)
                assert info.samplerate == 16000 and info.channels == 1 and info.frames >= 400
                assert abs(info.duration-row['duration']) < .001
                target_ids = row['targets']['ottc_arpabet']
                assert all(t in inverse and t not in (0,41) for t in target_ids)
                assert all(a != b for a,b in zip(target_ids,target_ids[1:]))
                reference = [[mapping.get(p,p),float(s),float(e)] for p,s,e in row['references']['ifmdd_arpabet'] if p not in ('sil','q') and e>s]
                assert all(p in vocab and e>s>=0 and e<=info.duration+.001 for p,s,e in reference)
                counts['empty_speech_references'] += not reference
                assert all(a[1]<=b[1] for a,b in zip(reference,reference[1:]))
                frames = (info.frames-400)//320+1
                minimum = len(reference)+sum(a[0]==b[0] for a,b in zip(reference,reference[1:]))
                counts['ctc_reference_infeasible'] += minimum>frames
                counts['reference_intervals'] += len(reference)
                assert row['utt'] not in utterances[split]
                speakers[split].add(row['speaker']); utterances[split].add(row['utt']); sources[split].add(str(wav))
                durations.append(info.duration); subsets[row['subset']] += 1
                prepared = dict(utt=row['utt'], wav=str(wav), speaker=row['speaker'], source_utt=row['utt'],
                    subset=row['subset'], duration=info.duration, samples=info.frames, reference=reference,
                    training_target_ids=target_ids, training_phones=[inverse[i] for i in target_ids])
                f.write(json.dumps(prepared)+'\n')
                if split != 'train': subset_rows[row['subset']].append(prepared)
        assert len(utterances[split]) == expected
        report['splits'][split] = dict(utterances=expected, speakers=sorted(speakers[split]),
            hours=sum(durations)/3600, max_duration_seconds=max(durations), counts=dict(counts), subsets=dict(subsets),
            source_manifest=str(path), source_manifest_sha256=digest(path), prepared_manifest_sha256=digest(manifest))
        print('MANIFEST',split,expected,report['splits'][split]['hours'],'hours',flush=True)
    for a,b in itertools.combinations(speakers,2):
        assert not speakers[a]&speakers[b] and not utterances[a]&utterances[b] and not sources[a]&sources[b], (a,b)
    for subset, rows in subset_rows.items():
        path = output/(subset.replace('-','_')+'.jsonl')
        path.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    for split in ['dev','test']:
        src = BUCKEYE/(split+'.jsonl'); dst = output/('buckeye_'+split+'.jsonl')
        dst.write_bytes(src.read_bytes())
        report.setdefault('zero_shot',{})['buckeye_'+split] = dict(utterances=sum(1 for _ in src.open()),
            source_manifest=str(src), source_sha256=digest(src), prepared_manifest_sha256=digest(dst),
            reference_source='original human partial19; identical to Buckeye-trained comparison')
    report['speaker_utterance_source_disjointness_verified'] = True
    (output/'audit.json').write_text(json.dumps(report, indent=2)+'\n')
    features = datasets.Features({'input_values':datasets.Sequence(datasets.Value('float32')),
        'labels':datasets.Sequence(datasets.Value('int64')), 'length':datasets.Value('int64')})
    for split in ['train','dev']:
        destination = output/('cache_'+split)
        arrow_path = output/(split+'_normalized.arrow')
        if (destination/'state.json').exists():
            ds = datasets.load_from_disk(destination)
            assert len(ds) == report['splits'][split]['utterances']
        else:
            ds = None
            if arrow_path.exists():
                try:
                    cached = datasets.Dataset.from_file(str(arrow_path))
                    if len(cached)==report['splits'][split]['utterances']:ds=cached
                except (OSError, ValueError):
                    pass
            if ds is None:
                from datasets.arrow_writer import ArrowWriter
                with ArrowWriter(path=str(arrow_path), features=features, writer_batch_size=32) as writer:
                    for i,example in enumerate(generate_audio(output/(split+'.jsonl'), output/'processor')):
                        writer.write(example)
                        if (i+1)%2000 == 0: print('CACHE',split,i+1,flush=True)
                    writer.finalize()
                ds = datasets.Dataset.from_file(str(arrow_path))
            else:
                print('Reuse completed normalized Arrow',split,flush=True)
            ds.save_to_disk(destination)
            assert len(ds) == report['splits'][split]['utterances']
        report['splits'][split]['training_cache_columns'] = ds.column_names
        assert set(ds.column_names) == {'input_values','labels','length'}
    (output/'PREPARE_COMPLETE.json').write_text(json.dumps(report, indent=2)+'\n')
    print('PREPARE_COMPLETE',flush=True)


if __name__ == '__main__':
    main()
