"""Prepare disjoint existing human-annotated Buckeye splits without test leakage."""
import bootstrap
from bootstrap import HERE, ROOT
from collections import Counter
import hashlib
import itertools
import json
from pathlib import Path

import datasets
import numpy as np
import soundfile as sf
from transformers import Wav2Vec2Processor

SOURCE = Path('/work/gm64/m64000/IF-MDD/data/buckeye_raw_alignments')
ORIGINAL = ROOT / 'reproduction/models/large_models_results/wav2vec2-large-lv60/runexp_ottc_fuse_phoneme'


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def generate_audio(manifest, processor_directory):
    processor = Wav2Vec2Processor.from_pretrained(processor_directory)
    for line in Path(manifest).read_text().splitlines():
        row = json.loads(line)
        audio, sr = sf.read(row['wav'], dtype='float32')
        assert sr == 16000 and audio.ndim == 1
        values = processor(audio, sampling_rate=sr).input_values[0]
        assert len(values) == row['samples'] and np.isfinite(values).all()
        # The training cache intentionally contains no timestamps or test rows.
        yield {'input_values': values, 'labels': row['training_target_ids'], 'length': len(values)}


def main():
    output = HERE / 'data'
    output.mkdir(parents=True, exist_ok=True)
    processor = Wav2Vec2Processor.from_pretrained(ORIGINAL / 'processor')
    processor.save_pretrained(output / 'processor')
    vocab = json.loads((ORIGINAL / 'vocab_used.json').read_text())
    (output / 'vocab.json').write_text(json.dumps(vocab, indent=2)+'\n')
    report = {'corpus': 'Original human .phones/.words from intact tracks of a truncated archive; 19-speaker subset',
        'source': str(SOURCE), 'source_metadata_sha256': digest(SOURCE/'metadata/buckeye_raw_prepare_summary.json'),
        'full_40_speaker_buckeye': False, 'split_protocol': 'Existing seed0 speaker-disjoint 15/2/2 split',
        'phone_mapping': {'ao': 'aa'}, 'target_protocol': 'perceived_train_target including artificial edge sil; mapped consecutive duplicates merged as in original fused-phone recipe',
        'references': 'Human speech-phone intervals; ao relabelled aa without merging intervals; timestamps never training inputs',
        'splits': {}}
    speakers, utterances, sources = {}, {}, {}
    for split, expected in [('train', 10927), ('dev', 1524), ('test', 1840)]:
        path = SOURCE/'json'/f'{split}.json'
        original = json.loads(path.read_text())
        assert len(original) == expected
        speakers[split], utterances[split], sources[split] = set(), set(), set()
        durations = []
        counts = Counter()
        manifest = output / (split+'.jsonl')
        with manifest.open('w') as f:
            for key, row in sorted(original.items()):
                wav = Path(row['wav'])
                info = sf.info(wav)
                assert info.samplerate == 16000 and info.channels == 1 and info.frames >= 400
                reference = [[{'ao':'aa'}.get(p['phoneme'], p['phoneme']), float(p['start']), float(p['end'])] for p in row['phonemes']]
                assert reference and all(p in vocab and e>s>=0 and e<=info.duration+1e-4 for p,s,e in reference)
                assert all(a[1]<=b[1] for a,b in zip(reference,reference[1:]))
                targets = []
                for p in row['perceived_train_target'].split():
                    counts['ao_targets_mapped'] += p=='ao'
                    p = {'ao':'aa'}.get(p,p)
                    assert p in vocab and vocab[p] not in (0,41)
                    if not targets or p!=targets[-1]:
                        targets.append(p)
                    else:
                        counts['merged_target_duplicates'] += 1
                target_ids = [vocab[p] for p in targets]
                frames = (info.frames-400)//320+1
                min_ctc = len(reference)+sum(a[0]==b[0] for a,b in zip(reference,reference[1:]))
                counts['ctc_reference_infeasible'] += min_ctc>frames
                counts['reference_intervals'] += len(reference)
                counts['ao_references_mapped'] += sum(p['phoneme']=='ao' for p in row['phonemes'])
                speakers[split].add(row['spk_id']);utterances[split].add(key);sources[split].add(row['source_utt'])
                durations.append(info.duration)
                prepared = dict(utt=key, wav=str(wav), speaker=row['spk_id'], source_utt=row['source_utt'],
                    duration=info.duration, samples=info.frames, reference=reference,
                    training_target_ids=target_ids, training_phones=targets)
                f.write(json.dumps(prepared)+'\n')
        report['splits'][split] = dict(utterances=expected, speakers=sorted(speakers[split]), hours=sum(durations)/3600,
            max_duration_seconds=max(durations), counts=dict(counts), source_manifest=str(path),
            source_manifest_sha256=digest(path), prepared_manifest_sha256=digest(manifest))
    for a,b in itertools.combinations(speakers,2):
        assert not speakers[a]&speakers[b] and not utterances[a]&utterances[b] and not sources[a]&sources[b], (a,b)
    report['speaker_utterance_source_disjointness_verified'] = True
    (output/'audit.json').write_text(json.dumps(report, indent=2)+'\n')
    features = datasets.Features({'input_values':datasets.Sequence(datasets.Value('float32')),
        'labels':datasets.Sequence(datasets.Value('int64')), 'length':datasets.Value('int64')})
    for split in ['train','dev']:
        destination = output / ('cache_'+split)
        if destination.exists():
            ds = datasets.load_from_disk(destination)
            assert len(ds)==report['splits'][split]['utterances']
        else:
            from datasets.arrow_writer import ArrowWriter
            arrow_path = output/(split+'_normalized.arrow')
            with ArrowWriter(path=str(arrow_path), features=features, writer_batch_size=32) as writer:
                for example in generate_audio(str(output/(split+'.jsonl')), str(output/'processor')):
                    writer.write(example)
                writer.finalize()
            ds = datasets.Dataset.from_file(str(arrow_path))
            ds.save_to_disk(destination)
    (output/'PREPARE_COMPLETE.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
