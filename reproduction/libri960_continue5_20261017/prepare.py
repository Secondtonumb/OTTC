"""Compact full Libri960 phone supervision; normalize audio lazily on retrieval."""
import bootstrap
from bootstrap import ROOT, HERE
from collections import Counter
import hashlib
import itertools
import json
from pathlib import Path
import datasets
from datasets.arrow_writer import ArrowWriter

SOURCE=ROOT/'reproduction/multihead_libri960/data'
PARENT=ROOT/'reproduction/libri100_singlehead_20261014'


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def main():
    output=HERE/'data';output.mkdir(parents=True,exist_ok=True)
    if (output/'PREPARE_COMPLETE.json').exists():
        previous=json.loads((output/'PREPARE_COMPLETE.json').read_text())
        assert previous['source_sha256']==digest(SOURCE/'train-train.jsonl')
        assert previous['metadata_sha256']==digest(SOURCE/'metadata.json')
        assert previous['compact_sha256']==digest(output/'train_metadata.arrow')
        print('Validated prepared Libri960 metadata reused',flush=True);return
    metadata=json.loads((SOURCE/'metadata.json').read_text())
    assert not metadata['speaker_overlap']
    vocab=json.loads((PARENT/'data/vocab.json').read_text())
    assert vocab==json.loads((SOURCE/'vocabs.json').read_text())['ottc_arpabet']
    for name in ['vocab.json','processor','cache_dev','dev.jsonl','test.jsonl',
                 'dev_clean.jsonl','dev_other.jsonl','test_clean.jsonl','test_other.jsonl',
                 'buckeye_dev.jsonl','buckeye_test.jsonl']:
        target=output/name
        if not target.exists():target.symlink_to(PARENT/'data'/name,target_is_directory=(PARENT/'data'/name).is_dir())
    features=datasets.Features({'utt':datasets.Value('string'),'wav':datasets.Value('string'),
        'labels':datasets.Sequence(datasets.Value('int64')),'length':datasets.Value('int64'),
        'duration':datasets.Value('float64')})
    counts=Counter();seconds=Counter();speakers=set();ids=set();longest=[]
    temporary=output/'train_metadata.arrow.pending'
    with ArrowWriter(path=str(temporary),features=features,writer_batch_size=256) as writer:
        with (SOURCE/'train-train.jsonl').open() as f:
            for index,line in enumerate(f):
                row=json.loads(line);labels=row['targets']['ottc_arpabet']
                assert labels and all(t in vocab.values() and t not in (0,41) for t in labels)
                assert all(a!=b for a,b in zip(labels,labels[1:]))
                assert row['utt'] not in ids and row['duration']>0
                ids.add(row['utt']);speakers.add(row['speaker'])
                counts[row['subset']]+=1;seconds[row['subset']]+=row['duration']
                record=dict(utt=row['utt'],wav=row['wav'],labels=labels,
                    length=round(row['duration']*16000),duration=row['duration'])
                writer.write(record)
                longest.append((row['duration'],index,row['utt']))
                if (index+1)%25000==0:print('COMPACT_MANIFEST',index+1,flush=True)
        writer.finalize()
    assert counts==Counter({s:m['utterances'] for s,m in metadata['splits']['train-train']['subsets'].items()})
    assert len(ids)==281191 and speakers==set(metadata['splits']['train-train']['speakers'])
    for split in ['train-dev','test']:
        assert not speakers&set(metadata['splits'][split]['speakers'])
    temporary.replace(output/'train_metadata.arrow')
    report=dict(dataset='LibriSpeech full train-clean-100 + train-clean-360 + train-other-500',
        utterances=len(ids),speakers=len(speakers),subsets=dict(counts),hours=sum(seconds.values())/3600,
        subset_hours={s:v/3600 for s,v in seconds.items()},source=str(SOURCE/'train-train.jsonl'),
        source_sha256=digest(SOURCE/'train-train.jsonl'),metadata_sha256=digest(SOURCE/'metadata.json'),
        compact_sha256=digest(output/'train_metadata.arrow'),phone_inventory_sha256=digest(output/'vocab.json'),
        training_columns=list(features),training_timestamps=False,audio='full uncropped 16kHz mono; same per-utterance processor normalization, decoded lazily',
        normalization_reference=str(PARENT/'prepare.py'),
        longest64=[dict(duration=d,index=i,utt=u) for d,i,u in sorted(longest,reverse=True)[:64]],
        max_duration_seconds=max(d for d,i,u in longest),speaker_disjointness_verified=True,
        evaluation_manifest_sha256={s:digest(output/(s+'.jsonl')) for s in ['dev_clean','dev_other','test_clean','test_other','buckeye_dev','buckeye_test']})
    (output/'PREPARE_COMPLETE.json').write_text(json.dumps(report,indent=2)+'\n')
    print('PREPARE_COMPLETE',report['utterances'],report['hours'],'hours; max',report['max_duration_seconds'],flush=True)


if __name__=='__main__':main()
