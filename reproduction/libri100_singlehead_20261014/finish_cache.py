"""Finish interrupted cache creation from already audited manifests."""
import bootstrap
from bootstrap import HERE
import json
from pathlib import Path
import datasets
from prepare import digest, generate_audio

output=HERE/'data'
report=json.loads((output/'audit.json').read_text())
for split, record in report['splits'].items():
    assert record['source_manifest_sha256']==digest(record['source_manifest'])
    assert record['prepared_manifest_sha256']==digest(output/(split+'.jsonl'))
assert report['speaker_utterance_source_disjointness_verified']
features=datasets.Features({'input_values':datasets.Sequence(datasets.Value('float32')),
    'labels':datasets.Sequence(datasets.Value('int64')), 'length':datasets.Value('int64')})
for split in ['train','dev']:
    destination=output/('cache_'+split)
    arrow_path=output/(split+'_normalized.arrow')
    if (destination/'state.json').exists():
        ds=datasets.load_from_disk(destination)
    else:
        ds=None
        if arrow_path.exists():
            try:
                cached=datasets.Dataset.from_file(str(arrow_path))
                if len(cached)==report['splits'][split]['utterances']:ds=cached
            except (OSError,ValueError):pass
        if ds is None:
            from datasets.arrow_writer import ArrowWriter
            with ArrowWriter(path=str(arrow_path),features=features,writer_batch_size=32) as writer:
                for i,row in enumerate(generate_audio(output/(split+'.jsonl'),output/'processor')):
                    writer.write(row)
                    if (i+1)%1000==0:print('CACHE',split,i+1,flush=True)
                writer.finalize()
            ds=datasets.Dataset.from_file(str(arrow_path))
        ds.save_to_disk(destination)
    assert len(ds)==report['splits'][split]['utterances']
    assert set(ds.column_names)=={'input_values','labels','length'}
    report['splits'][split]['training_cache_columns']=ds.column_names
    print('CACHE_READY',split,len(ds),flush=True)
(output/'PREPARE_COMPLETE.json').write_text(json.dumps(report,indent=2)+'\n')
print('PREPARE_COMPLETE',flush=True)
