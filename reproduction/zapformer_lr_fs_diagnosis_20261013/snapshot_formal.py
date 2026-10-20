"""Read-only backfill with matched original epochs/steps; no interpolation."""
import collections
import hashlib
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
PRIOR=HERE.parent/'zapformer_loss_diagnosis_20261013'


def main():
    destination=HERE/'formal_snapshot'
    if destination.exists():raise RuntimeError('Existing source snapshots must be preserved.')
    destination.mkdir()
    report=dict(backfill=True,interpolation=False,training_timestamps_replaced=False,
                metric_scope='Native Libri100 training; recognition/forced-boundary diagnostics on fixed TIMIT development 400 utterances. Not LibriSpeech WER.',arms={})
    for arm,info in json.loads((PRIOR/'inputs.json').read_text()).items():
        if arm=='origin':continue
        out=Path(info['output']);saved=destination/arm;saved.mkdir()
        rows={};hashes={}
        for name in ['updates.jsonl','evaluation_loss.jsonl','timit_metrics.jsonl']:
            # Retain only complete lines if a trainer is concurrently appending.
            data=(out/name).read_bytes();data=data[:data.rfind(b'\n')+1]
            (saved/name).write_bytes(data)
            rows[name]=[json.loads(l) for l in data.splitlines()]
            hashes[name]=hashlib.sha256(data).hexdigest()
        epochs=collections.defaultdict(list)
        for r in rows['updates.jsonl']:epochs[r['epoch']].append(r)
        summarized=[]
        for epoch,records in sorted(epochs.items()):
            if len(records)!=106:continue
            utterances=sum(r['utterances'] for r in records)
            summarized.append(dict(epoch=epoch,last_step=records[-1]['optimizer_step'],
                              ottc_per_utterance=sum(r.get('ottc_sum',r.get('jsd_sum',0)) for r in records)/utterances,
                              fs_mean_across_ranks_and_steps=sum(r['forward_sum_raw'] for r in records)/8/len(records),
                              first_lr=records[0]['lr'],last_lr=records[-1]['lr']))
        metrics=[r for r in rows['timit_metrics.jsonl'] if r['split']=='development' and 'checkpoint' not in r]
        report['arms'][arm]=dict(source=str(out),snapshot_hashes=hashes,
                                 last_update=rows['updates.jsonl'][-1],epochs=summarized,metrics=metrics)
    common=set.intersection(*(set(r['epoch'] for r in a['epochs']) for a in report['arms'].values()))
    epoch=max(common)
    report['matched_epoch']=epoch
    report['matched_epoch_values']={arm:next(r for r in a['epochs'] if r['epoch']==epoch)
                                   for arm,a in report['arms'].items()}
    common_steps=set.intersection(*(set(r['optimizer_step'] for r in a['metrics']) for a in report['arms'].values()))
    step=max(common_steps)
    report['matched_metric_step']=step
    report['matched_development_metrics']={arm:next(r for r in a['metrics'] if r['optimizer_step']==step)
                                          for arm,a in report['arms'].items()}
    (HERE/'formal_snapshot.json').write_text(json.dumps(report,indent=2)+'\n')
    print('Matched epoch',epoch,'matched metric step',step)
    for arm,value in report['matched_epoch_values'].items():print(arm,value)


if __name__=='__main__':main()
