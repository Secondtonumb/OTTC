"""Rescore retained audio-only HuPER output on the seed-sweep reference."""
import json
from pathlib import Path
from reproduction.timit_external_supplement_20261022.evaluate import HERE,ROOT,helpers,write,sha
from reproduction.cif_timit_20261016.data import load_rows
from reproduction.timit_seed_sweep_20261022.pfer import score_predictions
h=helpers();rows,vocab,data=load_rows();out=HERE/'huper_common';out.mkdir(exist_ok=True)
for split in ['dev','test']:
 source=HERE/'huper'/f'{split}.predictions.jsonl';source_done=json.loads((HERE/'huper'/f'{split}.COMPLETE.json').read_text());assert sha(source)==source_done['predictions_sha256']
 ref={r['utt']:r for r in rows[split]};c=h['new_counts']();f=out/f'{split}.predictions.jsonl'
 with f.open('w') as stream:
  for line in source.open():
   r=json.loads(line);r['reference']=ref.pop(r['utt'])['references']['ifmdd_arpabet'];c['utterances']+=1
   c['refs'].append(' '.join('d' if p=='dx' else p for p,_,_ in r['reference'] if p not in ['sil','q']))
   c['hyps'].append(' '.join('d' if p=='dx' else p for p in r['text_free']['hypothesis'] if p not in ['sil','q']))
   bounds=[s for _,s,_ in r['text_free']['predicted_spans'][1:]];h['score'](c,r['reference'],bounds,r['duration']);stream.write(json.dumps(r)+'\n')
 assert not ref
 result=dict(model='HuPER',split=split,utterances=c['utterances'],modes={'text_free':h['summary'](c)},protocol=dict(reference='identical ifmdd_arpabet reference as timit_seed_sweep_20261022',boundary='unique reference endpoints; exclude edges20ms; physical20ms next-phone-onset predictions; chronological one-to-one matching',new_inference=False,reference_times_used_by_aligner=False))
 result['pfer']=score_predictions(f,result);result['modes']['text_free']['PFER_percent']=result['pfer']['PFER_percent'];write(out/f'{split}.json',result);write(out/f'{split}.COMPLETE.json',dict(utterances=c['utterances'],predictions_sha256=sha(f),metrics_sha256=sha(out/f'{split}.json'),source_predictions_sha256=sha(source)))
write(out/'provenance.json',dict(data=data,source_inference=str(HERE/'huper'),reference='ifmdd_arpabet',new_inference=False,source_sha256=sha(Path(__file__))))
write(out/'COMPLETE.json',dict(status='complete',dev=400,test=192,source_inference_complete=True));print('COMPLETE HuPER common-reference rescoring')
