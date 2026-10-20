"""Freeze four preselected utterances and matching existing CIF checkpoints."""
import hashlib,json,os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent
VIZ=ROOT/'reproduction/segmentation_figures_20261017'

def main():
 examples=json.loads((VIZ/'examples.json').read_text())['examples']
 manifest=json.loads((VIZ/'timestamp_manifest.json').read_text())
 timit=json.loads((ROOT/'reproduction/cif_timit_20261016/active_runs.json').read_text())
 english=json.loads((ROOT/'reproduction/cif_quantity_20261016/active_runs.json').read_text())
 vocab_path=ROOT/'reproduction/cif_timit_20261016/data.py'
 # Read the frozen vocabulary location directly; preparation needs no torch.
 vocab_file=ROOT/'reproduction/models/large_models_results/wav2vec2-large-lv60/runexp_ottc_fuse_phoneme/processor/vocab.json'
 vocab=json.loads(vocab_file.read_text())
 variants=[('CIF CE+Quantity','ce','no_ctc'),('CIF + CTC','ctc_0p5','ctc_0p5'),('CIF + OTTC','ottc_jsd_0p5','ottc_jsd_0p5')]
 inputs=[];jobs=[];sources={}
 for rid,ex in examples.items():
  split=ex['split'];ds='timit' if split=='timit' else 'buckeye' if split=='buckeye' else 'libri100'
  path=ROOT/('reproduction/multihead_timit/data/test.jsonl' if ds=='timit' else 'reproduction/buckeye_singlehead_20261014/data/test.jsonl' if ds=='buckeye' else 'reproduction/multihead_libri100/data/test.jsonl')
  rows=[json.loads(s) for s in path.read_text().splitlines()]
  row=next(r for r in rows if Path(r['utt']).stem==ex['original_id'])
  native=row['reference'] if ds=='buckeye' else row['references']['ottc_arpabet']
  # Libri CIF targets retain edge silences; existing common FA targets omit
  # them. Match the supplied phone sequence without cropping or time input.
  trimmed=list(native)
  if ds=='libri100':
   if trimmed[0][0]=='sil' and ex['reference']['phones'][0]!='sil':trimmed=trimmed[1:]
   if trimmed[-1][0]=='sil' and ex['reference']['phones'][-1]!='sil':trimmed=trimmed[:-1]
  assert ['d' if p=='dx' else p for p,_,_ in trimmed]==['d' if p=='dx' else p for p in ex['reference']['phones']]
  phones=[p for p,_,_ in trimmed]
  inputs.append(dict(recording_id=rid,utt=ex['original_id'],dataset=ds,split=split,
    wav=manifest[rid]['wav'],duration=ex['duration'],reference=ex['reference'],supplied_phones=phones,
    supplied_token_ids=[vocab[p] for p in phones],native_training_target_count=len(native),
    removed_edge_sil=len(native)-len(trimmed)))
  sources[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
  for label,variant,tkey in variants:
   run=Path(timit[tkey] if ds=='timit' else english[ds+'_'+variant])
   key=ds+'_'+variant
   if any(j['key']==key for j in jobs):continue
   checkpoint=run/'best_dev_r.pt'
   pinned=HERE/'checkpoints'/(key+'.pt');pinned.parent.mkdir(exist_ok=True)
   if not pinned.exists():os.link(checkpoint,pinned)
   assert os.path.samefile(checkpoint,pinned)
   expected=json.loads((run/'final_test.json').read_text())['step']
   jobs.append(dict(key=key,dataset=ds,variant=variant,label=label,source_run=str(run),
      checkpoint=str(pinned),source_checkpoint=str(checkpoint),expected_step=expected,
      original_training_contract=json.loads((run/'run_config.json').read_text())))
 for path in [vocab_file,VIZ/'examples.json',VIZ/'timestamp_manifest.json']:
  sources[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
 (HERE/'INPUTS.json').write_text(json.dumps(dict(scope='four existing visualization examples; bounded real-data probe, not full benchmark',inputs=inputs,jobs=jobs,vocab=vocab,sources_sha256=sources),ensure_ascii=False,indent=2)+'\n')
 print('Prepared',len(inputs),'utterances;',len(jobs),'existing checkpoints pinned by hardlink; no weight copy or retraining.')

if __name__=='__main__':main()
