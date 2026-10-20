"""Prepare audio-only decoder inputs separately from scoring references."""
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent
VIZ=ROOT/'reproduction/segmentation_figures_20261017'


def main():
 examples=json.loads((VIZ/'examples.json').read_text())['examples']
 manifest=json.loads((VIZ/'timestamp_manifest.json').read_text())
 jobs={};references={}
 for rid,ex in examples.items():
  split=ex['split'];dataset='timit' if split=='timit' else 'buckeye' if split=='buckeye' else 'libri100'
  if dataset=='timit':
   meta_path=ROOT/'reproduction/timit_singlehead_k2_fa_20261013/full/jsd_upstream/metrics.json'
   source=ROOT/'reproduction/results/runexp_ottc_fuse_phoneme_jsd/timit_core_test_192.jsonl'
  else:
   directory=ROOT/f'reproduction/{dataset}_singlehead_20261014/evaluation/jsd_upstream'
   meta_path=directory/'metrics.json'
   part='test' if dataset=='buckeye' else 'test_'+split.split('_')[1]
   source=directory/(part+'_predictions.jsonl')
  metadata=json.loads(meta_path.read_text())
  prior=next(r for r in (json.loads(s) for s in source.open()) if Path(r['utt']).stem==ex['original_id'])
  expected={'hypothesis':prior['hypothesis'],'spans':prior['predicted_spans'],'boundaries':prior['predicted_boundaries']} if dataset=='timit' else prior['text_free']
  case=dict(recording_id=rid,utt=ex['original_id'],split=split,wav=manifest[rid]['wav'],duration=ex['duration'],
   wav_sha256=hashlib.sha256(Path(manifest[rid]['wav']).read_bytes()).hexdigest(),prior_audio_only_prediction=expected)
  if dataset not in jobs:
   jobs[dataset]=dict(dataset=dataset,checkpoint=metadata['checkpoint'],checkpoint_sha256=metadata['checkpoint_sha256'],
    prior_metadata=str(meta_path),prior_metadata_sha256=hashlib.sha256(meta_path.read_bytes()).hexdigest(),cases=[])
  jobs[dataset]['cases'].append(case)
  references[rid]=dict(split=split,duration=ex['duration'],reference=ex['reference'],reference_kind=ex['reference_kind'])
 (HERE/'INPUTS.json').write_text(json.dumps(dict(scope='four preselected visualization examples; single-head upstream-JSD only',jobs=list(jobs.values())),indent=2)+'\n')
 (HERE/'SCORING_REFERENCES.json').write_text(json.dumps(references,indent=2)+'\n')
 print('Prepared 3 frozen models, 4 audio inputs. Scoring references are in a separate file.')

if __name__=='__main__':main()
