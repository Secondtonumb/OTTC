"""Independent raw-IPA and normalization sensitivity audit; no new inference."""
from pathlib import Path
import collections,json,unicodedata,importlib.util
import panphon
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];ft=panphon.FeatureTable()
spec=importlib.util.spec_from_file_location('distance_independent',ROOT.parent/'fa-bench/analysis/panphon_pfer/distance.py');dist=importlib.util.module_from_spec(spec);spec.loader.exec_module(dist)
table=json.loads((ROOT.parent/'seamless_communication/outputs/panphon_libri100/panphon_table.json').read_text())['phones']
def norm(s):return unicodedata.normalize('NFD',s.replace('g','ɡ').replace('/','').replace(' ',''))
results={}
for model in ['zipa','powsm']:
 for split in ['dev','test']:
  pred={r['utt']:r for r in map(json.loads,(HERE/model/f'{split}.predictions.jsonl').open())};scores=[json.loads(l) for l in (HERE/model/f'{split}.scores.jsonl').open()];counts=collections.Counter();subs=collections.Counter()
  for r in scores:
   p=pred[r['utt']];ref=r['reference'];hyp=r['hypothesis'];rawhyp=norm(p['hypothesis']);rawref=norm(''.join(table[ph]['ipa'] for ph in ref));rh=ft.ipa_segs(rawhyp);rr=ft.ipa_segs(rawref)
   # Direct IPA sensitivity; unlike the headline, PanPhon cannot represent unknown orthographic chars. Record them explicitly.
   rem=rawhyp
   for v in rh:rem=rem.replace(norm(v),'',1)
   counts.update(rawIPA_edits=dist.token_distance(rh,rr),rawIPA_reference=len(rr),rawIPA_feature_cost=dist.feature_distance(ft.word_to_vector_list(rawhyp,numeric=True),ft.word_to_vector_list(rawref,numeric=True)),dropped_unparsed_chars=len(rem))
   for label,mp in [('original',{}),('ao_zh_39',{'ao':'aa','zh':'sh'}),('vowel_alias',{'ao':'aa','zh':'sh','iy':'ih','uw':'uh','ah':'ih'})]:
    a=[mp.get(t,t) for t in ref];b=[mp.get(t,t) for t in hyp];counts[label+'_edits']+=dist.token_distance(a,b);counts[label+'_refs']+=len(a)
  d={'utterances':len(scores),'rawIPA_PER':100*counts['rawIPA_edits']/counts['rawIPA_reference'],'rawIPA_PFER':100*counts['rawIPA_feature_cost']/counts['rawIPA_reference'],'unparsed_chars':counts['dropped_unparsed_chars'],'ARPAbet_PER':100*counts['original_edits']/counts['original_refs'],'ao_zh_merged_PER':100*counts['ao_zh_39_edits']/counts['ao_zh_39_refs'],'vowel_merge_diagnostic_PER_not_benchmark':100*counts['vowel_alias_edits']/counts['vowel_alias_refs']}
  results[model+'_'+split]=d;print(model,split,d,flush=True)
(HERE/'audit/SCORE_SENSITIVITY.json').write_text(json.dumps(results,indent=2)+'\n')
