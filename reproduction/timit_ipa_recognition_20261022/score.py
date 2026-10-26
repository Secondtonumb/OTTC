"""Score audio-only IPA hypotheses in the common TIMIT ARPAbet protocol."""
import ast,collections,csv,hashlib,importlib.util,json,sys,unicodedata
from pathlib import Path
import numpy as np
import panphon
ROOT=Path(__file__).resolve().parents[2];WORK=ROOT.parent;HERE=Path(__file__).resolve().parent
HELPER=WORK/'IF-MDD/zipa_try/eval_zipa_cmu_monophone.py';TABLE=WORK/'seamless_communication/outputs/panphon_libri100/panphon_table.json';DIST=WORK/'fa-bench/analysis/panphon_pfer/distance.py';RSOURCE=ROOT/'reproduction/english_suite/mfa_protocol_comparison/scorer_source/boundary_rvalue.py'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p):return json.loads(Path(p).read_text())
def imp(name,p):
 spec=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
def write(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')
backend=imp('validated_feature_distance',DIST);rv=imp('frozen_rvalue',RSOURCE);ft=panphon.FeatureTable();table=load(TABLE)['phones']
tree=ast.parse(HELPER.read_text());constants={}
for n in tree.body:
 if isinstance(n,ast.AnnAssign) and isinstance(n.target,ast.Name) and n.target.id=='IPA_TO_ARPABET_SEQS':constants[n.target.id]=ast.literal_eval(n.value)
 if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='DROP_IPA_TOKENS' for t in n.targets):constants['DROP_IPA_TOKENS']=ast.literal_eval(n.value)
# Keep the historical mapping, with explicit common-protocol flap/glottal aliases.
mapping=[(''.join(seq),phone) for seq,phone in constants['IPA_TO_ARPABET_SEQS'] if seq not in [('ɾ',),('ʔ',)]]+[('ɾ','d'),('ʔ','q'),('ɚ','er'),('ɝ','er')]
mapping.sort(key=lambda x:-len(x[0]));drop=constants['DROP_IPA_TOKENS']|{'͡','͜','ˈ','ˌ','/',' ','▁'}
def norm(seq):return ['d' if p=='dx' else p for p in seq if p not in ['sil','q']]
def canonical_segments(raw):
 chars=[];owners=[];ignored=collections.Counter()
 for i,s in enumerate(raw):
  for c in unicodedata.normalize('NFD',s['label'].replace('ɡ','g')):
   if c in drop:ignored[c]+=1;continue
   chars.append(c);owners.append(i)
 text=''.join(chars);i=0;phones=[];spans=[]
 while i<len(text):
  if text[i]=='˞':
   if phones:
    previous=phones[-1];base=table[previous]['ipa'] if previous in table else previous.removeprefix('ipa_')
    phones[-1]='ipa_'+base+'˞';spans[-1][0]=phones[-1];spans[-1][2]=raw[owners[i]]['end']
   else:ignored['orphan_rhotic_modifier']+=1
   i+=1;continue
  for ipa,phone in mapping:
   if text.startswith(ipa,i):length=len(ipa);break
  else:ipa=text[i];phone='ipa_'+ipa;length=1
  if phone not in ['sil','q']:
   a=raw[owners[i]]['start'];b=raw[owners[i+length-1]]['end'];phones.append(phone);spans.append([phone,a,b])
  i+=length
 return phones,spans,ignored
unknown=collections.Counter()
def vectors(phones):
 out=[]
 for p in phones:
  if p in table:out.extend(table[p]['segment_features'])
  else:
   assert p.startswith('ipa_'),p;ipa=p[4:];v=ft.word_to_vector_list(ipa,numeric=True);out.extend(v if v else ['UNMAPPED:'+ipa]);unknown[p]+=1
 return out
def feature_cost(a,b):
 if all(not isinstance(x,str) for x in a+b):return backend.feature_distance(a,b)
 # Exact existing 1/48-unit costs for valid vectors; an unmappable symbol
 # remains an explicit unit-cost edit, rather than disappearing from PFER.
 def gap(v):return 48 if isinstance(v,str) else 48-sum(x==0 for x in v)
 previous=[0]
 for v in b:previous.append(previous[-1]+gap(v))
 for av in a:
  current=[previous[0]+gap(av)]
  for j,bv in enumerate(b,1):
   sub=(0 if av==bv else 48) if isinstance(av,str) or isinstance(bv,str) else sum(abs(x-y) for x,y in zip(av,bv))
   current.append(min(previous[j]+gap(av),current[-1]+gap(bv),previous[j-1]+sub))
  previous=current
 return previous[-1]/48

# The feature backend and table must agree with PanPhon24-vector representation.
for p in ['aa','ah','ch','er','ey','d']:
 assert ft.word_to_vector_list(table[p]['ipa'],numeric=True)==table[p]['segment_features'],p
matcher=ast.parse((ROOT/'reproduction/libri100_singlehead_20261014/train.py').read_text());fn=next(n for n in matcher.body if isinstance(n,ast.FunctionDef) and n.name=='boundary_matches');ns={};exec(compile(ast.Module(body=[fn],type_ignores=[]),'<frozen matcher>','exec'),ns);matches=ns['boundary_matches']
all_results={}
for model in ['zipa','powsm']:
 out=HERE/model;assert load(out/'INFERENCE_COMPLETE.json')['status']=='complete';summaries={}
 for split,expected in [('dev',400),('test',192)]:
  unknown.clear()
  path=out/f'{split}.predictions.jsonl';done=load(out/f'{split}.INFERENCE_COMPLETE.json');assert sha(path)==done['predictions_sha256'];rows=[json.loads(l) for l in path.open()];assert len(rows)==len({r['utt'] for r in rows})==expected
  totals=collections.Counter();by_tol={ms:collections.Counter() for ms in [10,20,30]};scores=[];dropped=collections.Counter()
  for r in rows:
   ref=norm([p for p,_,_ in r['reference']]);hyp,spans,ignored=canonical_segments(r['raw_segments']);dropped.update(ignored);rf,hf=vectors(ref),vectors(hyp);cost=feature_cost(hf,rf);errors=backend.token_distance(ref,hyp)
   totals.update(utterances=1,errors=errors,reference_phones=len(ref),feature_cost=cost,reference_IPA_segments=len(rf))
   duration=r['duration'];rb=sorted({t for _,s,e in r['reference'] for t in [s,e] if .02<t<duration-.02});pb=sorted({s for _,s,_ in spans[1:] if .02<s<duration-.02})
   for ms,c in by_tol.items():c.update(matches=matches(rb,pb,ms/1000),reference=len(rb),predicted=len(pb))
   scores.append(dict(utt=r['utt'],reference=ref,hypothesis=hyp,feature_edit_cost=cost,phone_errors=errors,reference_phones=len(ref),reference_IPA_segments=len(rf),predicted_spans=spans))
  summary=dict(model=model,split=split,utterances=expected,PER=100*totals['errors']/totals['reference_phones'],PFER=100*totals['feature_cost']/totals['reference_IPA_segments'],counts=dict(totals),boundaries={},dropped_IPA_modifiers=dict(dropped),unknown_hypothesis_symbols=dict(unknown))
  for ms,c in by_tol.items():
   m,n,p=c['matches'],c['reference'],c['predicted'];prec=100*m/p if p else 0.;rec=100*m/n if n else 0.
   summary['boundaries'][str(ms)]=dict(c,precision=prec,recall=rec,F1=200*m/(n+p),R_value=100*rv.r_value(prec/100,rec/100)[0])
  (out/f'{split}.scores.jsonl').write_text(''.join(json.dumps(s,ensure_ascii=False)+'\n' for s in scores));write(out/f'{split}.json',summary);write(out/f'{split}.COMPLETE.json',dict(utterances=expected,predictions_sha256=sha(path),scores_sha256=sha(out/f'{split}.scores.jsonl'),metrics_sha256=sha(out/f'{split}.json')));summaries[split]=summary
  print('SCORED',model,split,summary['PER'],summary['PFER'],summary['boundaries']['20'],flush=True)
 write(out/'COMPLETE.json',dict(status='complete',dev=400,test=192));all_results[model]=summaries
write(HERE/'RESULTS.json',dict(results=all_results,protocol={'PER':'audio-only IPA mapped to ARPAbet; same ifmdd_arpabet reference; remove sil/q and dx→d; pooled token edit counts','PFER':'same frozen PanPhon0.22.2 feature table and24-vector edit backend as TIMIT sweep; canonicalized hypotheses; sum costs/reference expanded segments','mapping':'historical ZIPA reverse IPA map shared for both; tied affricates joined; IPA g spelling normalized; explicit ɾ→d and ʔ→q(remove); dropped modifiers audited; residual rhotic modifiers attached to preceding mapped phone as an explicit IPA phone; unknown IPA retained as distinct PER token with its PanPhon vectors','segmentation':'physical20/40ms; common reference unique endpoints; exclude edges20ms; next canonical phone onset; chronological one-to-one; frozen IF-MDD R-value','unmappable_hypothesis_symbols':'retained as distinct PER tokens and full unit-cost PFER edits; known-vector costs unchanged; counts reported','no_reference_conditioned_decode':True},source_sha256={str(p):sha(p) for p in [Path(__file__),HELPER,TABLE,DIST,RSOURCE]}))
write(HERE/'COMPLETE.json',dict(status='complete',models=2,dev_per_model=400,test_per_model=192));print('TIMIT_IPA_RECOGNITION_COMPLETE',flush=True)
