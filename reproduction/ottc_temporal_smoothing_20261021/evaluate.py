"""Native-grid recognition/segmentation and exact supplied-phone K2 FA."""
from reproduction.ottc_temporal_smoothing_20261021.common import ROOT,HERE,scorer,Collator,phones_from_ids,normalize,write,sha
import gc
import json
from pathlib import Path
import time
import numpy as np
import torch


def greedy_spans(ids,vocab,topo,frame_seconds,duration):
    labels=topo['labels']; spans=[]; previous_phone=None; previous_state=None
    for frame,token in enumerate(ids):
        if token==0: continue
        label=labels[int(token)];phone,state=label.rsplit('_',1);state=int(state)
        if phone!=previous_phone or (previous_state is not None and state<previous_state):
            if spans: spans[-1][2]=min(frame*frame_seconds,duration)
            spans.append([phone,frame*frame_seconds,duration])
        previous_phone,previous_state=phone,state
    return spans


def evaluate(config,model,rows,vocab,processor,topo,output,inference_precision='Standalone FP32 batch1; no Trainer autocast wrapper'):
    from reproduction.timit_ssl_native10_20261019.graphs import graph,decode,spans_from_starts
    from ifmdd_k2_alignment import load_backend,align
    helpers=scorer();inverse={v:k for k,v in vocab.items()};begin=time.monotonic();model.eval()
    modes=['text_free','forced_alignment']+(['fsa_text_free'] if topo else [])
    totals={mode:helpers['new_counts']() for mode in modes};losses=[]
    backend,provenance=load_backend();provenance.pop('source_snapshot')
    vector=graph(topo,vocab,device='cuda') if topo else None
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.with_suffix('.predictions.jsonl').open('w') as stream,torch.inference_mode():
        for i,row in enumerate(rows):
            batch=Collator(processor,topo['mapping'] if topo else None)([dict(wav=row['wav'],labels=row['training_target_ids'])])
            batch={k:v.to('cuda') for k,v in batch.items()};labels=batch.pop('labels')
            out=model(**batch);length=int(out.hidden_states[1][0].sum());logits=out.logits[0,:length].float()
            loss=model(**batch,labels=labels).loss;assert torch.isfinite(loss);losses.append(float(loss))
            ids=logits.argmax(-1).cpu().tolist();hyp=phones_from_ids(ids,vocab,topo)
            ref=row['reference'];duration=row['duration']
            rec=dict(utt=row['utt'],wav=row['wav'],duration=duration,reference=ref,raw_loss=float(loss),
                reference_times_used_by_aligner=False,text_free=dict(hypothesis=hyp,frame_token_ids=ids))
            if topo:
                spans=greedy_spans(ids,vocab,topo,config['frame_ms']/1000,duration)
                boundaries=[t for _,s,e in spans for t in (s,e)];rec['text_free']['predicted_spans']=spans
            else:
                path=np.asarray(ids);path[path==41]=0
                boundaries=((np.flatnonzero(path[1:]!=path[:-1])+1)*config['frame_ms']/1000).tolist()
            count=totals['text_free'];count['utterances']+=1;count['refs'].append(normalize([p for p,_,_ in ref]));count['hyps'].append(normalize(hyp))
            helpers['score'](count,ref,boundaries,duration)
            if topo:
                count=totals['fsa_text_free'];count['utterances']+=1
                try:
                    tokens,path,starts=decode(logits,vector)
                    # Native10ms phone-entry boundaries, shared with mono.
                    spans=[[inverse[t],0. if j==0 else starts[j]*.01,
                        starts[j+1]*.01 if j+1<len(starts) else duration] for j,t in enumerate(tokens)]
                    helpers['score'](count,ref,[t for _,s,e in spans for t in (s,e)],duration)
                    hypothesis=[inverse[t] for t in tokens]
                    rec['fsa_text_free']=dict(hypothesis=hypothesis,predicted_spans=spans,frame_token_ids=path)
                except (ValueError,RuntimeError) as exc:
                    count['alignment_failures']+=1;helpers['score'](count,ref,[],duration);hypothesis=[]
                    rec['fsa_text_free']=dict(failure=str(exc))
                count['refs'].append(normalize([p for p,_,_ in ref]));count['hyps'].append(normalize(hypothesis))
            count=totals['forced_alignment'];count['utterances']+=1;targets=[vocab[p] for p,_,_ in ref]
            try:
                if not targets: raise ValueError('No speech-phone target: retained as inapplicable FA')
                if topo:
                    occurrences,path,starts=decode(logits,graph(topo,vocab,targets,device='cuda'))
                    if occurrences!=list(range(1,len(targets)+1)): raise ValueError('Incomplete supplied-phone occurrence coverage')
                    spans=spans_from_starts([p for p,_,_ in ref],starts,duration,length)
                else:
                    segments,path=align(logits.log_softmax(-1),targets,duration,backend,blank_policy='previous')
                    spans=[[inverse[s['token_id']],s['start'],s['end']] for s in segments]
                assert [p for p,_,_ in spans]==[p for p,_,_ in ref]
                helpers['score'](count,ref,[t for _,s,e in spans for t in (s,e)],duration)
                for (_,rs,re),(_,ps,pe) in zip(ref,spans):
                    err=count['errors'];err['tokens']+=1;err['start']+=abs(rs-ps);err['end']+=abs(re-pe)
                    err['start_hit20']+=abs(rs-ps)<=.02+1e-9;err['end_hit20']+=abs(re-pe)<=.02+1e-9
                rec['forced_alignment']=dict(predicted_spans=spans,frame_token_ids=path,supplied_target_ids=targets)
            except (ValueError,RuntimeError) as exc:
                count['alignment_failures']+=1;helpers['score'](count,ref,[],duration)
                rec['forced_alignment']=dict(failure=str(exc),supplied_target_ids=targets)
            stream.write(json.dumps(rec)+'\n')
            if (i+1)%200==0 or i+1==len(rows):print('BENCHMARK',config['name'],output.name,i+1,'/',len(rows),flush=True)
    result=dict(recipe=config['name'],utterances=len(rows),raw_loss_per_utterance=sum(losses)/len(losses),
        modes={mode:helpers['summary'](count) for mode,count in totals.items()},seconds=time.monotonic()-begin,
        protocol=dict(frame_ms=config['frame_ms'],upsample=False,native_conv_stride=config['native_conv_stride'],
            inference=inference_precision,
            boundary_protocol='unique endpoints; exclude utterance edges20ms; one-to-one matching10/20/30ms; frozen IF-MDD R-value',
            text_free_clock='fixed native convolution step',FA_clock='duration/frame_count; blank previous',
            reference_times_used_by_aligner=False,optimizer_updated=False,k2_source=provenance,
            S3T2_recognition='headline=fsa_text_free original IF-MDD phone-loop; greedy text_free diagnostic only' if topo else None))
    write(output,result);return result


def final_benchmarks(config,model,rows,vocab,processor,topo,output,run):
    if (output/'BENCHMARK_COMPLETE.json').exists():return
    complete=json.loads((output/'TRAIN_COMPLETE.json').read_text());epoch=int(round(complete['epoch']))
    ranked=json.loads((output/'BEST_THREE.json').read_text())['checkpoints']
    candidates={f"best_{i+1}":Path(item['path']) for i,item in enumerate(ranked)}
    model.cuda();model.step=config['step_offset']+complete['step']
    results={}
    # Development for every candidate is completed before any test is opened.
    for split in ['dev','test']:
        for name,path in candidates.items():
            saved=torch.load(path,map_location='cpu',weights_only=False,mmap=True);model.load_state_dict(saved['model']);del saved;gc.collect()
            target=output/f'benchmark_{name}_{split}.json'
            result=json.loads(target.read_text()) if target.exists() else evaluate(config,model,rows[split],vocab,processor,topo,target)
            results.setdefault(name,{})[split]=result
            values={f'benchmark/{name}/{split}/raw_loss_per_utterance':result['raw_loss_per_utterance']}
            for mode,m in result['modes'].items():
                if 'nosil_PER_percent' in m:values[f'benchmark/{name}/{split}/{mode}/PER']=m['nosil_PER_percent']
                for key,value in m['boundaries']['20'].items():values[f'benchmark/{name}/{split}/{mode}/{key}_20ms']=value
            # Standalone checkpoint comparisons remain local-only.
            write(output/f'benchmark_{name}_{split}_scalars.json',values)
        if split=='dev':
            mode='fsa_text_free' if topo else 'text_free'
            selected=min(candidates,key=lambda n:results[n]['dev']['modes'][mode]['nosil_PER_percent'])
            write(output/'DEV_SELECTION.json',dict(selected=selected,mode=mode,test_used=False,
                criterion='minimum FP32 dev no-sil PER among the three best checkpoints retained by training dev PER'))
            write(output/'DEV_BOUNDARY_SELECTION.json',dict(test_used=False,
                nonFA_R=max(candidates,key=lambda n:results[n]['dev']['modes'][mode]['boundaries']['20']['r_value_percent']),
                FA_R=max(candidates,key=lambda n:results[n]['dev']['modes']['forced_alignment']['boundaries']['20']['r_value_percent']),
                FA_MAE=min(candidates,key=lambda n:results[n]['dev']['modes']['forced_alignment']['boundary_MAE_ms'])))
    for name in results:
        for split in ['dev','test']:
            assert results[name][split]['modes']['forced_alignment']['alignment_failures']==0,(name,split)
            if topo:assert results[name][split]['modes']['fsa_text_free']['alignment_failures']==0,(name,split)
    write(output/'BENCHMARK_COMPLETE.json',dict(candidates={n:str(p) for n,p in candidates.items()},selected=selected,test_used_for_selection=False))
