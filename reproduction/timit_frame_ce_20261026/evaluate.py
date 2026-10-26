"""Native-grid recognition/segmentation and exact supplied-phone K2 FA."""
from reproduction.timit_frame_ce_20261026.common import ROOT,HERE,scorer,Collator,phones_from_ids,normalize,write,sha
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
            reference_times_used_by_aligner=False,reference_times_used_for_inference=False,optimizer_updated=False,k2_source=provenance,
            training_supervision=config['frame_supervision'], frame_center_samples=config['frame_center_samples'],
            FA_note='Matched historical CTC-graph diagnostic on frame-CE posteriors; frame CE has no trained blank target',
            S3T2_recognition='headline=fsa_text_free original IF-MDD phone-loop; greedy text_free diagnostic only' if topo else None))
    write(output,result);return result


def average(paths, output, config):
    """FP32 accumulators and weights; leave all member checkpoints intact."""
    if output.exists():
        return
    total = None
    members = []
    for path in paths:
        saved = torch.load(path, map_location='cpu', weights_only=False, mmap=True)
        assert saved['recipe'] == config['name']
        members.append(dict(path=str(path), epoch=saved['epoch'], step=saved['step'], sha256=sha(path)))
        if total is None:
            total = {k: v.float().clone() if v.is_floating_point() else v.clone() for k, v in saved['model'].items()}
        else:
            for key, value in saved['model'].items():
                if value.is_floating_point():
                    assert value.dtype == torch.float32
                    total[key].add_(value)
                else:
                    assert torch.equal(total[key], value)
        del saved
    for value in total.values():
        if value.is_floating_point():
            value.div_(len(paths))
            assert value.dtype == torch.float32
    temporary = output.with_suffix('.tmp')
    torch.save(dict(model=total, recipe=config['name'], members=members), temporary)
    temporary.replace(output)
    write(output.with_suffix('.json'), dict(members=members, sha256=sha(output), accumulation_dtype='float32'))


def final_benchmarks(config, model, rows, vocab, processor, topo, output, run):
    from reproduction.timit_seed_sweep_20261022.pfer import score_predictions
    if (output / 'BENCHMARK_COMPLETE.json').exists():
        return
    complete = json.loads((output / 'TRAIN_COMPLETE.json').read_text())
    last = int(round(complete['epoch']))
    candidates = dict(best_dev=output / 'best_dev.pt')
    for count in (5, 10):
        path = output / f'final_{count}_average.pt'
        average([output / 'epoch_weights' / f'epoch-{e:03d}.pt' for e in range(last-count+1, last+1)], path, config)
        candidates[f'final_{count}'] = path
    results = {}
    model.cuda()
    for split in ('dev', 'test'):
        for name, path in candidates.items():
            saved = torch.load(path, map_location='cpu', mmap=True, weights_only=False)
            model.load_state_dict(saved['model'])
            del saved
            gc.collect()
            target = output / f'benchmark_{name}_{split}.json'
            summary = json.loads(target.read_text()) if target.exists() else evaluate(
                config, model, rows[split], vocab, processor, topo, target)
            pfer = score_predictions(target.with_suffix('.predictions.jsonl'), summary)
            summary['pfer'] = pfer
            summary['modes']['text_free']['PFER_percent'] = pfer['PFER_percent']
            write(target, summary)
            assert summary['utterances'] == len(rows[split])
            assert summary['modes']['forced_alignment']['alignment_failures'] == 0
            results.setdefault(name, {})[split] = summary
        if split == 'dev':
            selected = min(candidates, key=lambda n: results[n]['dev']['modes']['text_free']['nosil_PER_percent'])
            write(output / 'DEV_SELECTION.json', dict(selected=selected, test_used=False,
                criterion='minimum FP32 dev PER among best single and final5/final10',
                historical_comparison_candidate='best_dev'))
    write(output / 'BENCHMARK_COMPLETE.json', dict(selected=selected, test_used_for_selection=False,
        historical_comparison_candidate='best_dev', candidates={n: dict(path=str(p), sha256=sha(p)) for n, p in candidates.items()}))
    write(output / 'FINAL_SUMMARY.json', dict(seed=config['seed'], cost='frame_ce',
        comparison_candidate='best_dev', selected=selected, candidates=results))
