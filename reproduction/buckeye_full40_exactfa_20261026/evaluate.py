"""Separate local-only exact-FA replay; preserve every original benchmark artifact."""
import argparse
import fcntl
import gc
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import torch
from transformers import Wav2Vec2Processor
from reproduction.buckeye_full40_seed_sweep_20261025 import common
from reproduction.buckeye_full40_exactfa_20261026.exact import exact_ctc, distribution
from reproduction.ifmdd_k2_alignment import load_backend

HERE=Path(__file__).resolve().parent


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''): h.update(block)
    return h.hexdigest()


def write(p,x):
    p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_suffix(p.suffix+'.tmp'); tmp.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n'); tmp.replace(p)


def run_set(name,candidate,split,model,processor,vocab,rows,contract,source,output,checkpoint_sha,backend,helpers):
    target=output/f'{candidate}_{split}.json'
    old_json=source/'evaluation'/f'{candidate}_{split}.json'
    old_predictions=old_json.with_suffix('.predictions.jsonl')
    original=json.loads(old_json.read_text())
    assert original['checkpoint_sha256']==checkpoint_sha
    assert original['dataset']==contract
    provenance=dict(checkpoint_sha256=checkpoint_sha,dataset=contract,
                    original_metrics_sha256=sha(old_json),original_predictions_sha256=sha(old_predictions),
                    evaluator_sha256=sha(Path(__file__)),decoder_sha256=sha(HERE/'exact.py'))
    if target.exists():
        result=json.loads(target.read_text()); assert result['provenance']==provenance
        assert result['utterances']==len(rows)
        assert sha(target.with_suffix('.predictions.jsonl'))==result['predictions_sha256']
        return result
    inverse={v:k for k,v in vocab.items()}
    counts=helpers['new_counts'](); errors=[]; old_errors=[]; utterance_errors=[]
    changed=0; gained_score=0.; n=0; started=time.monotonic()
    staging=target.with_suffix('.predictions.jsonl.partial')
    assert not model.training
    with old_predictions.open() as old_stream, staging.open('w') as stream, torch.inference_mode():
        for n,row in enumerate(rows,1):
            previous=json.loads(next(old_stream))
            ref=row['reference']; assert previous['utt']==row['utt'] and previous['reference']==ref
            targets=[vocab[p] for p,_,_ in ref]
            # Audio-only forward; reference IDs are used only by the downstream FA graph.
            batch=common.base.Collator(processor,None)([dict(wav=row['wav'],labels=targets)])
            batch.pop('labels'); batch={k:v.cuda() for k,v in batch.items()}
            out=model(**batch); length=int(out.hidden_states[1][0].sum())
            logits=out.logits[0,:length].float()
            assert logits.argmax(-1).cpu().tolist()==previous['text_free']['frame_token_ids'], (name,candidate,split,row['utt'],'argmax changed')
            lp=logits.log_softmax(-1).cpu().numpy()
            path,path_score=exact_ctc(lp,targets)
            old_path=np.asarray(previous['forced_alignment']['frame_token_ids'],dtype=np.int64)
            assert len(old_path)==length
            old_score=float(lp[np.arange(length),old_path].sum(dtype=np.float64))
            assert path_score+1e-4>=old_score
            gains=path_score-old_score
            changed+=gains>1e-4; gained_score+=gains
            segments=backend['dense_path_to_target_segments'](
                torch.from_numpy(path),torch.from_numpy(np.exp(lp[np.arange(length),path])),targets,0,row['duration'],'previous')
            spans=[[inverse[s['token_id']],s['start'],s['end']] for s in segments]
            assert [p for p,_,_ in spans]==[p for p,_,_ in ref]
            counts['utterances']+=1
            helpers['score'](counts,ref,[t for _,s,e in spans for t in (s,e)],row['duration'])
            current=[]
            for r,p,old in zip(ref,spans,previous['forced_alignment']['predicted_spans']):
                e=counts['errors']; e['tokens']+=1
                es,ee=abs(r[1]-p[1]),abs(r[2]-p[2])
                e['start']+=es; e['end']+=ee; e['start_hit20']+=es<=.02+1e-9; e['end_hit20']+=ee<=.02+1e-9
                current.extend((es*1000,ee*1000)); old_errors.extend((abs(r[1]-old[1])*1000,abs(r[2]-old[2])*1000))
            errors.extend(current); utterance_errors.append(float(np.mean(current)))
            record=dict(utt=row['utt'],wav=row['wav'],duration=row['duration'],reference=ref,
                reference_times_used_by_aligner=False,forced_alignment=dict(predicted_spans=spans,
                frame_token_ids=path.tolist(),supplied_target_ids=targets,path_log_score=path_score,
                old_path_log_score=old_score),original_predictions=str(old_predictions))
            stream.write(json.dumps(record,ensure_ascii=False)+'\n')
            if n==8:
                torch.cuda.synchronize()
                write(output/f'{candidate}_{split}_GPU_GATE.json',dict(passed=True,utterances=8,
                    seconds=time.monotonic()-started,all_model_parameters_FP32=all(p.dtype==torch.float32 for p in model.parameters()),
                    audio_only_argmax_matches_original=True,exact_score_dominates_old_path=True,
                    peak_allocated_bytes=torch.cuda.max_memory_allocated(),evaluation_uploads=False))
            if n%200==0 or n==len(rows):
                progress=dict(recipe=name,candidate=candidate,split=split,utterances=n,expected=len(rows),
                              seconds=time.monotonic()-started,improved_search_paths=changed)
                write(output/'PROGRESS.json',progress); print(json.dumps(progress),flush=True)
        assert not old_stream.readline(), 'Extra original predictions'
    assert n==len(rows)
    metrics=helpers['summary'](counts)
    assert abs(metrics['boundary_MAE_ms']-np.mean(errors))<1e-7
    assert abs(original['modes']['forced_alignment']['boundary_MAE_ms']-np.mean(old_errors))<1e-7
    staging.replace(target.with_suffix('.predictions.jsonl'))
    result=dict(recipe=name,candidate=candidate,split=split,utterances=n,provenance=provenance,
        modes=dict(text_free=original['modes']['text_free'],forced_alignment=metrics),
        original_forced_alignment=original['modes']['forced_alignment'],
        endpoint_errors=distribution(errors),original_endpoint_errors=distribution(old_errors),
        utterance_MAE=distribution(utterance_errors),improved_search_paths=changed,total_path_log_score_gain=gained_score,
        seconds=time.monotonic()-started,predictions_sha256=sha(target.with_suffix('.predictions.jsonl')),
        protocol=dict(search='unpruned exact standard CTC Viterbi; float64 path-score accumulation',
            blank_id=0,blank_policy='previous',FA_clock='duration/frame_count',
            boundary_protocol='unchanged unique endpoints; exclude utterance edges20ms; one-to-one matching; frozen IF-MDD R-value',
            inference='FP32 batch1; eval; no autocast; audio only',reference_times_used_by_aligner=False,
            target_sequence='identical supplied mapped phone occurrences; no added sil; no occurrence deletion',
            text_free_metrics='reused unchanged from original predictions; argmax checked for every utterance',
            tail_handling='all samples retained; capped statistics diagnostic only; no gold-dependent repair selection'),
        evaluation_uploads=False)
    write(target,result)
    return result


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--recipe',required=True); args=parser.parse_args()
    os.environ['OTTC_EVAL_UPLOAD']='0'
    torch.set_num_threads(4)
    name=args.recipe; source=common.HERE/'models'/name; output=HERE/'models'/name
    output.mkdir(parents=True,exist_ok=True)
    with (output/'.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (output/'COMPLETE.json').exists(): return
        config=json.loads((source/'run_config.json').read_text())
        assert json.loads((source/'BENCHMARK_COMPLETE.json').read_text())['evaluation_uploads'] is False
        # Model, original evaluator, scorer and manifest source contracts remain frozen.
        assert config['source_sha256']==common.source_hashes()
        rows,vocab,contracts,data_root=common.checked_rows(); assert contracts==config['expected_data']
        selected=json.loads((source/'DEV_SELECTION.json').read_text())
        members=json.loads((source/'EPOCHS.json').read_text())
        best=min(members,key=lambda x:(x['dev_PER'],x['epoch']))
        paths=dict(best_dev=Path(best['path']),best3=source/'averages/best3.pt')
        expected=dict(best_dev=best['sha256'],best3=json.loads(paths['best3'].with_suffix('.json').read_text())['sha256'])
        processor=Wav2Vec2Processor.from_pretrained(data_root/'processor')
        model,topo=common.base.build_model(config,vocab,285); assert topo is None
        backend,_=load_backend(); helpers=common.base.scorer(); results={}
        for candidate,path in paths.items():
            assert sha(path)==expected[candidate]
            saved=torch.load(path,map_location='cpu',mmap=True,weights_only=False)
            assert all(not v.is_floating_point() or v.dtype==torch.float32 for v in saved['model'].values())
            model.load_state_dict(saved['model']); model.step=saved['step']; del saved; gc.collect()
            assert all(p.dtype==torch.float32 for p in model.parameters())
            model.cuda().eval(); results[candidate]={}
            # Best-dev test is prioritized for the user's immediate comparison.
            for split in ('test','dev','timit_test','timit_dev'):
                results[candidate][split]=run_set(name,candidate,split,model,processor,vocab,rows[split],contracts[split],source,output,expected[candidate],backend,helpers)
        write(output/'FINAL_SUMMARY.json',dict(recipe=name,results=results,selected=selected['selected'],
             selection=selected,selection_unchanged=True,test_used_for_selection=False,evaluation_uploads=False))
        write(output/'COMPLETE.json',dict(recipe=name,complete=True,checkpoints=expected,
            candidates=list(paths),splits=['test','dev','timit_test','timit_dev'],evaluation_uploads=False))


if __name__=='__main__': main()
