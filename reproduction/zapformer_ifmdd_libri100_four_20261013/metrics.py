"""Separate original IF-MDD greedy, constrained FSA and supplied-phone alignment."""
from pathlib import Path
from collections import Counter
import hashlib
import json
import sys
import time
import numpy as np
import torch
import torch.distributed as dist
import jiwer
from task_ottc import HERE, decode
from fsa_decode import graph, decode_fsa, forced_align

SUITE=HERE.parent/'english_suite'
DATA=HERE.parent/'spottc_jsd_s3t2_framece_libri100_20261012/eval_data'
sys.path.insert(0,str(SUITE))
from mfa_metrics import boundary_scores, summarize
sys.path.insert(0,str(HERE.parent/'ipa_arpabet_comparison_20261012'))
from rescore import mapped


def count_edits(count, ref, hyp):
    e=jiwer.process_words(' '.join(map(str,ref)),' '.join(map(str,hyp)))
    count.update(utterances=1,reference_phones=len(ref),phone_errors=e.substitutions+e.deletions+e.insertions,
                 substitutions=e.substitutions,deletions=e.deletions,insertions=e.insertions)


def evaluate(net, split, rank, world, output, limit=0):
    output.mkdir(parents=True,exist_ok=True)
    rows=[json.loads(line) for line in (DATA/f'{split}.jsonl').open()]
    if limit:rows=rows[:limit]
    vector,text=graph(net.loss_module,device='cuda');(output/'recognition.fsa.txt').write_text(text)
    counts={name:Counter() for name in ('text_free','fsa_text_free','forced_k2','mapped_arpabet','fsa_mapped_arpabet')}
    caches={};started=time.monotonic()
    # Map sorted substate outputs back to original MFA IDs, including silence/blank.
    from fsa_decode import INVENTORY
    state_to_mfa=[0]+[INVENTORY[label.rsplit('_',1)[0]] for label in net.loss_module.labels[1:]]
    state_to_mfa=torch.tensor(state_to_mfa,device='cuda')
    with (output/f'{split}_predictions.jsonl').open('w') as stream,torch.no_grad():
        for index,row in enumerate(rows):
            if index%world!=rank:continue
            path=row['feature_file']
            if path not in caches:caches[path]=np.memmap(path,dtype=np.float32,mode='r')
            x=np.array(caches[path][row['feature_offset']:row['feature_offset']+row['frames']*80]).reshape(-1,80)
            x=torch.from_numpy(x).unsqueeze(0).cuda();lengths=torch.tensor([row['frames']],device='cuda')
            with torch.autocast('cuda',dtype=torch.bfloat16):logits,out_lengths,_=net.acoustic_logits(x,lengths)
            scores=logits[0,:row['frames']].float();hyp=decode(logits,out_lengths,net.loss_module)[0]
            fsa_hyp,fsa_path,_=decode_fsa(scores,vector)
            hyp=[p for p in hyp if 3<=p<=102];fsa_hyp=[p for p in fsa_hyp if 3<=p<=102]
            frame_ids=state_to_mfa[scores.argmax(-1)].tolist();fsa_frame_ids=state_to_mfa[torch.tensor(fsa_path,device='cuda')].tolist()
            for name,phones,frames in [('text_free',hyp,frame_ids),('fsa_text_free',fsa_hyp,fsa_frame_ids)]:
                count_edits(counts[name],row['tokens'],phones)
                edges=[i*.01 for i in range(1,len(frames)) if frames[i]!=frames[i-1]]
                boundary_scores(counts[name],row['references'],edges,row['duration'])
            for name,phones in [('mapped_arpabet',hyp),('fsa_mapped_arpabet',fsa_hyp)]:
                count_edits(counts[name],mapped(row['tokens'],'english_collapsed'),mapped(phones,'english_collapsed'))
            supplied=[p['mfa_id'] for p in row['phone_intervals']]
            spans,k2path=forced_align(scores,supplied,row['duration'],net.loss_module)
            segments=[(s['token_id'],s['start'],s['end']) for s,p in zip(spans,row['phone_intervals']) if p['score_speech']]
            assert [p for p,_,_ in segments]==row['tokens']
            boundary_scores(counts['forced_k2'],row['references'],[t for _,s,e in segments for t in [s,e]],row['duration'])
            counts['forced_k2']['utterances']+=1
            for (_,rs,re),(_,ps,pe) in zip(row['references'],segments):
                union=max(re,pe)-min(rs,ps)
                counts['forced_k2'].update(aligned_tokens=1,start_error=abs(rs-ps),end_error=abs(re-pe),iou=max(0,min(re,pe)-max(rs,ps))/union if union else 0.)
            stream.write(json.dumps(dict(id=row['id'],reference_ids=row['tokens'],predicted_ids=hyp,
                                         fsa_predicted_ids=fsa_hyp,frame_phone_ids=frame_ids,
                                         fsa_state_ids=fsa_path,forced_k2=dict(spans_seconds=segments,supplied_ids=supplied,frame_ids=k2path)))+'\n')
    if world>1:
        gathered=[None]*world;dist.all_gather_object(gathered,{k:dict(v) for k,v in counts.items()})
        counts={k:Counter() for k in counts}
        for local in gathered:
            for k,v in local.items():counts[k].update(v)
    result={k:summarize(v) for k,v in counts.items()};result['forced_k2'].pop('PER_percent',None)
    from segmentation_metrics import SOURCE
    result.update(split=split,limit=limit,elapsed_seconds=time.monotonic()-started,
                  protocol=dict(subset='official dev400' if split=='development' else 'official core-test192',
                    recognition='Audio only; original IF-MDD greedy decoder; separate exact S3T2** phone-loop FSA shortest path',
                    training=dict(net.loss_config),
                    segmentation='Phone-level changes including blank, excludes within-phone substate changes; 10ms grid; one-to-one boundary matching at 10/20/30ms; edge exclusion 20ms',
                    fa='Exact IF-MDD supplied-phone S3T2** graph; native MFA silence retained; previous blank; duration/frames; no reference times used',
                    test_checkpoint_selection=False,R_value=dict(source=str(SOURCE),sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest()),
                    scorer_sha256=hashlib.sha256((SUITE/'mfa_metrics.py').read_bytes()).hexdigest(),
                    adapter_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
    (output/'metrics.json').write_text(json.dumps(result,indent=2)+'\n')
    return result
