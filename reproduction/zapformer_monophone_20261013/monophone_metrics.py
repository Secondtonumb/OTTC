"""Reuse frozen IF-MDD boundary and R-value scorers for single-phone outputs."""
from collections import Counter
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import jiwer
import torch
import torch.distributed as dist

HERE=Path(__file__).resolve().parent
SUITE=HERE.parent/'english_suite'
SOURCE=SUITE/'mfa_metrics.py'
spec=importlib.util.spec_from_file_location('frozen_monophone_scoring',SOURCE)
scorer=importlib.util.module_from_spec(spec);spec.loader.exec_module(scorer)
DATA=HERE.parent/'spottc_jsd_s3t2_framece_libri100_20261012/eval_data'
scorer.DATA=DATA
sys.path.insert(0,str(HERE.parent/'ipa_arpabet_comparison_20261012'))
from rescore import mapped


def audio_forward(net,arm,x,lengths):
    enc,out_lengths=net.acoustic.forward_encoder(x,lengths,aux_loss_scale=0.)
    logits=net.phone_logits(enc)
    with torch.autocast(x.device.type,enabled=False):
        valid=torch.arange(logits.shape[1],device=x.device)[None]<lengths[:,None]
        mass=net.weight_head(enc.float()).squeeze(-1).masked_fill(~valid,-torch.inf).softmax(-1)
    return logits,out_lengths,mass


scorer.audio_forward=audio_forward


def evaluate(net,split,rank,world,output,limit=0):
    output.mkdir(parents=True,exist_ok=True)
    result=scorer.evaluate(net,'monophone',split=split,rank=rank,world=world,forced=True,limit=limit,output=output)
    references={r['id']:r for r in (json.loads(line) for line in (DATA/f'{split}.jsonl').open())}
    counts=Counter()
    for row in (json.loads(line) for line in (output/f'{split}_predictions.jsonl').open()):
        ref=mapped(references[row['id']]['tokens'],'english_collapsed')
        hyp=mapped(row['predicted_ids'],'english_collapsed')
        edits=jiwer.process_words(' '.join(ref),' '.join(hyp))
        counts.update(reference_phones=len(ref),substitutions=edits.substitutions,
                      deletions=edits.deletions,insertions=edits.insertions,utterances=1)
    if world>1:
        gathered=[None]*world;dist.all_gather_object(gathered,dict(counts));counts=Counter()
        for local in gathered:counts.update(local)
    n=counts['reference_phones'];result['mapped_arpabet']=dict(counts)
    for key in ['substitutions','insertions','deletions']:result['mapped_arpabet'][key+'_percent']=100*counts[key]/n
    result['mapped_arpabet']['PER_percent']=100*sum(counts[k] for k in ['substitutions','insertions','deletions'])/n
    result['protocol'].update(training_topology='OTTC monophone targets; no forward-sum',
        recognition='Audio only; single-phone argmax/CTC collapse; remove non-speech IDs >=103',
        fa='Supplied phones; frozen IF-MDD phone CTC FSA for evaluation, previous blank, duration/frames; no CTC training objective',
        subset='official dev400' if split=='development' else 'official core-test192',
        mapped_recognition='IPA -> English-collapsed ARPAbet/TIMIT39; same frozen mapping and jiwer as SP runs',
        output_variant=net.variant,output_head_dimension=net.acoustic.ctc_output[0].out_features,
        source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        adapter_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),test_checkpoint_selection=False)
    (output/'metrics.json').write_text(json.dumps(result,indent=2)+'\n')
    return result
