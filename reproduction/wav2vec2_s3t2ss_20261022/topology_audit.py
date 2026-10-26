"""Independent path-language and transport checks; no auxiliary training loss."""
import inspect
import itertools
import json
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
import torch
import torch.nn.functional as F
from common import HERE, ROOT, topology, source_hashes, write, matrix, recipe
from graphs import graph, decode
import ottc.ops_loss.ot_loss as losses


@lru_cache(maxsize=16)
def prepared(fsa):
    fsa=fsa[0]
    arcs=fsa.arcs.values().cpu().tolist()
    aux=fsa.aux_labels.cpu().tolist()
    transitions=defaultdict(list); finals=set()
    for (src,dst,label,_),output in zip(arcs,aux):
        if label == -1: finals.add(src)
        else: transitions[(src,label)].append((dst,output))
    return transitions,finals


def paths(fsa, labels):
    """Execute exact acoustic labels, retaining lexical outputs and finality."""
    transitions,finals=prepared(fsa)
    active={(0,())}
    for label in labels:
        active={(dst, out+((phone,) if phone>0 else ()))
                for src,out in active for dst,phone in transitions[(src,label)]}
    return {out for state,out in active if state in finals}


def independent_phone_language(sequence, state_ids):
    """S3T2**: state0+, optional state1+, state2+; blank between phones."""
    # This recognizer is separate from the vendor arc construction.
    active={(0,0)}  # completed occurrences, phase: boundary/0/1/2
    for label in sequence:
        nxt=set()
        for count,phase in active:
            if phase==0 and label==0: nxt.add((count,0))
            if label==state_ids[0] and phase in (0,1): nxt.add((count,1))
            if label==state_ids[1] and phase in (1,2): nxt.add((count,2))
            if label==state_ids[2] and phase in (1,2,3):
                nxt.add((count,3));nxt.add((count+1,0))
        active=nxt
    return {count for count,phase in active if phase==0}


def transport_parity():
    # Device-only substitution for the existing CUDA kernel's CPU fixtures.
    src=inspect.getsource(losses.batched_ottc_loss_bucketized)
    namespace=dict(losses.batched_ottc_loss_bucketized.__globals__)
    exec(compile(src.replace(".to('cuda')", ".to('cpu')"), 'cpu_device_fixture', 'exec'),namespace)
    core=namespace['batched_ottc_loss_bucketized']
    torch.manual_seed(17)
    report={}
    for cost in ('ce','ag'):
        x=torch.randn(2,5,7,requires_grad=True)
        mass=torch.randn(2,5,requires_grad=True)
        a=mass.softmax(-1)
        ids=torch.tensor([[1,2,3,1],[4,5,6,4]])
        y=F.one_hot(ids,7).float(); b=torch.full((2,4),.25)
        actual=core(x,y,a,b,divergence='ag' if cost=='ag' else None)[0]
        ca=a.cumsum(-1);cb=b.cumsum(-1)
        la=F.pad(ca,(1,0))[:,:-1];lb=F.pad(cb,(1,0))[:,:-1]
        overlap=(torch.minimum(ca[:,:,None],cb[:,None,:])-
                 torch.maximum(la[:,:,None],lb[:,None,:])).clamp_min(0)
        if cost=='ce': costs=-(x.log_softmax(-1)[:,:,None,:]*y[:,None,:,:]).sum(-1)
        else:
            p=x.softmax(-1)[:,:,None,:]
            q=y[:,None,:,:];q=q+(1-q)*1e-12-q*1e-12
            m=(p+q)*.5
            costs=(.5*m*((m.log()-p.log())+(m.log()-q.log()))).sum(-1)
        expected=(overlap*costs).sum()
        actual_grad=torch.autograd.grad(actual,(x,mass),retain_graph=True)
        expected_grad=torch.autograd.grad(expected,(x,mass),retain_graph=True)
        torch.testing.assert_close(actual,expected,atol=3e-6,rtol=3e-6)
        for g,h in zip(actual_grad,expected_grad):
            torch.testing.assert_close(g,h,atol=5e-6,rtol=5e-5)
        legacy=None
        if cost=='ag':
            legacy=core(x,y,a,b,jsd=True)[0]
            torch.testing.assert_close(actual,legacy,atol=0,rtol=0)
        report[cost]=dict(loss=float(actual),dense_overlap_loss=float(expected),
            max_gradient_difference=max(float((g-h).abs().max()) for g,h in zip(actual_grad,expected_grad)),
            legacy_upstream_exact=bool(legacy is not None))
    return report


def main():
    vocab=json.loads((ROOT/'reproduction/models/large_models_results/wav2vec2-large-lv60/runexp_ottc_fuse_phoneme/processor/vocab.json').read_text())
    topo=topology(vocab);phone=vocab['b']; states=topo['mapping'][phone]
    loop=graph(topo,vocab,device='cpu')
    supplied={n:graph(topo,vocab,[phone]*n,device='cpu') for n in (1,2)}
    checked=0
    for length in range(1,7):
        for seq in itertools.product([0]+states,repeat=length):
            expected=independent_phone_language(seq,states)
            actual=paths(loop,seq)
            assert actual=={(phone,)*n for n in expected},(seq,expected,actual)
            for n,g in supplied.items():
                wanted={tuple(range(1,n+1))} if n in expected else set()
                assert paths(g,seq)==wanted,(seq,n,wanted,paths(g,seq))
            checked+=1
    # Silence uses one label with a minimum two-frame phone model, per IF-MDD.
    sil=vocab['sil']; sid=topo['mapping'][sil][0]
    sg=graph(topo,vocab,[sil],device='cpu')
    assert not paths(sg,[sid]) and paths(sg,[sid,sid])=={(1,)}
    # Repeat occurrences are expanded independently; no CTC-style deduplication.
    from common import Collator, load_data
    config=recipe(matrix()[0]);data,rows,_,processor,audit=load_data(config)
    collator=Collator(processor,topo['mapping'])
    actual=collator([dict(wav=rows['train'][0]['wav'],labels=[phone,phone,sil])])['labels'][0].tolist()
    assert actual==states+states+[sid]
    all_occurrences=0
    for split,items in rows.items():
        for row in items:
            for token in row['training_target_ids']:
                assert token in topo['mapping'] and len(topo['mapping'][token])==(1 if token==sil else 3)
                all_occurrences+=1
    # Exact high-margin Viterbi fixtures include optional middle state and repeats.
    for seq,wanted in [(states[::2],[phone]),([states[0]]*2+[states[1]]*2+[states[2]]*2,[phone]),
                       (states[::2]*2,[phone,phone])]:
        scores=torch.full((len(seq),len(topo['labels'])),-30.)
        scores[torch.arange(len(seq)),torch.tensor(seq)]=30.
        assert decode(scores,loop)[0]==wanted
    for name in matrix():
        conf=recipe(name)
        assert conf['topology_forward_sum_weight']==0 and not conf['auxiliary_losses']
    report=dict(source_sha256=source_hashes(),exhaustive_acoustic_sequences=checked,
        independent_grammar_vs_phone_loop_and_supplied_phone_graph=True,
        repeated_phone_occurrences_preserved=True,silence_single_state=True,
        training_phone_occurrences_checked=all_occurrences,data=audit,
        training='pure monotone OT on ordered state targets; state_uniform; no FSA/CTC auxiliary loss',
        topology_distinction='OT target expands all 3 states; S3T2** decoding permits middle-state skip. OT is not a forward-sum over FSA paths.',
        costs=transport_parity())
    write(HERE/'TOPOLOGY_PASS.json',report)
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
