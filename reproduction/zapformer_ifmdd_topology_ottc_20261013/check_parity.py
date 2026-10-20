"""Compare live IF-MDD loss/gradient and decoder, and independently audit the FSA."""
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import torch
from task_ottc import HERE, criterion, ottc_objective, frame_masses, decode
from ifmdd_losses import pad_flat_targets
from fsa_decode import graph, decode_fsa, INVENTORY

ROOT=Path('/work/gm64/m64000/IF-MDD')


def reference_module():
    # The live source is loaded intact. Frozen dependencies are byte-identical,
    # checked separately below; runtime transport import is also hash-checked.
    spec=importlib.util.spec_from_file_location('live_ifmdd_topology',ROOT/'utils/subphonetic_topology.py')
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
    return module


def main():
    torch.manual_seed(42);torch.set_num_threads(4)
    hashes=json.loads((HERE/'source_manifest.json').read_text())
    for name,record in hashes.items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==record['sha256']
        assert hashlib.sha256((HERE/'vendor'/name).read_bytes()).hexdigest()==record['sha256']
    from utils.losses import ot_loss
    assert Path(ot_loss.__file__).resolve()==(HERE/'vendor/utils/losses/ot_loss.py').resolve()
    live=reference_module()
    ref=live.SubphoneticTopologyLoss(str(HERE/'mfa_label_encoder.txt'),topology='s3_t2_star_star',
                                   loss_type='ottc',split_silence=False,ottc_mass_mode='state_uniform')
    frozen=criterion()
    assert frozen.labels==ref.labels and torch.equal(frozen.content_to_phone,ref.content_to_phone)
    source=ROOT/'models/phn_mono_ssl_model_v3_refactored.py'
    tree=ast.parse(source.read_text())
    fn=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='_compute_subphonetic_topology_frame_weights')
    ns={'torch':torch};exec(compile(ast.Module(body=[fn],type_ignores=[]),str(source),'exec'),ns)
    class ReferenceMass:
        def __init__(self,raw):
            self.modules=type('Modules',(),{'topology_lm_weight':lambda _self,_enc:raw.unsqueeze(-1)})()
    lengths=torch.tensor([31,23,17]);tl=torch.tensor([5,3,2])
    tokens=torch.tensor([7,7,31,103,104,31,62,62,7,7]);padded=pad_flat_targets(tokens,tl)
    cases=[]
    for name,uniform,extreme in [('learned',False,False),('uniform',True,False),('clamp',False,True)]:
        x=torch.randn(3,31,frozen.output_neurons,requires_grad=True)
        if extreme:
            with torch.no_grad():x[0,0,:5]=torch.tensor([100.,-100.,float('nan'),float('inf'),float('-inf')])
        z=x.detach().clone().requires_grad_();w=torch.randn(3,31,requires_grad=True);v=w.detach().clone().requires_grad_()
        actual=ottc_objective(frozen,x,lengths,tokens,tl,None if uniform else w)
        expected_masses=None if uniform else ns[fn.name](ReferenceMass(v),v.unsqueeze(-1),lengths.float()/31)
        if not uniform:torch.testing.assert_close(frame_masses(w,lengths),expected_masses,rtol=0,atol=0)
        expected=ref(z.log_softmax(-1),padded,lengths.float()/31,tl.float()/5,logits=z,weights_logits=expected_masses)
        actual.backward();expected.backward()
        torch.testing.assert_close(actual,expected,rtol=0,atol=0)
        torch.testing.assert_close(x.grad,z.grad,rtol=0,atol=0)
        if not uniform:
            torch.testing.assert_close(w.grad,v.grad,rtol=0,atol=0)
            assert float(w.grad.abs().sum())>0
        for i,length in enumerate(frozen.last_k2_debug['input_lens']):
            assert torch.count_nonzero(x.grad[i,int(length):])==0
        cases.append(dict(case=name,loss=float(actual),logit_gradient_max_difference=float((x.grad-z.grad).abs().max()),
                          mass_gradient_max_difference=None if uniform else float((w.grad-v.grad).abs().max())))
    # Independent interval-overlap transport oracle, not the bucketization code.
    captures=[];original=ot_loss.batched_ottc_loss_bucketized
    def capture(**kwargs):
        out=original(**kwargs);captures.append((kwargs,out));return out
    ot_loss.batched_ottc_loss_bucketized=capture
    x=torch.randn(3,31,frozen.output_neurons);w=torch.randn(3,31)
    actual=ottc_objective(frozen,x,lengths,tokens,tl,w)
    ot_loss.batched_ottc_loss_bucketized=original
    inputs,out=captures[0];oracle=actual.new_zeros(())
    for i in range(3):
        a,b=inputs['a'][i],inputs['b'][i]
        ca=torch.cat([a.new_zeros(1),a.cumsum(0)]);cb=torch.cat([b.new_zeros(1),b.cumsum(0)])
        overlap=(torch.minimum(ca[1:,None],cb[None,1:])-torch.maximum(ca[:-1,None],cb[None,:-1])).clamp_min(0)
        costs=-inputs['x'][i].log_softmax(-1)@inputs['y'][i].T
        oracle+=(overlap*costs).sum()
    torch.testing.assert_close(actual,oracle,rtol=1e-6,atol=1e-6)
    assert not inputs['jsd']
    # All normal target occurrences expand in order, silence retains one state.
    expanded=[frozen._content_targets_to_substate_ids(row,int(n)) for row,n in zip(padded,tl)]
    assert len(expanded[0])==11 and all(0 not in seq for seq in expanded)
    fixtures=[];b=INVENTORY['b'];pid=int(frozen.content_to_phone[b]);s=frozen.phone_to_substates[pid]
    for path,expected in [(s[:1]*2+s[1:2]*2+s[2:]*2,[b]),([s[0],s[2],s[0],s[2]],[b,b]),([s[1]],[b]),([s[0],0,s[0]],[b])]:
        logits=torch.full((1,len(path),frozen.output_neurons),-30.)
        logits[0,torch.arange(len(path)),torch.tensor(path)]=0
        hyp=decode(logits,torch.tensor([len(path)]),frozen)[0]
        from speechbrain.decoders.ctc import ctc_greedy_decode
        ids=ctc_greedy_decode(logits,torch.ones(1),blank_id=0)[0]
        expected_live=[INVENTORY[p] for p in live.decode_subphone_ids(ids,ref.labels,mode='phone').split()]
        assert hyp==expected==expected_live
        fixtures.append(dict(state_path=path,hypothesis=hyp,labels=[frozen.labels[p] for p in path]))
    vector,text=graph(frozen)
    for path,expected in [(s[:1]*2+s[1:2]*2+s[2:]*2,[b]),([s[0],s[2],s[0],s[2]],[b,b])]:
        scores=torch.full((len(path),frozen.output_neurons),-30.);scores[torch.arange(len(path)),torch.tensor(path)]=0
        hyp,actual_path,_=decode_fsa(scores,vector);assert hyp==expected and actual_path==path
    # Independent max-plus DP checks the shortest-path score over the complete graph.
    scores=torch.randn(8,frozen.output_neurons).log_softmax(-1)
    arcs=[tuple(map(int,line.split()[:4])) for line in text.splitlines() if len(line.split())==5 and int(line.split()[2])>=0]
    table={0:0.}
    for frame in scores:
        new={}
        for src,dst,label,output in arcs:
            if src in table:new[dst]=max(new.get(dst,-float('inf')),table[src]+float(frame[label]))
        table=new
    _,path,_=decode_fsa(scores,vector)
    best_score=sum(float(scores[t,label]) for t,label in enumerate(path))
    assert abs(best_score-table[0])<1e-5,(best_score,table[0])
    report=dict(status='passed',objective='exact IF-MDD topology OTTC CE, not CTC',cases=cases,
                independent_transport_oracle_difference=float((actual-oracle).abs()),
                expanded_targets=expanded,decoder_fixtures=fixtures,FSA_max_plus_score_difference=abs(best_score-table[0]),
                inventory_classes=frozen.output_neurons,blank_in_targets=False,
                caveat='Original greedy retains incomplete phones and permits illegal paths; separate FSA decode enforces topology.',
                live_source_sha256=hashes,live_model_mass_method_sha256=hashlib.sha256(ast.get_source_segment(source.read_text(),fn).encode()).hexdigest())
    (HERE/'PARITY_PASS.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n');print(json.dumps(report),flush=True)


if __name__=='__main__':main()
