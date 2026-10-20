"""Audit explicit JSD wiring against live IF-MDD and a dense transport oracle."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import torch
from task_ottc import HERE, criterion, ottc_objective, decode, frame_masses
from ifmdd_losses import pad_flat_targets
from fsa_decode import graph, decode_fsa, INVENTORY
from utils.losses import ot_loss

ROOT=Path('/work/gm64/m64000/IF-MDD')


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
    return module


def dense_oracle(x, y, a, b):
    """Independent interval overlaps and standard JS equation, for validation only."""
    ca=torch.cat([a.new_zeros((len(a),1)),a.cumsum(1)],1)
    cb=torch.cat([b.new_zeros((len(b),1)),b.cumsum(1)],1)
    overlap=(torch.minimum(ca[:,1:,None],cb[:,None,1:])-torch.maximum(ca[:,:-1,None],cb[:,None,:-1])).clamp_min(0)
    p=x.softmax(-1).clamp_min(1e-12);q=y.clamp_min(1e-12)
    p=p/p.sum(-1,keepdim=True);q=q/q.sum(-1,keepdim=True)
    p=p[:,:,None,:];q=q[:,None,:,:];m=(p+q)/2
    costs=.5*(p*(p.log()-m.log())+q*(q.log()-m.log())).sum(-1)
    return (overlap*costs).sum()


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,default=HERE/'PARITY_PASS.json')
    ap.add_argument('--device',choices=['cpu','cuda'],default='cpu');args=ap.parse_args()
    torch.manual_seed(42);torch.set_num_threads(4)
    manifest=json.loads((HERE/'source_manifest.json').read_text())
    for name,r in manifest.items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==r['sha256']
        assert hashlib.sha256((HERE/'vendor'/name).read_bytes()).hexdigest()==r['runtime_sha256']
    live=load('live_ifmdd_topology',ROOT/'utils/subphonetic_topology.py')
    live_loss=load('live_ifmdd_transport',ROOT/'utils/losses/ot_loss.py')
    reference=live.SubphoneticTopologyLoss(str(HERE/'mfa_label_encoder.txt'),topology='s3_t2_star_star',loss_type='ottc').to(args.device)
    module=criterion().to(args.device);assert reference.labels==module.labels
    lengths=torch.tensor([31,23,17],device=args.device);tl=torch.tensor([5,3,2],device=args.device)
    tokens=torch.tensor([7,7,31,103,104,31,62,62,7,7],device=args.device);padded=pad_flat_targets(tokens,tl)
    records=[];original=ot_loss.batched_ottc_loss_bucketized
    for name,cost,uniform,extreme in [('standard_jsd','ifmdd_standard_jsd',False,False),('uniform','ifmdd_standard_jsd',True,False),('clamped','ifmdd_standard_jsd',False,True),('ce_regression','ce',False,False)]:
        x=torch.randn(3,31,module.output_neurons,device=args.device,requires_grad=True)
        if extreme:
            with torch.no_grad():x[0,0,:5]=torch.tensor([100.,-100.,float('nan'),float('inf'),float('-inf')],device=args.device)
        z=x.detach().clone().requires_grad_();w=torch.randn(3,31,device=args.device,requires_grad=True);v=w.detach().clone().requires_grad_()
        captures=[]
        def capture(**kw):
            result=original(**kw);captures.append(kw);return result
        ot_loss.batched_ottc_loss_bucketized=capture
        current=criterion(cost).to(args.device)
        actual=ottc_objective(current,x,lengths,tokens,tl,None if uniform else w)
        assert captures[0]['jsd']==(cost=='ifmdd_standard_jsd')
        def reference_cost(**kw):
            # Reference-only intervention: live topology has jsd=False hardcoded.
            # Change only that cost selector and call the live IF-MDD loss.
            kw['jsd']=cost=='ifmdd_standard_jsd'
            return live_loss.batched_ottc_loss_bucketized(**kw)
        ot_loss.batched_ottc_loss_bucketized=reference_cost
        try:
            expected=reference(z.log_softmax(-1),padded,lengths.float()/31,tl.float()/5,
                               logits=z,weights_logits=None if uniform else frame_masses(v,lengths))
        finally:ot_loss.batched_ottc_loss_bucketized=original
        g1=torch.autograd.grad(actual,(x,w) if not uniform else (x,),retain_graph=True)
        g2=torch.autograd.grad(expected,(z,v) if not uniform else (z,))
        torch.testing.assert_close(actual,expected,rtol=0,atol=0)
        for a,b in zip(g1,g2):torch.testing.assert_close(a,b,rtol=0,atol=0)
        oracle_difference=None
        if cost=='ifmdd_standard_jsd' and not extreme:
            inputs=captures[0]
            oracle=dense_oracle(inputs['x'],inputs['y'],inputs['a'],inputs['b'])
            torch.testing.assert_close(actual,oracle,rtol=1e-6,atol=1e-6)
            go=torch.autograd.grad(oracle,(x,w) if not uniform else (x,))
            for a,b in zip(g1,go):torch.testing.assert_close(a,b,rtol=1e-4,atol=2e-6)
            oracle_difference=float((actual-oracle).abs())
        assert torch.isfinite(actual) and all(torch.isfinite(g).all() for g in g1)
        if not uniform:assert g1[1].abs().sum()>0
        records.append(dict(case=name,cost=cost,loss=float(actual),live_loss_difference=0.,live_gradient_difference=0.,independent_dense_oracle_difference=oracle_difference))
    # Force the normal-call failure to exercise the uniform fallback selector.
    calls=[]
    def fail_once(**kw):
        calls.append(kw['jsd']);result=original(**kw)
        return (result[0]*float('nan'),*result[1:]) if len(calls)==1 else result
    ot_loss.batched_ottc_loss_bucketized=fail_once
    try:
        loss=ottc_objective(module,x.detach(),lengths,tokens,tl,w.detach())
    finally:ot_loss.batched_ottc_loss_bucketized=original
    assert calls==[True,True] and torch.isfinite(loss)
    assert module.last_k2_debug['ottc_fallback']=='uniform_frame_weights'
    b=INVENTORY['b'];states=module.phone_to_substates[int(module.content_to_phone[b])]
    vector,text=graph(module,device=args.device);fixtures=[]
    for path,hyp in [(states[:1]*2+states[1:2]*2+states[2:]*2,[b]),([states[0],states[2],states[0],states[2]],[b,b])]:
        scores=torch.full((len(path),module.output_neurons),-30.,device=args.device)
        scores[torch.arange(len(path),device=args.device),torch.tensor(path,device=args.device)]=0.
        greedy=decode(scores[None],torch.tensor([len(path)],device=args.device),module)[0]
        fsa,actual_path,_=decode_fsa(scores,vector)
        assert greedy==fsa==hyp and actual_path==path
        fixtures.append(dict(path=path,hypothesis=hyp))
    report=dict(status='passed',objective='explicit IF-MDD standard-JSD cost SPOTTC',device=args.device,
                cases=records,fallback_jsd_flags=calls,inventory_classes=module.output_neurons,
                decoder_fixtures=fixtures,reference_intervention='Live IF-MDD topology + change only transport jsd argument to True; live loss implementation unchanged',
                source_manifest=manifest,GPU_training_completed=False)
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
