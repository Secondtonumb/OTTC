"""Replay IF-MDD CE/JSD and compare graph FS to an independent arc dynamic program."""
import argparse, hashlib, json
import torch
from task_ottc import HERE, criterion, ottc_objective, TopologyOTTC
from forward_sum import IFMDDForwardSum
from fsa_decode import INVENTORY


def arc_dp(module, lp, ids):
    n,src,dst,labels,final=module._arc_tensors_for_phone_ids(ids,lp.device)
    arcs=list(zip(src.tolist(),dst.tolist(),labels.tolist()))
    alpha={0:lp.new_zeros(())}
    for frame in lp:
        buckets={}
        for a,b,label in arcs:
            if a in alpha:buckets.setdefault(b,[]).append(alpha[a]+frame[label])
        alpha={b:torch.logsumexp(torch.stack(values),0) for b,values in buckets.items()}
    return -alpha[final]


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--gpu',action='store_true');a=ap.parse_args()
    device='cuda' if a.gpu else 'cpu';torch.set_num_threads(4);torch.manual_seed(42)
    if a.gpu:
        from k2_determinism import enable
        enable();torch.use_deterministic_algorithms(True);torch.autograd.set_multithreading_enabled(False)
    config=json.loads((HERE/'configs/jsd_fs.json').read_text());config['forward_sum_output_beam']=1000
    fs=IFMDDForwardSum(config).to(device)
    tokens=torch.tensor([INVENTORY['b'],INVENTORY['b'],103,104,INVENTORY['b']],device=device)
    tl=torch.tensor([3,2],device=device);lens=torch.tensor([13,10],device=device)
    x=torch.randn(2,13,303,device=device,requires_grad=True);z=x.detach().clone().requires_grad_()
    loss,raw=fs(x,lens,tokens,tl);lp=z.log_softmax(-1)
    targets=[tokens[:3].tolist(),tokens[3:].tolist()]
    expected=torch.stack([arc_dp(fs,lp[i,:int(lens[i])],[int(fs.content_to_phone[p]) for p in ids]) for i,ids in enumerate(targets)])
    torch.testing.assert_close(raw,expected.sum(),rtol=1e-6,atol=1e-5)
    torch.testing.assert_close(loss,(expected/lens).sum(),rtol=1e-6,atol=1e-5)
    g=torch.autograd.grad(loss,x)[0];h=torch.autograd.grad((expected/lens).sum(),z)[0]
    torch.testing.assert_close(g,h,rtol=1e-4,atol=1e-6)
    assert torch.isfinite(g).all()
    # Same seed and inventory give identical trainable initialization in every arm.
    fingerprints={};records={}
    for arm in ('jsd_fs','jsd_only','ce_fs','ce_only'):
        torch.manual_seed(42);cfg=json.loads((HERE/'configs'/f'{arm}.json').read_text())
        net=TopologyOTTC(tiny=True,loss_config=cfg)
        digest=hashlib.sha256()
        for name,value in net.state_dict().items():digest.update(name.encode());digest.update(value.numpy().tobytes())
        fingerprints[arm]=digest.hexdigest();del net
        logits=x.detach().clone().requires_grad_();weights=torch.zeros(2,13,device=device,requires_grad=True)
        ot=ottc_objective(criterion(cfg['ottc_cost']).to(device),logits,lens,tokens,tl,weights)
        f,_=fs(logits,lens,tokens,tl) if cfg['forward_sum_weight'] else (logits.new_zeros(()),None)
        total=ot+f
        gradients=torch.autograd.grad(total,(logits,weights));assert all(torch.isfinite(v).all() for v in gradients)
        records[arm]=dict(ottc=float(ot),fs=float(f),total=float(total))
    assert len(set(fingerprints.values()))==1
    assert records['jsd_fs']['ottc']==records['jsd_only']['ottc']
    assert records['ce_fs']['ottc']==records['ce_only']['ottc']
    assert records['jsd_fs']['fs']==records['ce_fs']['fs']
    report=dict(status='passed',device=device,graph_loss_difference=float((raw-expected.sum()).abs()),graph_gradient_max_difference=float((g-h).abs().max()),initialization=fingerprints,arms=records,source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in HERE.glob('*.py')})
    (HERE/('CRITERION_GPU_PASS.json' if a.gpu else 'CRITERION_CPU_PASS.json')).write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))

if __name__=='__main__':main()
