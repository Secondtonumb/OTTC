"""Replay upstream Charsiu values/gradients and unchanged OTTC against frozen source."""
import argparse
import ast
import json
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from continue_jsd import NATIVE,sha,torch
from forward_sum import ConfigurableForwardSum,position_log_probs
from charsiu_forward_sum import ForwardSumLoss
import task_adapter
from task_adapter import ConfigurableS3T2
from task_model import S3T2
from k2_determinism import enable


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--gpu',action='store_true');args=parser.parse_args()
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True);torch.autograd.set_multithreading_enabled(False)
    config=json.loads((HERE/'loss_config.json').read_text())
    source=ast.parse((NATIVE/'task_model.py').read_text());adapted=ast.parse((HERE/'task_adapter.py').read_text())
    for name in ('expanded','padded','weights','fw','tm','acoustic_ids'):
        def find(tree):
            return next(n for n in ast.walk(tree) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id==name or isinstance(t,ast.Tuple) and any(isinstance(x,ast.Name) and x.id==name for x in t.elts) for t in n.targets))
        assert ast.dump(find(source),include_attributes=False)==ast.dump(find(adapted),include_attributes=False),name
    original_source=Path(json.loads((HERE/'charsiu_source.json').read_text())['source_path'])
    assert sha(original_source)==json.loads((HERE/'charsiu_source.json').read_text())['source_sha256']
    assert sha(HERE/'charsiu_forward_sum.py')==json.loads((HERE/'charsiu_source.json').read_text())['extracted_sha256']
    torch.manual_seed(42)
    x=torch.randn(2,12,313,requires_grad=True)
    lengths=torch.tensor([12,9]);tokens=torch.tensor([3,3,4,5,6]);token_lengths=torch.tensor([3,2])
    actual=ConfigurableForwardSum(config)(x,lengths,tokens,token_lengths,'s3_t2_star_star')
    score=position_log_probs(x,tokens,token_lengths)
    assert torch.equal(score[0,:,:,0],score[0,:,:,1]),'Repeated phones lost separate positions'
    source_class=next(n for n in ast.parse(original_source.read_text()).body if isinstance(n,ast.ClassDef) and n.name=='ForwardSumLoss')
    ns=dict(torch=torch,nn=torch.nn,F=torch.nn.functional)
    exec(compile(ast.Module(body=[source_class],type_ignores=[]),str(original_source),'exec'),ns)
    expected=ns['ForwardSumLoss']()(score,token_lengths,lengths)
    ga=torch.autograd.grad(actual,x,retain_graph=True)[0];gb=torch.autograd.grad(expected,x)[0]
    assert torch.equal(actual,expected) and torch.equal(ga,gb)
    assert torch.isfinite(ga).all() and ga.abs().sum()>0
    csum=config|dict(forward_sum_reduction='sum')
    value=ConfigurableForwardSum(csum)(x,lengths,tokens,token_lengths,'s3_t2_star_star')
    torch.testing.assert_close(value,actual*2,rtol=1e-6,atol=1e-6)
    invalid=config|dict(forward_sum_normalization='target',forward_sum_reduction='mean')
    zero=ConfigurableForwardSum(invalid)(x[:1],torch.tensor([2]),tokens[:3],torch.tensor([3]),'s3_t2_star_star')
    assert zero.item()==0
    results=dict(charsiu_loss_and_gradient='bitwise equal on CPU',repeated_phone_occurrences='distinct positions',
                 configurable_reduction='sum equals batch size times mean',zero_infinity='matches original',
                 ottc_input_statements='AST identical to frozen adapter; upstream cost flag configurable')
    if args.gpu:
        enable();torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
        legacy=config|dict(ottc_cost='jsd_upstream',forward_sum_weight=1.,forward_sum_kind='native_topology',forward_sum_normalization='input',forward_sum_reduction='sum')
        base=S3T2('jsd',tiny=True,topology='s3_t2_star_star').cuda().eval()
        candidate=ConfigurableS3T2('jsd',tiny=True,topology='s3_t2_star_star',loss_config=legacy).cuda().eval()
        candidate.load_state_dict(base.state_dict(),strict=True)
        features=torch.randn(2,120,80,device='cuda');lens=torch.tensor([120,100],device='cuda')
        tok=tokens.cuda();tl=token_lengths.cuda()
        a,ac=base(features,lens,tok,tl);b,bc=candidate(features,lens,tok,tl)
        assert torch.equal(a,b) and torch.equal(ac['jsd_sum'],bc['jsd_sum'])
        ag=torch.autograd.grad(a,tuple(base.parameters()),allow_unused=True)
        bg=torch.autograd.grad(b,tuple(candidate.parameters()),allow_unused=True)
        for u,v in zip(ag,bg):
            assert u is None and v is None or u is not None and v is not None and torch.equal(u,v)
        assert base.state_dict().keys()==candidate.state_dict().keys()
        candidate.loss_config=config;candidate.forward_sum=ConfigurableForwardSum(config)
        c,cc=candidate(features,lens,tok,tl)
        assert torch.equal(ac['jsd_sum'],cc['jsd_sum'])
        grads=torch.autograd.grad(c,tuple(candidate.parameters()),allow_unused=True)
        assert all(g is None or bool(torch.isfinite(g).all()) for g in grads)
        assert all(g is not None for g in grads),'Expected all existing parameters to participate'
        original_loss=task_adapter.batched_ottc_loss_bucketized
        captured={}
        def capture(x,y,a,b,**kwargs):
            captured.update(x=x,y=y,a=a,b=b,kwargs=kwargs)
            return original_loss(x,y,a,b,**kwargs)
        task_adapter.batched_ottc_loss_bucketized=capture
        ce_config=config|dict(ottc_cost='ce',forward_sum_weight=0.)
        candidate.loss_config=ce_config
        ce,components=candidate(features,lens,tok,tl)
        assert captured['kwargs']['jsd'] is False
        assert components['forward_sum_raw'].item()==0
        from archived_ot_loss import ottc_loss_reference_implementation
        references=[]
        for i in range(features.shape[0]):
            n=int(captured['kwargs']['amask'][i].sum());m=int(captured['kwargs']['bmask'][i].sum())
            references.append(ottc_loss_reference_implementation(captured['a'][i,:n],captured['b'][i,:m],captured['x'][i,:n],captured['y'][i,:m]))
        reference=torch.stack(references).sum()
        torch.testing.assert_close(ce,reference,rtol=1e-5,atol=1e-5)
        ga=torch.autograd.grad(ce,tuple(candidate.parameters()),retain_graph=True,allow_unused=True)
        gb=torch.autograd.grad(reference,tuple(candidate.parameters()),allow_unused=True)
        for u,v in zip(ga,gb):
            assert (u is None)==(v is None)
            if u is not None:torch.testing.assert_close(u,v,rtol=2e-4,atol=2e-5)
        candidate.loss_config=config|dict(ottc_cost='ce',forward_sum_weight=1.)
        combined,combined_components=candidate(features,lens,tok,tl)
        torch.testing.assert_close(combined,combined_components['ottc_ce_sum']+combined_components['forward_sum_raw'])
        candidate.loss_config=config|dict(ottc_cost='jsd_upstream',forward_sum_weight=0.)
        only,only_components=candidate(features,lens,tok,tl)
        assert torch.equal(only,ac['jsd_sum']) and only_components['forward_sum_raw'].item()==0
        task_adapter.batched_ottc_loss_bucketized=original_loss
        results.update(native_legacy_total_loss_and_gradients='bitwise equal',charsiu_adapter_jsd='bitwise equal to original',
                       new_total_gradient='finite; existing parameters participate',model_state_keys='unchanged',
                       ottc_ce_loss_and_gradients='match upstream transport reference; jsd=False',
                       only_arms='forward-sum exactly zero; no framewise CE',four_models='identical parameter structure')
        (HERE/'CRITERION_PASS.json').write_text(json.dumps(dict(results=results,script_sha256={p.name:sha(p) for p in HERE.glob('*.py')}),indent=2)+'\n')
    print(json.dumps(results))


if __name__=='__main__':main()
