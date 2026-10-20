"""Run frozen native train_one_epoch with an explicit true105 inventory adapter."""
import argparse
import ast
import hashlib
import inspect
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
NATIVE=HERE.parent/'zapformer_s3t2ss_native_20261012'
sys.path.insert(0,str(NATIVE))
import runtime
import train_s3t2 as original
sys.path.insert(0,str(HERE))
from monophone_model import Monophone
import torch


class ObservedMonophone(Monophone):
    """Observe the existing mass output without changing its loss or gradient."""
    def forward(self,features,lengths,*args,**kwargs):
        captured={}
        hook=self.weight_head.register_forward_hook(
            lambda module,inputs,output:captured.update(logits=output.detach().float().squeeze(-1)))
        try:loss,components=super().forward(features,lengths,*args,**kwargs)
        finally:hook.remove()
        with torch.no_grad(),torch.autocast(features.device.type,enabled=False):
            weights=captured['logits']
            valid=torch.arange(weights.shape[1],device=weights.device)[None]<lengths[:,None]
            mass=weights.masked_fill(~valid,-torch.inf).softmax(-1)
            entropy=-(mass*mass.clamp_min(1e-30).log()).sum(-1)
            components.update(mass_max_sum=mass.max(-1).values.sum(),
                              mass_effective_fraction_sum=(entropy.exp()/lengths).sum())
        return loss,components


def decorate_contract(contract):
    contract.update(topology='s1_t1',model_variant='phone105',target_states_per_phone=1,
                    head_dimension=105,phone_objective_scale=1.,forward_sum_weight=0.,
                    inventory_adaptation='Replace native313 head with true105; no three-state target expansion; mask control phone IDs1/2; archived original OTTC-JSD and learned mass unchanged',
                    monophone_source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest()
                        for p in [HERE/'train_native.py',HERE/'monophone_model.py',HERE/'monophone_metrics.py']})
    contract['adaptations']=[item.replace('MFA IPA105 / three-state313 head','native IPA105 / true105 monophone head')
                             for item in contract['adaptations']]


def main():
    # Inventory/parser/metric hooks adapt the task; the vendor native training
    # loop, optimizer, scheduler, augmentation and persistence remain untouched.
    tree=ast.parse(inspect.getsource(original))
    for node in ast.walk(tree):
        if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute) and node.func.attr=='add_argument':
            if node.args and isinstance(node.args[0],ast.Constant) and node.args[0].value=='--topology':
                for keyword in node.keywords:
                    if keyword.arg=='choices':keyword.value=ast.List(elts=[ast.Constant('s1_t1')],ctx=ast.Load())
                    elif keyword.arg=='default':keyword.value=ast.Constant('s1_t1')
        if isinstance(node,ast.BinOp) and isinstance(node.op,ast.Div) and isinstance(node.right,ast.Constant) and node.right.value=='topology_metrics.py':
            node.left=ast.Call(func=ast.Name(id='Path',ctx=ast.Load()),args=[ast.Constant(str(HERE))],keywords=[])
            node.right=ast.Constant('monophone_metrics.py')
    main_fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main')
    for index,node in enumerate(main_fn.body):
        if isinstance(node,ast.Assign) and any(isinstance(target,ast.Name) and target.id=='contract' for target in node.targets):
            main_fn.body.insert(index+1,ast.Expr(value=ast.Call(func=ast.Name(id='decorate_contract',ctx=ast.Load()),
                                                            args=[ast.Name(id='contract',ctx=ast.Load())],keywords=[])))
            break
    else:raise RuntimeError('Frozen native entry no longer has the expected contract assignment')
    ast.fix_missing_locations(tree)
    namespace=dict(__name__='monophone_native_entry',__file__=original.__file__,decorate_contract=decorate_contract)
    exec(compile(tree,original.__file__,'exec'),namespace)
    def model(objective,tiny=False,topology='s1_t1'):
        if objective!='jsd' or topology!='s1_t1':raise ValueError('This experiment is true105 original OTTC-JSD only')
        return ObservedMonophone('phone105',tiny=tiny)
    namespace['S3T2']=model
    tracking=namespace['Tracking']
    original_log=tracking.log
    def log(self,record,prefix='train'):
        if 'mass_max_sum' in record:
            record['mass_max_mean']=record['mass_max_sum']/record['utterances']
            record['mass_effective_frame_fraction_mean']=record['mass_effective_fraction_sum']/record['utterances']
        return original_log(self,record,prefix)
    tracking.log=log
    namespace['main']()


if __name__=='__main__':main()
