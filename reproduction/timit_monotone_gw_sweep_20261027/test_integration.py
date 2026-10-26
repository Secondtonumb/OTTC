from dataclasses import asdict
import pytest
import torch
from accelerate.utils.operations import convert_to_fp32
from reproduction.ottc_temporal_smoothing_20261021.test_smoothing import tiny_encoder,cpu_loss_allocations,DeviceOTTC
from ottc.models.sequence.wav2vec2_ottc import wvOTTCConfig
from ottc.models.sequence.wav2vec2_ottc_monotone_gw import MonotoneGWTransportMixin
from ottc.models.sequence.wav2vec2_ottc_gw import GWTransportMixin as HistoricalGW
from ottc.ops_loss.gw_transport import GWOptions


class TinyGW(MonotoneGWTransportMixin,DeviceOTTC):pass
class OldAG(HistoricalGW,DeviceOTTC):pass


def config(path,cost,weight=.1):
    return wvOTTCConfig(wav2vec2_path=str(path),vocab_size=5,bins_metric=cost,forward_type='regular',
                       wav2vec2_output_dropout=0.,gw_options=asdict(GWOptions(mode='monotone',weight=weight)))


@pytest.mark.parametrize('cost',['crossentropy','ag','jsd_standard','jdiv'])
@pytest.mark.parametrize('weight',[0.,.1])
def test_cost_preservation_gradient_and_inference(tiny_encoder,cost,weight):
    torch.manual_seed(8)
    cfg=config(tiny_encoder,cost,weight)
    model=TinyGW(cfg).eval();base=DeviceOTTC(cfg).eval();base.load_state_dict(model.state_dict())
    samples=torch.randn(2,2400);mask=torch.ones(2,2400,dtype=torch.long);mask[1,1700:]=0
    labels=torch.tensor([[1,2,1],[2,3,-100]])
    with cpu_loss_allocations():
        actual=model(samples,attention_mask=mask,labels=labels)
        expected=base(samples,attention_mask=mask,labels=labels)
        torch.testing.assert_close(actual.logits,expected.logits)
        torch.testing.assert_close(model(samples,attention_mask=mask).logits,actual.logits)
        assert isinstance(convert_to_fp32(actual).loss,torch.Tensor)
        if weight==0:assert torch.equal(actual.loss,expected.loss)
        else:
            assert actual.loss>expected.loss
            # Plan regularizer has no direct phone-head gradient: preserve
            # the base criterion, while adding a gradient through frame mass.
            ga=torch.autograd.grad(actual.loss,model.lm_head.weight,retain_graph=True)[0]
            gb=torch.autograd.grad(expected.loss,base.lm_head.weight)[0]
            torch.testing.assert_close(ga,gb)
        actual.loss.backward()
        for head in (model.lm_head,model.lm_weight):
            assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in head.parameters())
            assert sum(float(p.grad.square().sum()) for p in head.parameters())>0


def test_ag_matches_completed_recipe_value_and_gradient(tiny_encoder):
    torch.manual_seed(11);cfg=config(tiny_encoder,'ag')
    new=TinyGW(cfg).eval();old=OldAG(cfg).eval();old.load_state_dict(new.state_dict())
    x=torch.randn(2,2400);mask=torch.ones_like(x,dtype=torch.long);labels=torch.tensor([[1,2,1],[2,3,-100]])
    with cpu_loss_allocations():
        a=new(x,attention_mask=mask,labels=labels).loss
        b=old(x,attention_mask=mask,labels=labels).loss
        torch.testing.assert_close(a,b,rtol=0,atol=0)
        a.backward();b.backward()
    for (n,p),(n2,q) in zip(new.named_parameters(),old.named_parameters()):
        assert n==n2
        if p.grad is not None:torch.testing.assert_close(p.grad,q.grad,rtol=1e-6,atol=1e-7)


@pytest.mark.parametrize('cost',['crossentropy','ag','jsd_standard','jdiv'])
def test_actual_bf16_trainer(tiny_encoder,tmp_path,cost):
    from transformers import TrainingArguments
    from reproduction.timit_monotone_gw_sweep_20261027.train import QuietTrainer
    model=TinyGW(config(tiny_encoder,cost));before=model.lm_head.weight.detach().clone()
    items=[dict(input_values=torch.randn(2400),attention_mask=torch.ones(2400,dtype=torch.long),labels=torch.tensor([1,2,1])) for _ in range(4)]
    def collate(rows):
        return dict(microbatches=[{k:torch.stack([r[k] for r in rows]) for k in rows[0]}],logical_batch_size=len(rows))
    args=TrainingArguments(output_dir=str(tmp_path/'trainer'),use_cpu=True,bf16=True,per_device_train_batch_size=2,
                          max_steps=2,save_strategy='no',report_to=[],remove_unused_columns=False,disable_tqdm=True)
    trainer=QuietTrainer(model=model,args=args,train_dataset=items,data_collator=collate)
    with cpu_loss_allocations():result=trainer.train()
    assert result.global_step==2 and torch.isfinite(torch.tensor(result.training_loss))
    assert not torch.equal(before,model.lm_head.weight)
