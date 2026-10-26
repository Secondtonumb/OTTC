"""Full40 frames, four objective formulas, CPU gradients and exact resume."""
import tempfile
from pathlib import Path
import torch
import torch.nn.functional as F
from transformers import TrainingArguments,set_seed
from reproduction.buckeye_frame_sweep_20261027 import common
from reproduction.buckeye_frame_sweep_20261027.model import frame_loss
from reproduction.buckeye_frame_sweep_20261027.train import Policy,QuietTrainer
from reproduction.timit_frame_ce_20261026.validate import Data,collate,Stop,TinyLoader
from reproduction.timit_frame_divergences_20261026.validate import independent
from reproduction.timit_frame_divergences_20261026.profile_recipe import gradient_checks
import ottc.models.sequence.wav2vec2_ottc as upstream


class Run:
    def log(self,*args,**kwargs):
        pass


def tiny(config,vocab):
    previous=upstream.Wav2Vec2Model
    upstream.Wav2Vec2Model=TinyLoader
    try:
        model,_=common.build_model(config,vocab,2)
    finally:
        upstream.Wav2Vec2Model=previous
    model.dropout.p=0.
    return model


def resume_case(path,config,vocab,stop=False,resume=False):
    set_seed(42)
    model=tiny(config,vocab)
    path.mkdir(exist_ok=True)
    policy=Policy(path,config,[],vocab,None,None,float('inf'),Run())
    args=TrainingArguments(output_dir=str(path),per_device_train_batch_size=2,
        num_train_epochs=2,learning_rate=1e-4,warmup_steps=1,save_strategy='no',
        save_total_limit=1,logging_steps=1,report_to=[],dataloader_num_workers=0,
        remove_unused_columns=False,disable_tqdm=True,seed=42,save_safetensors=True,use_cpu=True)
    trainer=QuietTrainer(model=model,args=args,train_dataset=Data(),data_collator=collate,
        callbacks=[policy]+([Stop()] if stop else []))
    trainer.train(resume_from_checkpoint=str(path/'checkpoint-1') if resume else None)
    return {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}


def main():
    torch.set_num_threads(2)
    data,rows,vocab,processor,audit=common.load_data(common.recipe('frame_ag__seed42'))
    assert [len(rows[s]) for s in ('train','dev','test')]==[18221,6417,6839]
    configs={name:common.recipe(name) for name in common.matrix()}
    assert len(configs)==12 and len({(c['cost'],c['seed']) for c in configs.values()})==12
    assert all(c['global_batch']==64 and c['epochs']==40 for c in configs.values())
    assert common.project([['a',.02,.1]],{'a':1},1600).tolist()==[-100]+[1]*7
    for bad in [[['unknown',0.,.1]],[['a',0.,.04],['a',.03,.1]]]:
        try:
            common.project(bad,{'a':1},1600)
        except (ValueError,AssertionError):
            pass
        else:
            raise AssertionError('Unknown phone or overlap accepted')
    batch=common.Collator(processor)([data['train'][i] for i in (0,1)])
    for i in (0,1):
        reference=common.project(rows['train'][i]['reference'],vocab,round(rows['train'][i]['duration']*16000))
        assert batch['frame_targets'][i,:len(reference)].tolist()==reference.tolist()
    checks={}
    for cost in common.COSTS:
        config=configs[f'frame_{cost}__seed42']
        logits=torch.tensor([[[2.,0.,-1.],[0.,2.,-1.]],[[1.,2.,0.],[1000.,-1000.,0.]]],requires_grad=True)
        targets=torch.tensor([[0,1],[1,-100]])
        value=frame_loss(logits,targets,torch.ones_like(targets),cost)
        if cost=='ce':
            expected=F.cross_entropy(logits[0],targets[0])+F.cross_entropy(logits[1,:1],targets[1,:1])
        else:
            expected=independent(logits[0],targets[0],cost).mean()+independent(logits[1,:1],targets[1,:1],cost).mean()
        torch.testing.assert_close(value,expected,atol=2e-6,rtol=2e-6)
        actual_gradient,=torch.autograd.grad(value,logits,retain_graph=True)
        expected_gradient,=torch.autograd.grad(expected,logits)
        torch.testing.assert_close(actual_gradient,expected_gradient,atol=2e-6,rtol=2e-6)
        assert torch.equal(actual_gradient[1,1],torch.zeros(3))
        set_seed(42)
        model=tiny(config,vocab)
        real=collate([Data()[0],Data()[1]])['microbatches'][0]
        result=model(**real)
        assert result.logits.shape==(2,30,42) and torch.isfinite(result.loss)
        result.loss.backward()
        gradients=gradient_checks(model)
        model.eval()
        with torch.inference_mode():
            a=model(input_values=real['input_values'],attention_mask=real['attention_mask']).logits
            b=model(input_values=real['input_values'],attention_mask=real['attention_mask'],
                    frame_targets=torch.zeros_like(real['frame_targets'])).logits
        assert torch.equal(a,b)
        with tempfile.TemporaryDirectory(prefix=f'buckeye_frame_{cost}_') as temporary:
            path=Path(temporary)
            full=resume_case(path/'continuous',config,vocab)
            resume_case(path/'resumed',config,vocab,stop=True)
            resumed=resume_case(path/'resumed',config,vocab,resume=True)
            delta=max(float((full[k]-resumed[k]).abs().max()) for k in full)
            assert delta==0.
            members=[path/'continuous/epoch_weights'/f'epoch-{e:03d}.pt' for e in (1,2)]
            assert all(p.exists() for p in members)
            from reproduction.buckeye_frame_sweep_20261027.evaluate import average
            average(members,path/'average.pt',config)
            saved=torch.load(path/'average.pt',weights_only=False)
            assert all(v.dtype==torch.float32 for v in saved['model'].values() if v.is_floating_point())
            # Exercise the actual dev callback using the newly committed epoch.
            policy=Policy(path/'continuous',config,[],vocab,None,None,float('inf'),Run())
            from transformers import TrainerState,TrainerControl
            policy.on_evaluate(None,TrainerState(epoch=2.,global_step=4),TrainerControl(),
                model=model,metrics={'eval_nosil_PER_percent':20.,'eval_loss':float(value)})
            assert (path/'continuous/best_dev.pt').resolve()==members[-1].resolve()
        checks[cost]=dict(loss=float(value),independent_loss_and_gradient_match=True,
            padding_ignored=True,gradient_groups=gradients,audio_only_inference=True,
            exact_resume_max_difference=delta,epoch_weights_verified=True,FP32_average_verified=True,
            actual_dev_selection_callback_verified=True)
    common.write(common.HERE/'CPU_PASS.json',dict(passed=True,source_sha256=common.source_hashes(),
        data=audit,coverage=processor.frame_ce_coverage,checks=checks,recipes=list(configs),
        all_timelines_validated=True,independent_objectives=True))
    print('CPU_PASS:12 recipes;4 objectives;all full40 timelines;exact resume;FP32 averages',flush=True)


if __name__=='__main__':
    main()
