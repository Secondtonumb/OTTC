"""Validate full40 label coverage, masking, actual callbacks and exact resume."""
import json
from pathlib import Path
import tempfile

import torch
from transformers import TrainingArguments, set_seed
from reproduction.buckeye_frame_ag_20261027 import common
from reproduction.buckeye_frame_ag_20261027.train import Policy, QuietTrainer
from reproduction.timit_frame_ce_20261026.validate import Data, collate, Stop
from reproduction.timit_frame_divergences_20261026.validate import tiny, independent
from reproduction.timit_frame_divergences_20261026.model import frame_loss


class Run:
    def log(self, *args, **kwargs):
        pass


def resume_case(path, config, vocab, stop=False, resume=False):
    set_seed(42)
    model = tiny(config, vocab)
    path.mkdir(exist_ok=True)
    policy = Policy(path, config, [], vocab, None, None, float('inf'), Run())
    args = TrainingArguments(output_dir=str(path), per_device_train_batch_size=2,
        num_train_epochs=2, learning_rate=1e-4, warmup_steps=1, save_strategy='no',
        save_total_limit=1, logging_steps=1, report_to=[], dataloader_num_workers=0,
        remove_unused_columns=False, disable_tqdm=True, seed=42,
        save_safetensors=True, use_cpu=True)
    trainer = QuietTrainer(model=model, args=args, train_dataset=Data(),
        data_collator=collate, callbacks=[policy] + ([Stop()] if stop else []))
    trainer.train(resume_from_checkpoint=str(path/'checkpoint-1') if resume else None)
    return {k: v.detach().cpu().clone() for k,v in model.state_dict().items()}


def main():
    torch.set_num_threads(2)
    config = common.recipe(common.NAME)
    data, rows, vocab, processor, audit = common.load_data(config)
    assert [len(rows[s]) for s in ('train','dev','test')] == [18221,6417,6839]
    # Gaps stay ignored; unknown labels and overlapping gold intervals fail.
    assert common.project([['a',.02,.1]], {'a':1}, 1600).tolist()==[-100]+[1]*7
    for intervals in [[['unknown',0.,.1]], [['a',0.,.04],['a',.03,.1]]]:
        try:
            common.project(intervals, {'a':1}, 1600)
        except (ValueError,AssertionError):
            pass
        else:
            raise AssertionError('Invalid annotation accepted')
    actual = common.Collator(processor)([data['train'][i] for i in range(2)])
    assert actual['frame_targets'].shape[0]==2
    for i in range(2):
        expected=common.project(rows['train'][i]['reference'],vocab,
                               round(rows['train'][i]['duration']*16000))
        assert actual['frame_targets'][i,:len(expected)].tolist()==expected.tolist()
    logits=torch.tensor([[[2.,0.,-1.],[0.,2.,-1.]],[[1.,2.,0.],[1000.,-1000.,0.]]],requires_grad=True)
    targets=torch.tensor([[0,1],[1,-100]])
    loss=frame_loss(logits,targets,torch.ones_like(targets),'ag')
    expected=independent(logits[0],targets[0],'ag').mean()+independent(logits[1,:1],targets[1,:1],'ag').mean()
    torch.testing.assert_close(loss,expected)
    loss.backward()
    assert torch.equal(logits.grad[1,1],torch.zeros(3))
    with tempfile.TemporaryDirectory(prefix='buckeye_frame_ag_') as temporary:
        path=Path(temporary)
        full=resume_case(path/'continuous',config,vocab)
        resume_case(path/'resumed',config,vocab,stop=True)
        resumed=resume_case(path/'resumed',config,vocab,resume=True)
        difference=max(float((full[k]-resumed[k]).abs().max()) for k in full)
        assert difference==0.
        for epoch in (1,2):
            assert (path/'continuous/epoch_weights'/f'epoch-{epoch:03d}.pt').exists()
        from reproduction.buckeye_frame_ag_20261027.evaluate import average
        average([path/'continuous/epoch_weights'/f'epoch-{epoch:03d}.pt' for epoch in (1,2)],
                path/'average.pt',config)
        saved=torch.load(path/'average.pt',weights_only=False)
        assert all(v.dtype==torch.float32 for v in saved['model'].values() if v.is_floating_point())
    common.write(common.HERE/'CPU_PASS.json',dict(passed=True,source_sha256=common.source_hashes(),
        data=audit,coverage=processor.frame_ce_coverage,choice=config['divergence_choice'],
        all_timelines_validated=True,masking_and_divergence_verified=True,
        exact_resume_max_difference=difference,epoch_weights_verified=True,FP32_average_verified=True))
    print('CPU_PASS',audit['Buckeye_counts'],flush=True)


if __name__=='__main__':
    main()
