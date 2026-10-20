"""Atomic per-epoch model snapshots, retained independently of resume rotation."""
import bootstrap
from bootstrap import HERE
import hashlib
import json
import os
from pathlib import Path
import uuid
from transformers import TrainerCallback


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def atomic_json(path,value):
    tmp=path.with_name(path.name+'.pending-'+uuid.uuid4().hex)
    tmp.write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n');tmp.replace(path)


def epoch_number(state):
    epoch=float(state.epoch or 0)
    return round(epoch) if epoch>=1 and abs(epoch-round(epoch))<1e-6 else None


class EpochArchive(TrainerCallback):
    def __init__(self,output,identity,offset=17840):
        self.output=Path(output);self.identity=identity;self.offset=offset

    def on_epoch_end(self,args,state,control,model=None,**kwargs):
        epoch=epoch_number(state)
        if epoch is None:return control
        target=self.output/'epoch_weights'/f'epoch-{epoch:03d}'
        if (target/'EPOCH_COMPLETE.json').exists():
            previous=json.loads((target/'EPOCH_COMPLETE.json').read_text())
            assert previous['stage_step']==state.global_step and previous['identity']==self.identity
        else:
            target.parent.mkdir(parents=True,exist_ok=True)
            temporary=target.with_name(target.name+'.pending-'+uuid.uuid4().hex)
            model.save_pretrained(temporary,safe_serialization=True)
            record=dict(stage_epoch=epoch,cumulative_epoch=40+epoch,stage_step=state.global_step,
                cumulative_optimizer_step=self.offset+state.global_step,identity=self.identity,
                weights_sha256=digest(temporary/'model.safetensors'),config_sha256=digest(temporary/'config.json'),
                model_only=True,atomic_completed_epoch=True)
            atomic_json(temporary/'EPOCH_COMPLETE.json',record)
            os.replace(temporary,target)
            print('EPOCH_WEIGHTS_SAVED',json.dumps(record),flush=True)
        control.should_save=True
        return control


class BestDev(TrainerCallback):
    def __init__(self,output,parent_checkpoint,parent_metric):
        self.output=Path(output)
        if not (self.output/'BEST_DEV.json').exists():
            atomic_json(self.output/'BEST_DEV.json',dict(stage_epoch=0,stage_step=0,
                checkpoint=str(parent_checkpoint),metric='eval_nosil_PER_percent',value=parent_metric,
                selection='development PER only; includes the unchanged Libri100 epoch40 baseline'))
        self.refresh_link()

    def refresh_link(self):
        record=json.loads((self.output/'BEST_DEV.json').read_text())
        temporary=self.output/('.best-dev.pending-'+uuid.uuid4().hex)
        temporary.symlink_to(record['checkpoint'],target_is_directory=True)
        os.replace(temporary,self.output/'best-dev')

    def on_evaluate(self,args,state,control,metrics=None,**kwargs):
        epoch=epoch_number(state)
        if epoch is None or not metrics:return
        archive=self.output/'epoch_weights'/f'epoch-{epoch:03d}'
        assert (archive/'EPOCH_COMPLETE.json').exists()
        atomic_json(archive/'development_metrics.json',metrics)
        previous=json.loads((self.output/'BEST_DEV.json').read_text())
        score=metrics['eval_nosil_PER_percent']
        if score<previous['value']:
            atomic_json(self.output/'BEST_DEV.json',dict(stage_epoch=epoch,stage_step=state.global_step,
                checkpoint=str(archive),metric='eval_nosil_PER_percent',value=score,
                selection='development PER only; includes unchanged parent baseline'))
            self.refresh_link()
