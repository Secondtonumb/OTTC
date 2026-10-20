"""Save resumable state before an allocation ends, keeping the 40-epoch schedule."""
import bootstrap
from bootstrap import HERE
import argparse
import json
import sys
import time
from transformers import TrainerCallback
import train
import run_exp


class Budget(TrainerCallback):
    def __init__(self,seconds):
        self.deadline=time.monotonic()+seconds;self.triggered=False

    def on_step_end(self,args,state,control,**kwargs):
        if time.monotonic()>=self.deadline:
            self.triggered=True;control.should_training_stop=True;control.should_save=True
        return control


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--objective',choices=list(train.OBJECTIVES),required=True)
    parser.add_argument('--max-wall-seconds',type=float,required=True)
    args=parser.parse_args();assert args.max_wall_seconds>0
    callback=Budget(args.max_wall_seconds)
    original=run_exp.create_trainer
    def bounded(*a,**k):
        trainer=original(*a,**k);trainer.add_callback(callback);return trainer
    run_exp.create_trainer=bounded
    sys.argv=['train.py','--objective',args.objective]
    try:train.main()
    except AssertionError:
        if not callback.triggered:raise
        output=train.model_directory(args.objective)
        state=json.loads((output/'trainer_state.json').read_text())
        checkpoint=run_exp.latest_checkpoint(output)
        assert checkpoint is not None and int(checkpoint.name.split('-')[1])==state['global_step']
        (HERE/f'PAUSED_{args.objective}.json').write_text(json.dumps(dict(epoch=state['epoch'],
            global_step=state['global_step'],checkpoint=str(checkpoint),
            reason='allocation time budget; same entry resumes optimizer/scheduler/W&B state'),indent=2)+'\n')
        print('RESUMABLE_TIME_LIMIT',args.objective,state['epoch'],state['global_step'],flush=True)
        raise SystemExit(75)


if __name__=='__main__':main()
