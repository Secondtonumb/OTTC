"""Continue successful recipe segments; final evaluation preserves its W&B identity."""
import json,subprocess
from pathlib import Path
HERE=Path(__file__).resolve().parent

def main():
    progress=json.loads((HERE/'training/progress.json').read_text())
    latest=max((HERE/'training').glob('step-*.pt'))
    complete=progress['epoch']>=10
    record=dict(checkpoint=str(latest),progress=progress,complete=complete)
    if complete:
        (HERE/'training/TRAIN_COMPLETE.json').write_text(json.dumps(record,indent=2)+'\n')
        if not (HERE/'evaluation/EVAL_COMPLETE.json').exists():
            record['evaluation_job']=subprocess.check_output(['/opt/pbs/bin/qsub',str(HERE/'evaluate.pbs')],text=True).strip()
    else:
        record['continuation_job']=subprocess.check_output(['/opt/pbs/bin/qsub',str(HERE/'train.pbs')],text=True).strip()
    (HERE/'training/segment_status.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record),flush=True)

if __name__=='__main__':main()

