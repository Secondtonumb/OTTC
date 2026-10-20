"""Continue the existing ten-epoch MFA configuration after successful checkpointing."""
import json
import subprocess
import sys
from pathlib import Path

import torch

SUITE = Path(__file__).resolve().parent


def main():
    arm = sys.argv[1]
    assert arm in ['ctc', 'ottc', 'spottc', 'film_k4']
    directory = SUITE/'mfa_models'/arm
    checkpoint = max(directory.glob('step-*.pt'))
    state = torch.load(checkpoint, map_location='cpu', weights_only=False)
    progress = state['progress']
    del state
    complete = progress['epoch'] >= 10
    result = {'arm': arm, 'checkpoint': str(checkpoint), 'progress': progress, 'complete': complete}
    if complete:
        (directory/'TRAIN_COMPLETE.json').write_text(json.dumps(result, indent=2)+'\n')
        evaluated=SUITE/'mfa_evaluation'/arm/('step-'+str(progress['global_step']))/'EVAL_COMPLETE.json'
        if not evaluated.exists():
            result['evaluation_job']=subprocess.check_output(
                ['qsub','-v','ARM='+arm+',MFA_EVAL_CHECKPOINT='+str(checkpoint),
                 '-o',str(directory/'final_evaluation_job.log'),str(SUITE/'mfa_evaluate_final.pbs')],text=True).strip()
    else:
        command=['qsub','-N','mfa_full_'+arm,'-v','ARM='+arm,'-o',str(directory/'continuation_job.log')]
        if not (SUITE/'MFA_EVAL_GATE_PASS.json').exists():
            evaluation=json.loads((SUITE/'submission_status.json').read_text())['evaluation_updates']['mfa_inference_gate_job']
            command+=['-W','depend=afterok:'+evaluation]
        command.append(str(SUITE/'mfa_train_optimized.pbs'))
        result['continuation_job']=subprocess.check_output(command,text=True).strip()
    (directory/'segment_status.json').write_text(json.dumps(result, indent=2)+'\n')
    from summarize_results import main as summarize
    summarize()
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
