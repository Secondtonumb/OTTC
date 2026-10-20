#!/usr/bin/env python3
"""Submit recoverable 20-minute units, with dev-selection dependencies."""
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import subprocess

HERE=Path(__file__).resolve().parent


def save(state):
    temp=HERE/'dispatch.json.tmp'
    temp.write_text(json.dumps(state,indent=2)+'\n');temp.replace(HERE/'dispatch.json')


def main():
    file=HERE/'dispatch.json'
    old=json.loads(file.read_text())
    if old.get('unit_walltime')=='00:20:00':state=old
    else:
        state=dict(submitted_at_utc=datetime.now(timezone.utc).isoformat(),
            mode='independent single-GPU 20-minute units with afterok dev dependencies',queue='short-g',
            unit_walltime='00:20:00',nodes_per_job=1,script=str(HERE/'run_one.pbs'),
            replaced_queued_jobs=old['jobs'],replaced_jobs_cancelled=True,original_dispatch='dispatch_2h_original.json',
            jobs={},task_jobs={},dependencies={},submissions_complete=False,
            policies=['last3','dev_fa_best3','dev_per_best3'],original_models_and_results_preserved=True)
        save(state)
    codes={'jsd_upstream':'off','ce':'ce','jsd_standard':'std'}
    tasks={'last3':'last3','dev_a':'devA','dev_b':'devB','dev_fa_best3':'bestFA','dev_per_best3':'bestPER'}
    # Last3 first, so it can fill a short available slot immediately.
    order=[(o,'last3') for o in codes]+[(o,t) for o in codes for t in ['dev_a','dev_b']]+[(o,t) for o in codes for t in ['dev_fa_best3','dev_per_best3']]
    for objective,task in order:
        known=state['task_jobs'].setdefault(objective,{})
        if task in known:continue
        command=['/opt/pbs/bin/qsub','-N',f'l100_{codes[objective]}_{tasks[task]}',
            '-l','walltime=00:20:00','-v',f'LIBRI_OBJECTIVE={objective},LIBRI_AVG_TASK={task}',
            '-o',str(HERE/f'pbs_{objective}_{task}.log')]
        if task.startswith('dev_') and task not in ['dev_a','dev_b']:
            required=[known['dev_a'],known['dev_b']]
            command+=['-W','depend=afterok:'+':'.join(required)]
            state['dependencies'][objective+'_'+task]=required
        command.append(str(HERE/'run_one.pbs'))
        response=subprocess.run(command,text=True,capture_output=True)
        if response.returncode:
            state['submission_error']=dict(objective=objective,task=task,stderr=response.stderr)
            save(state);raise RuntimeError(response.stderr)
        job=response.stdout.strip();assert re.fullmatch(r'\d+\.\w+',job), response.stdout
        known[task]=job
        if task=='last3':state['jobs'][objective]=job
        save(state);print('SUBMITTED',objective,task,job,flush=True)
    state['submissions_complete']=True;state.pop('submission_error',None);save(state)


if __name__=='__main__':main()
