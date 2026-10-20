#!/usr/bin/env python3
"""Resume remaining evaluations in one-at-a-time interactive allocations."""
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time

import pexpect

HERE=Path(__file__).resolve().parent
OBJECTIVES=['jsd_upstream','ce','jsd_standard']
TASKS=['last3','dev_a','dev_b','dev_fa_best3','dev_per_best3']


def now():return datetime.now(timezone.utc).isoformat()


def save(path,value):
    temp=path.with_name(path.name+'.tmp');temp.write_text(json.dumps(value,indent=2)+'\n');temp.replace(path)


def pending(task):return [o for o in OBJECTIVES if not (HERE/'short_tasks'/o/(task+'_COMPLETE.json')).exists()]


def ready_units():
    units=[]
    for task in TASKS:
        for o in pending(task):
            if task in ['dev_fa_best3','dev_per_best3'] and any(o in pending(t) for t in ['dev_a','dev_b']):continue
            units.append(dict(objective=o,task=task))
    return units


def wait_release(job,state,file):
    if not job:return
    while True:
        active=subprocess.run(['/opt/pbs/bin/qselect','-u','m64000'],capture_output=True,text=True,check=True).stdout.split()
        if job not in active:return
        state.update(status='Waiting for previous interactive allocation to release',waiting_for_job=job,updated_at_utc=now())
        save(file,state)
        time.sleep(10)


def request(arguments,log,state,file):
    while True:
        child=pexpect.spawn('/opt/pbs/bin/qsub',arguments,encoding='utf-8',timeout=60)
        child.logfile_read=log
        match=child.expect([r'waiting for job (\d+\.\w+) to start',pexpect.EOF])
        if match==0:return child,child.match.group(1)
        error=str(child.before);child.close()
        if 'njobs_int-g' not in error:raise RuntimeError('Interactive request failed: '+error)
        state.update(status='PBS interactive slot still releasing; retrying',last_request_error=error.strip(),updated_at_utc=now())
        state['request_retries']=state.get('request_retries',0)+1;save(file,state)
        print('INTERACTIVE_SLOT_RETRY',now(),flush=True)
        time.sleep(10)


def main():
    lock=(HERE/'.interactive_driver.lock').open('w')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    file=HERE/'interactive_driver_state.json'
    state=json.loads(file.read_text()) if file.exists() else dict(started_at_utc=now(),queue='interact-g',walltime={'1node':'00:20:00','8nodes':'00:10:00'},
        per_user_interactive_limit=1,objectives=OBJECTIVES,allocations=[],complete=False,
        mode='One independent unit per GPU; interactive queue provides 1-node or 8-node allocations')
    state.setdefault('restarts',[]).append(now());state['driver_pid']=os.getpid();save(file,state)
    if 'LIBRI_INTERACTIVE_NODES' in os.environ:
        state['preferred_node_count']=int(os.environ['LIBRI_INTERACTIVE_NODES'])
    active_job=os.environ.get('LIBRI_ACTIVE_INTERACTIVE_JOB',state.get('job',''))
    wait_release(active_job,state,file)
    attempts=state.setdefault('attempts',{})
    round_id=max([0]+[int(p.stem.split('pool')[1].split('_')[0]) for p in HERE.glob('interactive_driver_pool*_tasks.json')])
    while ready_units():
            ready=ready_units();nodes=state.get('preferred_node_count',8 if len(ready)>1 else 1)
            assert nodes in [1,8]
            units=ready[:(8 if nodes==8 else 6)];round_id+=1
            if nodes==1:
                # Finish the nearly completed official/CE averages first.
                units.sort(key=lambda i:(i['objective']=='jsd_standard',i['task']=='dev_per_best3'))
            for item in units:
                key=item['objective']+'_'+item['task'];attempts[key]=attempts.get(key,0)+1
                # Normal walltime renewals resume saved splits and are not failures.
            task=f'pool{round_id}';task_file=HERE/f'interactive_driver_pool{round_id}_tasks.json';save(task_file,units)
            state.update(status='Requesting interactive allocation',task=task,units=units,node_count=nodes,updated_at_utc=now(),
                mode='At most 3 independent processes on one GPU' if nodes==1 else 'One independent process per GPU node')
            save(file,state)
            wait_release(state.get('job',''),state,file)
            arguments=['-I','-q','interact-g','-W','group_list=gm64','-l',f'select={nodes}:ncpus=32:mpiprocs=1',
                '-l','walltime='+('00:10:00' if nodes==8 else '00:20:00'),'-N','l100avg_live']
            with (HERE/f'interactive_{task}.log').open('a') as log:
                child,job=request(arguments,log,state,file)
                allocation=dict(task=task,units=units,nodes=nodes,job=job,requested_at_utc=now(),
                    execution_mode='At most 3 independent processes on one GPU' if nodes==1 else 'One independent process per GPU node')
                state['allocations'].append(allocation);state.update(status='Interactive allocation queued',job=job);save(file,state)
                print('INTERACTIVE_REQUESTED',job,units,flush=True)
                while True:
                    ready=child.expect([r'qsub: job '+job.replace('.',r'\.')+r' ready',pexpect.EOF,pexpect.TIMEOUT],timeout=30)
                    if ready==0:break
                    if ready==1:raise RuntimeError('Interactive request ended before ready: '+str(child.before))
                child.sendline('exec bash '+str(HERE/'interactive_pool.sh')+' '+str(task_file)+' '+str(nodes))
                child.expect(r'INTERACTIVE_STAGE_START (\S+) (\S+) (\S+)',timeout=60)
                allocation.update(node=child.match.group(2),started_at_utc=now())
                state.update(status='Running directly on interactive node',node=allocation['node']);save(file,state)
                print('INTERACTIVE_RUNNING',job,allocation['node'],flush=True)
                dispatch=json.loads((HERE/'dispatch.json').read_text())
                dispatch.update(interactive_job=job,interactive_node=allocation['node'],interactive_stage=task,
                    jobs={i['objective']:job for i in units},updated_at_utc=now(),mode=allocation['execution_mode'],interactive_nodes=nodes)
                for item in units:dispatch['task_jobs'].setdefault(item['objective'],{})[item['task']]=job
                save(HERE/'dispatch.json',dispatch)
                while child.expect([pexpect.EOF,pexpect.TIMEOUT],timeout=30)==1:
                    state.update(pending_units=[i for i in units if i['objective'] in pending(i['task'])],updated_at_utc=now());save(file,state)
                child.close();allocation.update(ended_at_utc=now(),exitstatus=child.exitstatus,signalstatus=child.signalstatus,
                    pending_after=[i for i in units if i['objective'] in pending(i['task'])]);save(file,state)
                print('INTERACTIVE_ENDED',job,allocation['pending_after'],flush=True)
            if allocation['pending_after']:
                errors=[]
                for item in allocation['pending_after']:
                    o=item['objective'];exitfile=HERE/f"worker_{o}_{item['task']}_exit_code.txt"
                    if exitfile.exists() and exitfile.read_text().strip() not in ['0','137','143']:
                        errors.append((o,exitfile.read_text().strip()))
                if errors:
                    state.update(status='Task error; see worker logs',errors=errors,updated_at_utc=now());save(file,state)
                    raise RuntimeError(str(errors))
    wait_release(state.get('job',''),state,file)
    state.update(complete=True,status='All objectives and all three averaging policies evaluated',updated_at_utc=now())
    save(file,state)
    print('INTERACTIVE_ALL_COMPLETE',flush=True)


if __name__=='__main__':main()
