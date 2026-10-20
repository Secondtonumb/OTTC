"""Artifact-first progress for the new fresh-SSL TIMIT matrix."""
from datetime import datetime,timezone
import json
from pathlib import Path
import subprocess
from common import HERE,write


def main():
    dispatch=json.loads((HERE/'dispatch.json').read_text());rows={}
    for name,entry in dispatch['experiments'].items():
        out=HERE/'models'/name;profile=HERE/'profiles'/name
        record=dict(job=entry.get('job'),profile_passed=(profile/'PASS.json').exists(),
            training_complete=(out/'TRAIN_COMPLETE.json').exists(),benchmark_complete=(out/'BENCHMARK_COMPLETE.json').exists(),
            saved_epochs=len(list((out/'epoch_weights').glob('epoch-*.pt'))))
        for filename,key in [('LATEST_STATE.json','latest'),('TRAIN_COMPLETE.json','training'),('DEV_SELECTION.json','dev_selection'),('BEST_DEV.json','best_dev')]:
            if (out/filename).exists():record[key]=json.loads((out/filename).read_text())
        if (out/'history.jsonl').exists():
            lines=(out/'history.jsonl').read_text().splitlines()
            if lines:record['last_history']=json.loads(lines[-1])
        if entry.get('job'):
            proc=subprocess.run(['qstat','-f',entry['job']],capture_output=True,text=True)
            record['scheduler']=proc.stdout.strip() if proc.returncode==0 else proc.stderr.strip()
        log=Path(entry.get('log',''))
        if log.is_file():record['log_tail']='\n'.join(log.read_text(errors='replace').splitlines()[-8:])
        rows[name]=record
    value=dict(updated_at_UTC=datetime.now(timezone.utc).isoformat(),experiments=rows)
    write(HERE/'STATUS.json',value)
    print(json.dumps(value,indent=2))


if __name__=='__main__':main()
