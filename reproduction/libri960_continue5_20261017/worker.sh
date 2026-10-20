#!/bin/bash
set -euo pipefail
cd /work/gm64/m64000/OTTC
source reproduction/libri960_continue5_20261017/environment.sh
objective=${1:?objective}
phase=${2:?validate or train}
export CUDA_VISIBLE_DEVICES=0
exec 9> "reproduction/libri960_continue5_20261017/.${objective}.lock"
flock 9
if [[ "$phase" == validate ]]; then
    "$LIBRI_PYTHON" -u reproduction/libri960_continue5_20261017/train.py --objective "$objective" --smoke
    "$LIBRI_PYTHON" -u reproduction/libri960_continue5_20261017/train.py --objective "$objective" --smoke --resume-validation
    "$LIBRI_PYTHON" -u reproduction/libri960_continue5_20261017/evaluate.py --objective "$objective" --smoke
    "$LIBRI_PYTHON" -u reproduction/libri960_continue5_20261017/gate.py --objective "$objective"
else
    remaining=$((LIBRI960_DEADLINE-$(date +%s)))
    if ((remaining < 600)); then exit 75; fi
    "$LIBRI_PYTHON" -u reproduction/libri960_continue5_20261017/train.py --objective "$objective" --max-runtime-seconds "$remaining"
    "$LIBRI_PYTHON" -u reproduction/libri960_continue5_20261017/average.py --objective "$objective"
    for variant in final best_dev average_last5; do
        "$LIBRI_PYTHON" -u reproduction/libri960_continue5_20261017/evaluate.py --objective "$objective" --variant "$variant"
    done
    exec 8> reproduction/libri960_continue5_20261017/.report.lock
    flock 8
    "$LIBRI_PYTHON" -u reproduction/libri960_continue5_20261017/summarize.py
    "$LIBRI_PYTHON" - <<'PY'
from pathlib import Path
import datetime,json,os
r=Path('reproduction/libri960_continue5_20261017');p=r/'dispatch.json'
if p.exists():
    a=json.loads(p.read_text());a['updated_at_JST']=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))).isoformat()
    a['experiments']={}
    for objective in ['ce','jsd_upstream','jsd_standard']:
        d=r/'models'/objective
        record={'training_complete':(d/'TRAIN_COMPLETE.json').exists(),'best_dev':None,
            'averaging_complete':(d/'AVERAGE_LAST5.json').exists(),
            'evaluation_complete':all((r/'evaluation'/objective/v/'EVAL_COMPLETE.json').exists() for v in ['final','best_dev','average_last5'])}
        for filename,key in [('wandb_run.json','wandb'),('TRAIN_COMPLETE.json','training'),('BEST_DEV.json','best_dev')]:
            if (d/filename).exists():record[key]=json.loads((d/filename).read_text())
        a['experiments'][objective]=record
    a['comparison_complete']=(r/'COMPARE_COMPLETE.json').exists()
    a['status']='all three continuations and comparisons complete' if a['comparison_complete'] else 'some continuations/variant evaluations still pending; see experiment markers'
    temp=p.with_name(p.name+'.pending-'+str(os.getpid()));temp.write_text(json.dumps(a,indent=2,ensure_ascii=False)+'\n');temp.replace(p)
PY
fi
