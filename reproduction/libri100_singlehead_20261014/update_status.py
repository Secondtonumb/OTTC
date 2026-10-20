"""Refresh dispatch state from completion markers, trainer state, and W&B IDs."""
from pathlib import Path
import datetime
import json
import sys

here=Path(__file__).resolve().parent
root=here.parents[1]
path=here/'dispatch.json'
status=json.loads(path.read_text()) if path.exists() else {}
status['updated_at_JST']=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))).isoformat()
if len(sys.argv)>1:status['status']=sys.argv[1]
status['data_prepared']=(here/'data/PREPARE_COMPLETE.json').exists()
status['gpu_validation_complete']=(here/'GPU_VALIDATION_COMPLETE.json').exists()
status['experiments']={}
for objective in ['ce','jsd_upstream','jsd_standard']:
    directory=root/'reproduction/models/large_models_results/wav2vec2-large-lv60'/('runexp_libri100_singlehead_'+objective+'_20261014')
    record=dict(training_complete=(directory/'TRAIN_COMPLETE.json').exists(),
        evaluation_complete=(here/'evaluation'/objective/'EVAL_COMPLETE.json').exists())
    for filename,key in [('wandb_run.json','wandb'),('TRAIN_COMPLETE.json','training'),('trainer_state.json','last_saved_state')]:
        f=directory/filename
        if key=='last_saved_state':
            checkpoints=[p for p in directory.glob('checkpoint-*') if p.name.split('-')[-1].isdigit() and (p/filename).exists()]
            if checkpoints:
                latest=max(checkpoints,key=lambda p:int(p.name.split('-')[-1]))
                f=latest/filename
                record['last_saved_checkpoint']=str(latest)
        if f.exists():
            value=json.loads(f.read_text())
            record[key]={k:value[k] for k in ['epoch','global_step'] if k in value} if key=='last_saved_state' else value
    status['experiments'][objective]=record
status['comparison_complete']=(here/'COMPARE_COMPLETE.json').exists()
temporary=path.with_suffix('.json.tmp')
temporary.write_text(json.dumps(status,indent=2,ensure_ascii=False)+'\n');temporary.replace(path)
