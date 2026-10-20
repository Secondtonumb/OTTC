"""Upload the matched-clock final comparison to each original experiment run."""
from pathlib import Path
import sys
HERE=Path(__file__).resolve().parent
EXPERIMENT=HERE.parent/'native_stride_ottc_20261018'
sys.path.insert(0,str(EXPERIMENT))
from common import write
from train import start_wandb
import json
import wandb


def main():
    report=json.loads((HERE/'RESULTS.json').read_text())
    assert len(report['rows'])==6 and report['topology_metrics_complete']
    published=[]
    for row in report['rows']:
        output=HERE/'results'/row['recipe']
        config=json.loads((EXPERIMENT/'models'/row['recipe']/'run_config.json').read_text())
        run=start_wandb(output,config,'evaluation-comparison')
        values=dict(test_PER=row['test_PER'],dev_PER=row['dev_PER'],selected=row['selected'],
            boundary_clock='fixed native10ms',tolerance_ms=20,**row['common_native_10ms_metrics']['20'])
        run.summary['matched_native_clock/test']=values
        run.log({f'matched_native_clock/test/{k}':v for k,v in values.items() if isinstance(v,(int,float))})
        artifact=wandb.Artifact(row['recipe']+'-matched-clock-comparison',type='evaluation')
        for filename in ['RESULTS.json','REPORT_ZH.md','CORRECTED_PATH_AUDIT.json','TRAINER_AMP_PROBE.json']:
            artifact.add_file(str(HERE/filename),name=filename)
        run.log_artifact(artifact);run.finish()
        published.append(dict(recipe=row['recipe'],id=run.id,status='synced' if run.settings.mode=='online' else 'offline_logs_retained'))
        write(HERE/'COMPARISON_PUBLICATION.json',dict(experiments=published))


if __name__=='__main__':main()
