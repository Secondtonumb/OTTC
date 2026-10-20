"""Publish validated corrected summaries and verify retained Trainer AMP behavior."""
from pathlib import Path
import sys
HERE=Path(__file__).resolve().parent
EXPERIMENT=HERE.parent/'native_stride_ottc_20261018'
sys.path.insert(0,str(EXPERIMENT))
from common import write,sha
from train import start_wandb
import json
import torch
import wandb
from accelerate import Accelerator
import accelerate.accelerator


def main():
    assert json.loads((HERE/'CORRECTED_PATH_AUDIT.json').read_text())['valid_supplied_phone_paths']==7104
    accelerator=Accelerator(mixed_precision='bf16')
    model=torch.nn.Sequential(torch.nn.Linear(8,8)).cuda()
    captured=[]
    model[0].register_forward_hook(lambda module,inputs,output:captured.append(str(output.dtype)))
    model=accelerator.prepare_model(model)
    with torch.no_grad():out=model(torch.randn(2,8,device='cuda'))
    proof=dict(internal_linear_output_dtype=captured[-1],returned_output_dtype=str(out.dtype),
        retained_original_forward=hasattr(model,'_original_forward'),
        accelerator_source=str(Path(accelerate.accelerator.__file__)),
        accelerator_source_sha256=sha(Path(accelerate.accelerator.__file__)))
    assert proof['internal_linear_output_dtype']=='torch.bfloat16' and proof['returned_output_dtype']=='torch.float32'
    write(HERE/'TRAINER_AMP_PROBE.json',proof)
    for cost in ['jsd_upstream','ce','jsd_standard']:
        name=f'wav2vec2__timit__10ms__s3t2ss__{cost}';output=HERE/'results'/name
        config=json.loads((EXPERIMENT/'models'/name/'run_config.json').read_text())
        run=start_wandb(output,config,'evaluation-correction')
        run.config.update({'standalone_evaluation_precision':'FP32, fresh model without Trainer AMP wrapper',
            'corrected_K2_graph_context':'torch.inference_mode(False)'},allow_val_change=True)
        marker=json.loads((output/'CORRECTED_EVALUATION_COMPLETE.json').read_text())
        selected=marker['selected'];test=json.loads((output/f'benchmark_{selected}_test.json').read_text())
        for mode,metrics in test['modes'].items():run.summary[f'corrected/{mode}/test']=metrics
        run.summary['corrected_evaluation_complete']=True
        artifact=wandb.Artifact(name+'-corrected-evaluation',type='evaluation')
        for path in output.glob('*.json'):artifact.add_file(str(path),name=path.name)
        for filename in ['CORRECTED_PATH_AUDIT.json','TRAINER_AMP_PROBE.json']:
            artifact.add_file(str(HERE/filename),name=filename)
        run.log_artifact(artifact);run.finish()
        marker['wandb_artifact_status']='synced' if run.settings.mode=='online' else 'offline_logs_retained'
        write(output/'CORRECTED_EVALUATION_COMPLETE.json',marker)


if __name__=='__main__':main()
