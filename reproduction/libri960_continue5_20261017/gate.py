"""Certify gradients, inherited optimizer state, resume, archives and native K2."""
import bootstrap
from bootstrap import HERE
import argparse
import json
from archive import atomic_json,digest
from train import hashes,model_directory,parent_checkpoint


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--objective',required=True);args=parser.parse_args()
    output=model_directory(args.objective,True)
    validation=json.loads((output/'VALIDATION_COMPLETE.json').read_text())
    evaluation=json.loads((HERE/'validation_evaluation'/args.objective/'EVAL_COMPLETE.json').read_text())
    assert validation['source_sha256']==hashes()
    assert validation['optimizer_and_rng_resume_verified'] and validation['epoch_archives']==4
    assert validation['normalization']['lazy_audio_matches_parent_cached_normalization']
    assert evaluation['smoke'] and evaluation['failures']=={'dev_clean':0}
    assert all(b['batch_size']==64 and b['alignment_remains_frozen'] for b in validation['batches'])
    _,parent=parent_checkpoint(args.objective)
    assert validation['parent_weights_sha256']==digest(parent/'model.safetensors')
    identity=json.loads((output/'wandb_run.json').read_text())
    assert identity['id']==validation['wandb']['id'],'Smoke/resume must keep W&B identity'
    target=HERE/'validation'/args.objective;target.mkdir(parents=True,exist_ok=True)
    atomic_json(target/'PASS.json',dict(source_sha256=hashes(),validation=validation,evaluation=evaluation))
    print('CONTINUATION_GPU_RESUME_EPOCH_K2_GATE_PASSED',args.objective,flush=True)


if __name__=='__main__':main()
