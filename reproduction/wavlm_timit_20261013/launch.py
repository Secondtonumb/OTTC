"""Matched WavLM Large / TIMIT single-head upstream JSD, 40 epochs."""
import bootstrap
from bootstrap import ROOT,HERE,source_hashes
import argparse
import hashlib
import json
import os
import platform
import sys
import time
import torch
import run_exp
from model import WavLMForOTTC,WavLMOTTCConfig
from development import DevelopmentCallback
from time_budget import TimeBudgetCallback

SUFFIX='runexp_ottc_fuse_phoneme_jsd_wavlm_20261013'
ENCODER='wavlm-large'
MODEL_DIR=ROOT/'reproduction/models/large_models_results'/ENCODER/SUFFIX
RESULT_DIR=ROOT/'reproduction/results'/SUFFIX


def configure():
    run_exp.wvForOTTC=WavLMForOTTC
    run_exp.wvOTTCConfig=WavLMOTTCConfig


def argv():
    return ['run_exp.py','--dataset','timit_fused_phoneme','--train_name','train',
        '--test_name','kaldi_test','--loss_type','ottc','--prepare_dataset','prepare_dataset_for_timit_phoneme',
        '--encoder_name',ENCODER,'--batch_size','64','--learning_rate','2e-4',
        '--num_train_epochs','40','--suffix',SUFFIX,'--forward_type','regfrozen',
        '--frozen_epochs_for_regfrozen','30','--bins_metric','jsd','--logging_steps','50']


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--max-wall-seconds',type=float,default=0,
        help='Pause and save after this invocation budget; 0 means no local limit')
    args=parser.parse_args()
    if args.max_wall_seconds<0:parser.error('Budget must be nonnegative')
    deadline=time.monotonic()+args.max_wall_seconds if args.max_wall_seconds else None
    assert torch.cuda.is_available(),'Use an allocated GPU node'
    torch.set_num_threads(4)
    configure()
    if (RESULT_DIR/'TRAIN_COMPLETE.json').exists():
        print('Already completed; reuse trained WavLM checkpoint');return
    gate=json.loads((HERE/'profile/PASS.json').read_text())
    hashes=source_hashes()
    if gate['source_sha256']!=hashes:raise ValueError('Source differs from GPU profile gate')
    base=ROOT/'reproduction/models/wavlm-large'
    receipt=dict(command=argv(),encoder='microsoft/wavlm-large',
        encoder_revision='c1423ed94bb01d80a3f5ce5bc39f6026a0f4828c',
        initialization='official self-supervised pretrained checkpoint; no IF-MDD finetuned weights',
        checkpoint_compatibility='legacy weight_g/weight_v renamed to parametrizations.weight.original0/original1; tensors unchanged',
        model_sha256=hashlib.sha256((base/'pytorch_model.bin').read_bytes()).hexdigest(),
        backbone_config=json.loads((base/'config.json').read_text()),
        original_comparison=str(ROOT/'reproduction/results/runexp_ottc_fuse_phoneme_jsd/metrics.json'),
        profile_gate=gate,source_sha256=hashes,job=os.environ.get('PBS_JOBID'),host=platform.node(),seed=42,
        frame_ms=20,single_head=True,upstream_jsd=True,task='TIMIT40',
        selection='final epoch40; matching original single-head reproduction',
        original_readme_eval='kaldi test240 used each epoch as in original; core192 reported separately')
    receipt['max_wall_seconds_this_invocation']=args.max_wall_seconds
    RESULT_DIR.mkdir(parents=True,exist_ok=True)
    runtime=RESULT_DIR/'runtime.json'
    if runtime.exists():
        previous=json.loads(runtime.read_text())
        for key in ('command','source_sha256','model_sha256','backbone_config'):
            if previous[key]!=receipt[key]:raise ValueError('Resume changed '+key)
        (RESULT_DIR/f"runtime_resume_{receipt['job']}.json").write_text(json.dumps(receipt,indent=2)+'\n')
    else:runtime.write_text(json.dumps(receipt,indent=2)+'\n')
    original_load=run_exp.load_dataset_and_processor
    def matched_data(config,break_douuble_letter):
        data,vocab,processor=original_load(config,break_douuble_letter)
        old=ROOT/'reproduction/models/large_models_results/wav2vec2-large-lv60/runexp_ottc_fuse_phoneme_jsd'
        assert vocab==json.loads((old/'vocab_used.json').read_text())
        assert {k:len(v) for k,v in data.items()}==json.loads((old/'dataset_sizes.json').read_text())
        assert data['train']._fingerprint==gate['train_fingerprint']
        assert data['kaldi_test']._fingerprint==gate['kaldi_test_fingerprint']
        return data,vocab,processor
    run_exp.load_dataset_and_processor=matched_data
    original_trainer=run_exp.create_trainer
    def tracked_trainer(model,data,processor,config,metrics_fn,output):
        trainer=original_trainer(model,data,processor,config,metrics_fn,output)
        trainer.add_callback(DevelopmentCallback(processor,output))
        trainer.add_callback(TimeBudgetCallback(deadline))
        return trainer
    run_exp.create_trainer=tracked_trainer
    sys.argv=argv();run_exp.main()
    state=json.loads((MODEL_DIR/'trainer_state.json').read_text())
    if state['global_step']<2920:
        (RESULT_DIR/'PAUSED.json').write_text(json.dumps(dict(job=receipt['job'],epoch=state['epoch'],global_step=state['global_step'],reason='invocation time budget; resume same entry'))+'\n')
        print('Saved partial training state; rerun the interactive entry to continue to total 40 epochs',flush=True)
        return
    assert state['epoch']==40 and state['global_step']==2920,state
    (RESULT_DIR/'PAUSED.json').unlink(missing_ok=True)
    (RESULT_DIR/'TRAIN_COMPLETE.json').write_text(json.dumps(dict(job=receipt['job'],epoch=40,global_step=2920))+'\n')


if __name__=='__main__':main()
