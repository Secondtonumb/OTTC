"""Re-evaluate preserved checkpoints with inference-safe K2 graph construction."""
from pathlib import Path
import sys
HERE = Path(__file__).resolve().parent
EXPERIMENT = HERE.parent / 'native_stride_ottc_20261018'
sys.path.insert(0, str(EXPERIMENT))
from common import load_data, build_model, source_hashes, sha, write
from train import start_wandb
import argparse
import copy
import fcntl
import gc
import json
import math
import os
import torch
from transformers import set_seed
import reproduction.native_stride_ottc_20261018.evaluate as evaluation
import reproduction.native_stride_ottc_20261018.graphs as graphs

original_graph = graphs.graph


def safe_graph(*args, **kwargs):
    # k2 FSA construction creates mutable tensors. The original function is
    # unchanged; only its execution context is corrected for per-utterance FA.
    with torch.inference_mode(False), torch.no_grad():
        return original_graph(*args, **kwargs)


def main():
    parser = argparse.ArgumentParser();parser.add_argument('--recipe', required=True)
    args = parser.parse_args();torch.set_num_threads(4);set_seed(42)
    assert torch.cuda.is_available()
    probe = json.loads((HERE/'GPU_PROBE.json').read_text())
    assert probe['inference_False_real_silence']['passed'] and not probe['inference_True_real_silence']['passed']
    old = EXPERIMENT/'models'/args.recipe
    output = HERE/'results'/args.recipe;output.mkdir(parents=True, exist_ok=True)
    lock=(output/'.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    config=json.loads((old/'run_config.json').read_text())
    assert config['source_sha256']==source_hashes(), 'Preserved recipe source changed'
    complete=json.loads((old/'TRAIN_COMPLETE.json').read_text())
    assert complete['epoch']==config['epochs']
    data,rows,vocab,processor,audit=load_data(config)
    assert audit==config['data'], 'Actual evaluation manifests changed'
    original_config=copy.deepcopy(config)
    # Construct normal tensors outside inference mode; do not reuse a model
    # whose frozen encoder was deep-copied during epoch-boundary evaluation.
    with torch.inference_mode(False):
        model,topo=build_model(config,vocab,math.ceil(len(data['train'])/config['global_batch']))
    assert config==original_config
    assert not any(p.is_inference() for p in model.parameters())
    del data;gc.collect()
    for filename in ['best_dev.pt','epoch_weights','final_5_average.pt','final_10_average.pt']:
        target=output/filename
        if not target.exists():target.symlink_to(old/filename,target_is_directory=filename=='epoch_weights')
    for filename in ['TRAIN_COMPLETE.json','run_config.json','wandb_run.json']:
        target=output/filename
        if not target.exists():target.write_text((old/filename).read_text())
    metadata=dict(recipe=args.recipe,training_restarted=False,optimizer_updated=False,
        original_model_directory=str(old),original_training_complete=complete,
        original_source_sha256=config['source_sha256'],evaluation_driver_sha256=sha(Path(__file__)),
        graph_fix='Original IF-MDD/K2 graph built under torch.inference_mode(False); arc topology unchanged',
        weight_loading_fix='Fresh model constructed outside inference mode, then existing checkpoint weights loaded',
        recognition_graph='original IF-MDD S3T2** phone loop, no transcript, phone LM or insertion penalty' if topo else 'native monophone greedy',
        scoring='unchanged frozen scorer; dev-only selection among best/final/final5/final10; separate test',
        job=os.environ.get('PBS_JOBID'))
    write(output/'EVALUATION_PROVENANCE.json',metadata)
    graphs.graph=safe_graph
    run=start_wandb(output,config,'evaluation-correction')
    run.config.update({'corrected_evaluation':metadata},allow_val_change=True)
    evaluation.final_benchmarks(config,model,rows,vocab,processor,topo,output,run)
    comparisons={}
    for target in sorted(output.glob('benchmark_*.json')):
        value=json.loads(target.read_text());value['protocol']['evaluation_correction']=metadata
        assert value['modes']['forced_alignment']['alignment_failures']==0, target
        write(target,value)
        previous=old/target.name
        if previous.exists():
            before=json.loads(previous.read_text())
            modes=['text_free']+(['fsa_text_free'] if topo else [])
            for mode in modes:
                assert before['modes'][mode]==value['modes'][mode], (target,mode,'recognition/segmentation changed')
            comparisons[target.name]='audio-only recognition and segmentation exactly match preserved results'
    write(output/'CORRECTION_VALIDATION.json',dict(all_forced_alignments_successful=True,
        existing_audio_only_metrics_exactly_unchanged=comparisons,
        no_new_training=True,source_and_data_contracts_verified=True))
    import wandb
    artifact=wandb.Artifact(args.recipe+'-corrected-evaluation',type='evaluation')
    for path in output.glob('*.json'):artifact.add_file(str(path),name=path.name)
    run.log_artifact(artifact);run.summary['corrected_evaluation_complete']=True
    selected=json.loads((output/'DEV_SELECTION.json').read_text())['selected']
    test=json.loads((output/f'benchmark_{selected}_test.json').read_text())
    for mode,metrics in test['modes'].items():run.summary[f'corrected/{mode}/test']=metrics
    run.finish()
    write(output/'CORRECTED_EVALUATION_COMPLETE.json',dict(recipe=args.recipe,selected=selected,
        all_FA_successful=True,training_restarted=False))


if __name__=='__main__':main()
