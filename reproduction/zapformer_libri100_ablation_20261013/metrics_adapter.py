"""Retain the existing scorer; label its diagnostic graph independently of the loss."""
from pathlib import Path
import importlib.util
import hashlib

SOURCE=Path(__file__).resolve().parent.parent/'zapformer_s3t2ss_native_20261012/topology_metrics.py'
spec=importlib.util.spec_from_file_location('original_frozen_topology_metrics',SOURCE)
original=importlib.util.module_from_spec(spec);spec.loader.exec_module(original)


def evaluate(net,*args,**kwargs):
    result=original.evaluate(net,*args,**kwargs)
    config=net.loss_config
    result['protocol'].update(forward_sum_loss_config=config,
        loss_adapter_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        forced_alignment_graph='Retained S3T2** supplied-phone graph; evaluated independently of Charsiu position-CTC auxiliary loss' if config['forward_sum_kind']=='charsiu_positions' else 'S3T2** native topology',
        training_topology='OTTC three-state targets; auxiliary '+(config['forward_sum_kind'] if config['forward_sum_weight'] else 'disabled'))
    result['protocol']['k2_backend']['alignment_graph']='retained S3T2** supplied-phone diagnostic graph'
    result['protocol']['fa']='Supplied phones; retained S3T2** diagnostic graph; previous blank; duration/frames; auxiliary training loss may use position CTC'
    import json
    (kwargs['output'] if 'output' in kwargs else args[3]).joinpath('metrics.json').write_text(json.dumps(result,indent=2)+'\n')
    return result
