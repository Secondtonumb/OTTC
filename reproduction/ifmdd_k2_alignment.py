"""Load exact IF-MDD native-k2 and timestamp functions without CLI imports."""
import ast
import hashlib
from pathlib import Path

import numpy as np
import torch

SOURCE=Path('/home/m64000/work/IF-MDD/evaluate_l2arctic_fa_timestamps.py')
FUNCTIONS=['native_k2_forced_align_ids','native_k2_forced_align_ids_batch',
           'dense_path_to_target_segments','apply_blank_policy_to_segments']


def load_backend():
    text=SOURCE.read_text();tree=ast.parse(text)
    nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in FUNCTIONS]
    assert {n.name for n in nodes}==set(FUNCTIONS)
    namespace={'torch':torch,'np':np}
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(SOURCE),'exec'),namespace)
    return namespace,{'source':str(SOURCE),'source_sha256':hashlib.sha256(text.encode()).hexdigest(),
                      'functions':FUNCTIONS,'blank_policy':'previous','frame_time':'duration / number of frames',
                      'fallback_backend':None,'source_snapshot':'\n\n'.join(ast.get_source_segment(text,n) for n in nodes)}


def align(log_probs,targets,duration,backend,blank_policy='previous'):
    # K2 Fsa tracks tensor version counters, unavailable under inference_mode.
    # Clone into normal tensors while retaining no-grad model inference.
    with torch.inference_mode(False),torch.no_grad():
        scores=log_probs.detach().clone()
        scores=scores.unsqueeze(0) if scores.ndim==2 else scores
        labels,frame_scores=backend['native_k2_forced_align_ids'](scores,list(targets),0)
    collapsed=[];previous=None
    for token in labels.tolist():
        if token!=previous and token!=0:collapsed.append(token)
        previous=token
    if collapsed!=list(targets):raise ValueError('K2 path does not cover the exact supplied target sequence')
    segments=backend['dense_path_to_target_segments'](labels,frame_scores,list(targets),0,duration,blank_policy)
    return segments,labels.tolist()
