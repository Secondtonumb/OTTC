"""Decode a frozen S3T2** checkpoint; no training or checkpoint selection."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

os.environ['TORCHDYNAMO_DISABLE'] = '1'
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
sys.path.insert(0, '/work/gm64/m64000/fa-bench/analysis/mfa_spottc_jsd_s3t2ss')
import torch
from model_star_star import MFAZapformer
from metrics import evaluate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    state = torch.load(args.checkpoint, map_location='cpu', mmap=True, weights_only=False)
    assert state['contract']['topology'] == 's3_t2_star_star'
    assert state['contract']['ottc_cost'] == 'jsd_upstream'
    model = MFAZapformer()
    model.load_state_dict(state['model'], strict=True)
    model.cuda().eval()
    result = dict(checkpoint=str(args.checkpoint), progress=state['progress'],
                  checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
                  final_training_checkpoint=False, purpose='fixed epoch-5 inventory rescore')
    del state
    result['splits'] = {}
    for split in ['test', 'development']:
        result['splits'][split] = evaluate(model, split=split, forced=False,
                                         output=args.output, topology='s3_t2_star_star')
        (args.output / 'metrics.json').write_text(json.dumps(result, indent=2) + '\n')
        print(split, json.dumps(result['splits'][split]['text_free']), flush=True)
    (args.output / 'EVAL_COMPLETE.json').write_text(json.dumps(result['progress']) + '\n')


if __name__ == '__main__':
    main()
