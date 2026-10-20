"""Average exactly the five retained completed continuation epochs."""
import bootstrap
from bootstrap import HERE
import argparse
import json
import os
from pathlib import Path
import shutil
import uuid
import torch
from safetensors.torch import load_file,save_file
from archive import digest,atomic_json


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--objective',required=True)
    args=parser.parse_args();output=HERE/'models'/args.objective
    complete=json.loads((output/'TRAIN_COMPLETE.json').read_text());assert complete['epoch']==5
    if (output/'AVERAGE_LAST5.json').exists():
        record=json.loads((output/'AVERAGE_LAST5.json').read_text())
        assert digest(Path(record['checkpoint'])/'model.safetensors')==record['weights_sha256']
        print('Existing last-five average reused',flush=True);return
    members=[];accumulation=None;dtypes={};keys=None;identity=None
    for epoch in range(1,6):
        directory=output/'epoch_weights'/f'epoch-{epoch:03d}'
        record=json.loads((directory/'EPOCH_COMPLETE.json').read_text())
        assert record['stage_epoch']==epoch
        assert record['weights_sha256']==digest(directory/'model.safetensors')
        if identity is None:identity=record['identity']
        assert identity==record['identity']
        values=load_file(str(directory/'model.safetensors'),device='cpu')
        if accumulation is None:
            keys=set(values)
            accumulation={k:v.double() if v.is_floating_point() else v.clone() for k,v in values.items()}
            dtypes={k:v.dtype for k,v in values.items()}
        else:
            assert set(values)==keys
            for k,v in values.items():
                assert v.dtype==dtypes[k] and v.shape==accumulation[k].shape
                if v.is_floating_point():accumulation[k].add_(v.double())
                else:assert torch.equal(accumulation[k],v)
        members.append(dict(path=str(directory),stage_epoch=epoch,stage_step=record['stage_step'],
            cumulative_epoch=record['cumulative_epoch'],cumulative_optimizer_step=record['cumulative_optimizer_step'],
            sha256=record['weights_sha256']))
        del values
    target=output/'averages/last5';target.parent.mkdir(exist_ok=True)
    temporary=target.with_name('last5.pending-'+uuid.uuid4().hex);temporary.mkdir()
    values={k:(v/5).to(dtypes[k]).contiguous() if v.is_floating_point() else v.contiguous() for k,v in accumulation.items()}
    save_file(values,str(temporary/'model.safetensors'),metadata={'format':'pt'})
    shutil.copy2(output/'epoch_weights/epoch-005/config.json',temporary/'config.json')
    record=dict(checkpoint=str(target),members=members,member_count=5,accumulation_dtype='float64',
        weights_sha256=digest(temporary/'model.safetensors'),optimizer_state_averaged=False,
        selection='fixed last five completed stage epochs 1..5; no test-based member choice',
        last10_available=False,last10_reason='only five new epochs authorized; no ten-epoch average claimed')
    atomic_json(temporary/'AVERAGE.json',record);os.replace(temporary,target)
    atomic_json(output/'AVERAGE_LAST5.json',record)
    print('AVERAGE_LAST5_COMPLETE',json.dumps(record),flush=True)


if __name__=='__main__':main()
