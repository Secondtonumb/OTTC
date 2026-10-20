"""Read the CROTTC GPU discrepancy at default and strict FP32 precision."""
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'reproduction/crottc_timit_20ms_20261019'),str(ROOT/'reproduction/vendor'),str(ROOT/'reproduction'),str(ROOT)]
import torch
from transformers import set_seed
from common import recipe,matrix,SpeechViews,pair_chunks,write,source_hashes
from reproduction.crottc_timit_20ms_20261019.validate import tiny,Data,stack


def measure(name,strict):
    torch.backends.cuda.matmul.allow_tf32=not strict
    torch.backends.cudnn.allow_tf32=not strict
    set_seed(42);cfg=recipe(name)
    vocab=json.loads((ROOT/'reproduction/models/large_models_results/wav2vec2-large-lv60/runexp_ottc_fuse_phoneme/processor/vocab.json').read_text())
    model,_=tiny(cfg,vocab);model.cuda().train()
    batch={k:v.cuda() for k,v in stack([Data()[i] for i in range(4)]).items()}
    chunks,aug=pair_chunks(model,batch,SpeechViews('cuda'),4);data=chunks[0]
    full=model(**data).loss;full.backward()
    reference={k:p.grad.clone() for k,p in model.named_parameters() if p.grad is not None}
    model.zero_grad(set_to_none=True);total=0.
    for i in range(4):
        part={k:(torch.cat([v[i:i+1],v[4+i:5+i]]) if k in ['input_values','attention_mask','labels'] else
            v[i:i+1] if k=='cr_relative_lengths' else v) for k,v in data.items()}
        loss=model(**part).loss;loss.backward();total+=float(loss)
    grads={k:float((p.grad-reference[k]).abs().max()) for k,p in model.named_parameters() if k in reference}
    norm=sum(float(v.float().square().sum()) for v in reference.values())**.5
    diff=sum(float((p.grad-reference[k]).float().square().sum()) for k,p in model.named_parameters() if k in reference)**.5
    value=dict(recipe=name,strict_FP32=strict,full_loss=float(full),sum_micro_losses=total,
        loss_absolute_difference=abs(float(full)-total),normalized_loss_difference=abs(float(full)-total)/4,
        loss_relative_difference=abs(float(full)-total)/max(abs(float(full)),1e-12),
        max_gradient_difference=max(grads.values()),gradient_relative_L2_difference=diff/max(norm,1e-12),
        largest_gradient_differences=sorted(grads.items(),key=lambda p:p[1],reverse=True)[:5],augmentation=aug)
    print(json.dumps(value),flush=True)
    del model;torch.cuda.empty_cache();return value


def main():
    assert torch.cuda.is_available();torch.set_num_threads(4)
    records=[measure(name,strict) for strict in [False,True] for name in matrix()]
    write(ROOT/'reproduction/unfinished_recovery_20261019/CROTTC_NUMERICS.json',dict(source_sha256=source_hashes(),records=records,
        torch_version=torch.__version__,GPU=torch.cuda.get_device_name(),training_or_loss_changed=False))


if __name__=='__main__':main()
