"""Compare microbatch throughput on fixed real audio without updating production models."""
import gc
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'reproduction'))
from train_multihead import AudioTargets, MultiPhoneModel, collate
import torch
import wandb


def main():
    from reproduction.wandb_policy import init_evaluation
    frozen = "--frozen" in sys.argv
    assert torch.cuda.is_available()
    torch.set_num_threads(4)
    output = Path(__file__).resolve().parent/('throughput_probe_frozen' if frozen else 'throughput_probe')
    output.mkdir(exist_ok=True)
    data = AudioTargets(ROOT/'reproduction/multihead_libri100/data/train-train.jsonl')
    ranked = sorted(range(len(data)), key=lambda i: data.records[i]['duration'])
    median_start = len(ranked)//2-32
    fixtures = {'typical': ranked[median_start:median_start+64], 'long': ranked[-64:]}
    batches = {name: [data[i] for i in ids] for name, ids in fixtures.items()}
    results = []
    run = init_evaluation(project='ottc-english-multiphoneset', entity='kevingenghaopeng-',
                     group='throughput_probe', name='libri_frozen_microbatch_4_8_16' if frozen else 'libri_microbatch_4_8_16', job_type='profiling',
                     config={'effective_batch': 64, 'microbatches': [4, 8, 16], 'production_updates': False, 'alignment_frozen': frozen},
                     dir=str(output), settings=wandb.Settings(init_timeout=45))
    for objective in (['ce', 'jsd_upstream'] if frozen else ['ctc', 'ce', 'jsd_upstream']):
        directory = ROOT/f'reproduction/multihead_libri100/models/{objective}'
        # Use the oldest retained checkpoint to avoid racing the checkpoint writer.
        candidates = [p for p in directory.glob('checkpoint-*') if (p/'config.json').exists()
                      and (not frozen or json.loads((p/'config.json').read_text()).get('freeze_alignment'))]
        checkpoint = min(candidates, key=lambda p: int(p.name[11:]))
        model = MultiPhoneModel.from_pretrained(checkpoint).cuda().train()
        model.gradient_checkpointing_enable()
        optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=1e-4)
        for kind, samples in batches.items():
            audio_seconds = sum(s['record']['duration'] for s in samples)
            for size in [4, 8, 16]:
                times = []
                peak = 0
                error = None
                for repetition in range(4):
                    inputs = cpu = loss = None
                    gc.collect()
                    torch.cuda.empty_cache()
                    torch.cuda.reset_peak_memory_stats()
                    optimizer.zero_grad(set_to_none=True)
                    torch.cuda.synchronize()
                    started = time.perf_counter()
                    try:
                        for start in range(0, 64, size):
                            cpu = collate(samples[start:start+size])
                            inputs = {k: {h: v.cuda() for h, v in value.items()} if isinstance(value, dict) else value.cuda()
                                      for k, value in cpu.items()}
                            with torch.autocast('cuda', dtype=torch.bfloat16):
                                loss = model(**inputs)['loss']/(64//size)
                            assert torch.isfinite(loss)
                            loss.backward()
                            del inputs, cpu, loss
                        # Same optimizer work with lr=0; benchmark does not advance production weights/state.
                        for group in optimizer.param_groups:
                            group['lr'] = 0
                        optimizer.step()
                        torch.cuda.synchronize()
                        elapsed = time.perf_counter()-started
                        peak = max(peak, torch.cuda.max_memory_allocated())
                        if repetition:
                            times.append(elapsed)
                    except torch.cuda.OutOfMemoryError:
                        error = 'OOM'
                        inputs = cpu = loss = None
                        optimizer.zero_grad(set_to_none=True)
                        gc.collect()
                        torch.cuda.empty_cache()
                        break
                entry = {'objective': objective, 'fixture': kind, 'microbatch': size,
                         'checkpoint': str(checkpoint), 'alignment_frozen': model.config.freeze_alignment, 'gpu': torch.cuda.get_device_name(),
                         'capacity_GiB': torch.cuda.get_device_properties(0).total_memory/2**30,
                         'peak_GiB': peak/2**30, 'error': error}
                if times:
                    entry.update(mean_step_seconds=sum(times)/len(times),
                                 audio_seconds_per_wall_second=audio_seconds/(sum(times)/len(times)))
                results.append(entry)
                print(json.dumps(entry), flush=True)
                run.log(entry)
                (output/'results.json').write_text(json.dumps(results, indent=2)+'\n')
        del model, optimizer
        gc.collect()
        torch.cuda.empty_cache()
    run.finish()
    (output/'BENCHMARK_COMPLETE.json').write_text(json.dumps({'passed': True, 'production_updated': False})+'\n')


if __name__ == '__main__':
    main()
