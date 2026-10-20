"""Time unmodified official OTTC training steps; never save trained weights."""
import copy
import ast
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
import time
import types

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent / "vendor"))
# Supply paths in memory: leave the original checkout's config untouched.
paths = types.ModuleType("ottc.config.path_config")
paths.DATASETS_ROOT_PATH = Path("/work/gm64/m64000/dataset")
paths.HUGGINGFACE_CACHE = Path(__file__).parent / "cache"
paths.LARGE_MODELS_PATH = Path("/work/gm64/m64000/reborn-eval-20261007/models")
sys.modules[paths.__name__] = paths

import soundfile as sf
import torch
import transformers
from transformers import Wav2Vec2FeatureExtractor
from transformers import Wav2Vec2Model
from ottc.models.sequence.wav2vec2_ottc import wvForOTTC, wvOTTCConfig
tree = ast.parse((ROOT / "ottc/tools/dataset_loader/timit.py").read_text())
timit_phoneme_fusion_map = next(ast.literal_eval(node.value) for node in tree.body
    if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and
    t.id == "timit_phoneme_fusion_map" for t in node.targets))


def main():
    assert torch.cuda.is_available()
    torch.set_num_threads(4)
    torch.manual_seed(0)
    raw = Path("/work/gm64/m64000/dataset/TIMIT/timit/train")
    wavs = sorted(raw.rglob("*.wav"))
    items = []
    vocabulary = set()
    for wav in wavs:
        duration = int(wav.with_suffix(".txt").read_text().split()[1]) / 16000
        phones = [timit_phoneme_fusion_map.get(x.split()[-1], x.split()[-1])
                  for x in wav.with_suffix(".phn").read_text().splitlines()]
        merged = [p for i, p in enumerate(phones) if i == 0 or p != phones[i-1]]
        vocabulary.update(merged)
        items.append((duration, wav, merged))
    vocab = {p: i+1 for i, p in enumerate(sorted(vocabulary))}
    vocab["<pad>"] = 0
    vocab["<unk>"] = len(vocab)
    items.sort(key=lambda x: (x[0], str(x[1])))
    processor = Wav2Vec2FeatureExtractor.from_pretrained(
        paths.LARGE_MODELS_PATH / "upstream", local_files_only=True)
    batch_size = 64
    n_batches = math.ceil(len(items) / batch_size)
    batches = []
    for fraction in (0.2, 0.5, 0.8, 0.95):
        index = min(n_batches-2, int(n_batches*fraction))
        chosen = items[index*batch_size:(index+1)*batch_size]
        audio = [sf.read(wav, dtype="float32")[0] for _, wav, _ in chosen]
        values = processor(audio, sampling_rate=16000, padding=True,
                           return_tensors="pt", return_attention_mask=True)
        labels = torch.full((len(chosen), max(len(p) for _, _, p in chosen)), -100,
                            dtype=torch.long)
        for i, (_, _, phones) in enumerate(chosen):
            labels[i, :len(phones)] = torch.tensor([vocab[p] for p in phones])
        batches.append((dict(input_values=values.input_values.to("cuda"),
                             attention_mask=values.attention_mask.to("cuda"),
                             labels=labels.to("cuda")),
                        {"fraction": fraction, "max_seconds": max(x[0] for x in chosen)}))
    config = wvOTTCConfig(wav2vec2_path=str(paths.LARGE_MODELS_PATH / "upstream"),
                          vocab_size=len(vocab), forward_type="regfrozen",
                          final_layers="linear", frozen_steps_for_regfrozen=100000)
    # Torch 2.6 uses parametrization keys for weight norm; convert the older
    # downloaded bin's names in memory so Transformers 4.39 loads both tensors.
    pretrained_loader = Wav2Vec2Model.from_pretrained
    def load_compatible(path, **kwargs):
        state = torch.load(Path(path) / "pytorch_model.bin", map_location="cpu", weights_only=True)
        for old, new in (("weight_g", "parametrizations.weight.original0"),
                         ("weight_v", "parametrizations.weight.original1")):
            keys = [k for k in state if k.endswith("pos_conv_embed.conv." + old)]
            for key in keys:
                state[key[:-len(old)] + new] = state.pop(key)
        loaded, info = pretrained_loader(path, state_dict=state, output_loading_info=True, **kwargs)
        assert not info["missing_keys"], info
        print("UPSTREAM_LOADING", json.dumps(info), flush=True)
        return loaded
    Wav2Vec2Model.from_pretrained = staticmethod(load_compatible)
    model = wvForOTTC(config).to("cuda")
    model.freeze_feature_extractor()
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=0.005)
    scaler = torch.amp.GradScaler("cuda")
    result = dict(gpu=torch.cuda.get_device_name(), torch=torch.__version__,
                  transformers=transformers.__version__, batch_size=batch_size,
                  vocab_size=len(vocab), real_phone_types=len(vocabulary),
                  train_utterances=len(items), train_hours=sum(x[0] for x in items)/3600,
                  batches_per_epoch=n_batches, epochs=40, precision="fp16",
                  gradient_checkpointing=model.wav2vec2.is_gradient_checkpointing,
                  head="official linear/linear (README default)",
                  trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
                  phases={}, timings=[],
                  source_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                 for p in (ROOT/"ottc/models/sequence/wav2vec2_ottc.py",
                                           ROOT/"ottc/ops_loss/ot_loss.py")})

    def step(batch):
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            out = model(**{k: v.clone() if k == "labels" else v for k, v in batch.items()})
        scaler.scale(out.loss).backward()
        scaler.step(optimizer)
        scaler.update()
        torch.cuda.synchronize()
        return float(out.loss.detach())

    for phase in ("joint_first30", "frozen_last10"):
        if phase == "frozen_last10":
            model.wav2vec2_freeze = copy.deepcopy(model.wav2vec2)
            model.frozen_steps_for_regfrozen = 1
            model.step = 2
        times = []
        for batch, meta in batches:
            step(batch)  # Warm up this shape and phase.
            torch.cuda.reset_peak_memory_stats()
            sample = []
            for _ in range(2):
                begin = time.perf_counter()
                loss = step(batch)
                sample.append(time.perf_counter()-begin)
            seconds = statistics.mean(sample)
            times.append(seconds)
            entry = dict(phase=phase, **meta, step_seconds=seconds, loss=loss,
                         peak_gb=torch.cuda.max_memory_allocated()/1e9)
            result["timings"].append(entry)
            print(json.dumps(entry), flush=True)
        result["phases"][phase] = dict(mean_step_seconds=statistics.mean(times),
                                      min_step_seconds=min(times), max_step_seconds=max(times))
        probes = {"encoder_layer0": model.wav2vec2.encoder.layers[0].attention.q_proj.weight,
                  "encoder_layer23": model.wav2vec2.encoder.layers[-1].attention.q_proj.weight,
                  "lm_head": model.lm_head.weight}
        result["phases"][phase]["gradient_norms"] = {
            name: float(p.grad.float().norm()) if p.grad is not None else None
            for name, p in probes.items()}
        assert all(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().max() > 0
                   for p in probes.values()), result["phases"][phase]
    # Measure a real shared-filesystem checkpoint, then delete only these newly
    # created timing files. No probe-trained checkpoint is retained.
    checkpoint = Path(__file__).parent / "temporary_timing_checkpoint"
    checkpoint.mkdir(exist_ok=False)
    begin = time.perf_counter()
    model.save_pretrained(checkpoint, safe_serialization=False)
    torch.save(optimizer.state_dict(), checkpoint / "optimizer.pt")
    result["checkpoint_write_seconds"] = time.perf_counter() - begin
    result["checkpoint_bytes"] = sum(p.stat().st_size for p in checkpoint.iterdir())
    for p in checkpoint.iterdir():
        p.unlink()
    checkpoint.rmdir()
    first = result["phases"]["joint_first30"]["mean_step_seconds"]
    last = result["phases"]["frozen_last10"]["mean_step_seconds"]
    result["projected_train_only_hours"] = n_batches*(30*first+10*last)/3600
    result["projection_note"] = (
        "Four duration-ordered batch probes, two measured optimizer steps each. "
        "Excludes evaluation, checkpoint writing, data loading, and queue time. "
        "The README linear heads differ from the paper's specified OT-weight MLP.")
    target = Path(__file__).parent / "timing.json"
    target.write_text(json.dumps(result, indent=2)+"\n")
    print("COMPLETE", target, result["projected_train_only_hours"], flush=True)


if __name__ == "__main__":
    main()
