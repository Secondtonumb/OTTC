"""Cache frozen original OTTC/K2 outputs, train only a refiner, evaluate pairs."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "reproduction")]
# The existing frozen-model overlay contains CPython 3.10 binary wheels.
# Pure refiner training/tests also work in the normal CPU Python environment.
if sys.version_info[:2] == (3, 10):
    sys.path.insert(0, str(ROOT / "reproduction/vendor"))

import torch
from ottc.models.alignment import (
    CIFBoundaryRefiner,
    boundary_loss,
    refine_from_ottc_mass,
)
from segmentation_metrics import _scorer, SOURCE as SCORER_SOURCE

MODEL_ROOT = ROOT / "reproduction/models/large_models_results/wav2vec2-large-lv60"
MODELS = {
    "jsd_upstream": "runexp_ottc_fuse_phoneme_jsd",
    "ce": "runexp_ottc_fuse_phoneme",
}
MANIFESTS = ROOT / "reproduction/multihead_timit/data"


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def fresh_directory(path):
    if path.exists() and any(path.iterdir()):
        raise ValueError(f"Refusing to overwrite nonempty output: {path}")
    path.mkdir(parents=True, exist_ok=True)


def synchronize(device):
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)


def export_cache(args):
    if sys.version_info[:2] != (3, 10):
        raise RuntimeError(
            "Frozen OTTC export requires the sb_k2_cuda Python 3.10 environment"
        )
    import soundfile as sf
    import transformers

    transformers.utils.import_utils._peft_available = False
    transformers.utils.import_utils._bitsandbytes_available = False
    from transformers import Wav2Vec2Processor
    from ottc.models.sequence.wav2vec2_ottc import wvForOTTC
    from ifmdd_k2_alignment import align, load_backend

    fresh_directory(args.output)
    model_dir = MODEL_ROOT / MODELS[args.model]
    checkpoint = model_dir / "checkpoint-2920"
    model, loading = wvForOTTC.from_pretrained(checkpoint, output_loading_info=True)
    if any(
        loading.get(k)
        for k in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")
    ):
        raise RuntimeError(f"Checkpoint mismatch: {loading}")
    model.requires_grad_(False).eval().to(args.device)
    model.step = 2920
    processor = Wav2Vec2Processor.from_pretrained(model_dir / "processor")
    vocab = processor.tokenizer.get_vocab()
    if vocab.get("<pad>") != 0:
        raise ValueError("Native-k2 bridge expects blank id 0")
    backend, backend_meta = load_backend()
    manifest = (
        MANIFESTS
        / {
            "train": "train-train.jsonl",
            "dev": "train-dev.jsonl",
            "test": "test.jsonl",
        }[args.split]
    )
    rows = [json.loads(line) for line in manifest.read_text().splitlines()]
    if args.limit:
        rows = rows[: args.limit]
    entries = []
    with torch.no_grad():
        for index, row in enumerate(rows):
            audio, rate = sf.read(row["wav"], dtype="float32")
            if rate != 16000 or audio.ndim != 1:
                raise ValueError(f'Expected mono 16 kHz: {row["wav"]}')
            duration = len(audio) / rate
            refs = row["references"]["ottc_arpabet"]
            ids = [
                vocab[p] for p, _, _ in refs
            ]  # Map strings, never borrow a different head's IDs.
            if not ids or any(i in (0, vocab.get("<unk>")) for i in ids):
                raise ValueError("Empty or special-token phone sequence")
            for left, right in zip(refs, refs[1:]):
                if abs(left[2] - right[1]) > 1 / rate + 1e-8:
                    raise ValueError(f'Noncontiguous reference intervals: {row["utt"]}')
            inputs = processor(
                audio,
                sampling_rate=rate,
                return_tensors="pt",
                return_attention_mask=True,
            ).to(args.device)
            # Use the actual original forward so regfrozen selects the correct
            # frozen alignment encoder for alpha. Capture content features in
            # the same pass for the optional learned-head experiment.
            captured = []
            hook = model.wav2vec2.register_forward_hook(
                lambda _m, _a, out: captured.append(out[0])
            )
            synchronize(args.device)
            forward_start = time.perf_counter()
            try:
                output = model(**inputs, return_dict=True)
            finally:
                hook.remove()
            synchronize(args.device)
            model_forward_ms = 1000 * (time.perf_counter() - forward_start)
            hidden = model.dropout(captured[0])
            logits = output.logits
            frame_mass = output.hidden_states[0][0]
            if index == 0:
                torch.testing.assert_close(logits, model.lm_head(hidden))
            synchronize(args.device)
            start = time.perf_counter()
            segments, _ = align(logits.log_softmax(-1)[0], ids, duration, backend)
            synchronize(args.device)
            k2_ms = 1000 * (time.perf_counter() - start)
            if [s["token_id"] for s in segments] != ids:
                raise ValueError("K2 returned different phone positions")
            for left, right in zip(segments, segments[1:]):
                if abs(left["end"] - right["start"]) > 1e-7:
                    raise ValueError("Refiner requires contiguous K2 segments")
            filename = f"{index:06d}.pt"
            torch.save(
                dict(
                    utt=row["utt"],
                    features=hidden[0].cpu().float(),
                    frame_mass=frame_mass.cpu().float(),
                    phone_ids=torch.tensor(ids),
                    duration=duration,
                    coarse_times=torch.tensor([s["start"] for s in segments[1:]]),
                    reference_times=torch.tensor([r[1] for r in refs[1:]]),
                    coarse_segments=segments,
                    references=refs,
                    k2_ms=k2_ms,
                    model_forward_ms=model_forward_ms,
                ),
                args.output / filename,
            )
            entries.append(dict(utt=row["utt"], file=filename))
            print(
                f'cached {args.model}/{args.split} {index + 1}/{len(rows)} {row["utt"]}',
                flush=True,
            )
    # Hash actual weights for cross-cache/checkpoint provenance, not just names.
    metadata = dict(
        format_version=2,
        model=args.model,
        checkpoint=str(checkpoint),
        step=2920,
        checkpoint_sha256=digest(checkpoint / "model.safetensors"),
        manifest=str(manifest),
        manifest_sha256=digest(manifest),
        split=args.split,
        subset_limit=args.limit,
        utterances=len(entries),
        vocab=vocab,
        feature_dim=model.lm_head.in_features,
        vocab_size=model.lm_head.out_features,
        features="prediction_encoder_last_hidden_state_eval; base model frozen",
        mass_source="original forward at step2920; regfrozen uses wav2vec2_freeze -> lm_weight -> temporal softmax",
        task="supplied_phone_forced_alignment",
        time_clock="duration / valid_frames",
        backend=backend_meta,
        entries=entries,
        recognition_metrics="not measured: phone transcript is supplied",
        device_name=(
            torch.cuda.get_device_name(args.device)
            if torch.device(args.device).type == "cuda"
            else "CPU"
        ),
        torch_version=str(torch.__version__),
        cuda_version=torch.version.cuda,
        implementation_sha256={
            str(p.relative_to(ROOT)): digest(p)
            for p in (
                Path(__file__),
                ROOT / "ottc/models/alignment/cif_boundary.py",
                ROOT / "ottc/models/alignment/ottc_mass_boundary.py",
            )
        },
    )
    dump(args.output / "index.json", metadata)


class Cache:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.meta = json.loads((self.directory / "index.json").read_text())
        self.entries = self.meta["entries"]
        if not self.entries:
            raise ValueError("Empty cache")

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, index):
        return torch.load(
            self.directory / self.entries[index]["file"],
            map_location="cpu",
            weights_only=True,
        )


def predict(model, item, device):
    # Deliberately never pass references into the inference interface.
    return model(
        item[
            "frame_mass" if getattr(model, "uses_frozen_mass", False) else "features"
        ].to(device),
        item["coarse_times"].to(device),
        item["phone_ids"].to(device),
        item["duration"],
    )


def compatible(first, second):
    for key in ("checkpoint_sha256", "vocab", "feature_dim", "time_clock"):
        if first[key] != second[key]:
            raise ValueError(f"Incompatible caches/refiner: {key}")


class FrozenOTRefiner(torch.nn.Module):
    uses_frozen_mass = True

    def __init__(self, max_shift_ms=20.0):
        super().__init__()
        self.max_shift_ms = max_shift_ms

    def forward(self, mass, coarse, ids, duration):
        if len(ids) != len(coarse) + 1:
            raise ValueError("N phones require N-1 coarse boundaries")
        return refine_from_ottc_mass(
            mass, coarse, duration, max_shift_ms=self.max_shift_ms
        )


def measure(model, cache, device, predictions_path=None):
    model.eval()
    frozen = getattr(model, "uses_frozen_mass", False)
    names = (
        ("k2", "k2_refined", "ot_global", "ot_global_frame_left")
        if frozen
        else ("k2", "k2_refined")
    )
    tolerances = (5, 10, 20)
    counts = {
        name: {
            tol: dict(
                precision_matches=0,
                recall_matches=0,
                predicted_count=0,
                reference_count=0,
            )
            for tol in tolerances
        }
        for name in names
    }
    errors = {name: [] for name in counts}
    unreachable = total = accepted = 0
    losses, elapsed, predictions = [], [], []
    with torch.no_grad():
        predict(model, cache[0], device)  # Warm up; exclude from timing.
        for index in range(len(cache)):
            item = cache[index]
            # Exclude disk I/O and device copies from head-only timing.
            features, coarse, ids = (
                item[k].to(device)
                for k in (
                    "frame_mass" if frozen else "features",
                    "coarse_times",
                    "phone_ids",
                )
            )
            synchronize(device)
            start = time.perf_counter()
            out = model(features, coarse, ids, item["duration"])
            synchronize(device)
            elapsed.append(1000 * (time.perf_counter() - start))
            gold = item["reference_times"]
            n = len(gold)
            losses.append((boundary_loss(out, gold).item(), n))
            total += n
            unreachable += int(
                (
                    (gold < out.window_edges[:, 0].cpu())
                    | (gold > out.window_edges[:, -1].cpu())
                ).sum()
            )
            variants = {"k2": coarse.cpu(), "k2_refined": out.times.cpu()}
            if frozen:
                accepted += int(out.accepted.sum())
                variants["ot_global"] = out.global_times.cpu()
                edges = torch.linspace(
                    0, item["duration"], len(features) + 1, device=device
                )
                cells = (torch.searchsorted(edges, out.global_times) - 1).clamp(
                    0, len(features) - 1
                )
                variants["ot_global_frame_left"] = edges[cells].cpu()
            for name, times in variants.items():
                errors[name].extend((1000 * (times - gold).abs()).tolist())
                for tol in tolerances:
                    c = _scorer.precision_recall_counts(
                        times.tolist(), gold.tolist(), tol / 1000
                    )
                    acc = counts[name][tol]
                    acc["precision_matches"] += c["precision_matches"]
                    acc["recall_matches"] += c["recall_matches"]
                    acc["predicted_count"] += c["predicted_boundaries"]
                    acc["reference_count"] += c["reference_boundaries"]
            predictions.append(
                dict(
                    utt=item["utt"],
                    coarse=coarse.cpu().tolist(),
                    refined=out.times.cpu().tolist(),
                    reference=gold.tolist(),
                    offsets_ms=(out.offsets * 1000).cpu().tolist(),
                )
            )
            if frozen:
                predictions[-1].update(
                    global_crossings=out.global_times.cpu().tolist(),
                    accepted=out.accepted.cpu().tolist(),
                )
    result = dict(
        split=cache.meta["split"],
        utterances=len(cache),
        internal_boundaries=total,
        task="supplied_phone_forced_alignment",
        protocol="All internal token boundaries, including silence/q/dx; endpoints excluded; repeated tokens retained. Unlabeled frozen directional matching.",
        scorer_source=str(SCORER_SOURCE),
        scorer_sha256=digest(SCORER_SOURCE),
        loss_smooth_l1_ms=sum(v * n for v, n in losses) / total if total else None,
        reference_outside_window=unreachable,
        head_only_ms_per_utterance=sum(elapsed) / len(elapsed),
        head_only_median_ms=torch.tensor(elapsed).median().item(),
        head_only_p90_ms=torch.quantile(torch.tensor(elapsed), 0.9).item(),
        timing_device=str(device),
        timing_scope="Includes validation/synchronization; excludes encoder, K2, disk and device copies; no end-to-end speed claim",
        recognition_metrics="not measured: supplied phone transcript",
    )
    if frozen:
        result.update(
            training_required=False,
            accepted_crossings=accepted,
            accepted_fraction=accepted / total if total else None,
            max_shift_ms=model.max_shift_ms,
            gate="Accept global crossing only inside symmetric K2 window bounded by neighbor midpoints; otherwise keep K2",
            global_frame_baseline="Left edge of the same OT crossing cell, duration/T clock",
            mass_source=cache.meta.get("mass_source"),
        )
    for name in counts:
        e = torch.tensor(errors[name])
        result[name] = dict(
            mae_ms=e.mean().item() if len(e) else None,
            p90_ms=torch.quantile(e, 0.9).item() if len(e) else None,
            boundaries={
                str(tol): _scorer.finalize_counts(**c)
                for tol, c in counts[name].items()
            },
        )
    if predictions_path:
        Path(predictions_path).write_text(
            "".join(json.dumps(p) + "\n" for p in predictions)
        )
    return result


def train(args):
    import wandb

    train_cache, dev_cache = Cache(args.train_cache), Cache(args.dev_cache)
    if train_cache.meta["split"] != "train" or dev_cache.meta["split"] != "dev":
        raise ValueError(
            "Training requires train cache; model selection requires dev, never test"
        )
    compatible(train_cache.meta, dev_cache.meta)
    if {e["utt"] for e in train_cache.entries} & {e["utt"] for e in dev_cache.entries}:
        raise ValueError("Train/dev utterances overlap")
    fresh_directory(args.output)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    model = CIFBoundaryRefiner(
        train_cache.meta["feature_dim"],
        train_cache.meta["vocab_size"],
        max_shift_ms=args.max_shift_ms,
    ).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    config.update(
        model.config,
        base_checkpoint=train_cache.meta["checkpoint"],
        base_checkpoint_sha256=train_cache.meta["checkpoint_sha256"],
        train_utterances=len(train_cache),
        dev_utterances=len(dev_cache),
        base_frozen=True,
        loss="boundary SmoothL1(beta=5ms), no CTC",
        recognition_metrics="not applicable to this fixed-phone timestamp head",
    )
    defaults = json.loads((ROOT / "reproduction/wandb_config.json").read_text())
    options = dict(
        project=defaults["project"],
        entity=defaults.get("entity"),
        name=args.output.name,
        config=config,
        dir=str(args.output.resolve()),
        mode=args.wandb_mode,
        settings=wandb.Settings(init_timeout=45),
    )
    try:
        run = wandb.init(**options)
    except wandb.errors.CommError:
        if args.wandb_mode != "online":
            raise
        options["mode"] = "offline"
        run = wandb.init(**options)
    dump(
        args.output / "wandb_run.json",
        dict(id=run.id, project=run.project, entity=run.entity, mode=options["mode"]),
    )
    run.define_metric("step")
    run.define_metric("*", step_metric="step")
    dump(args.output / "config.json", config)
    best, step = float("inf"), 0
    try:
        with (args.output / "history.jsonl").open("w") as history:
            baseline = measure(model, dev_cache, args.device)
            dump(args.output / "dev_initial.json", baseline)
            if baseline["k2"]["mae_ms"] is None:
                raise ValueError("Dev set has no internal boundaries")
            # Identity is a valid candidate; never force a worse trained head.
            best = baseline["k2_refined"]["mae_ms"]

            def save(epoch):
                torch.save(
                    dict(
                        config=model.config,
                        state=model.state_dict(),
                        optimizer=optimizer.state_dict(),
                        epoch=epoch,
                        step=step,
                        best_dev_mae_ms=best,
                        cache_metadata=train_cache.meta,
                    ),
                    args.output / "best.pt",
                )

            save(0)
            dump(args.output / "dev_best.json", baseline)
            for epoch in range(1, args.epochs + 1):
                model.train()
                order = list(range(len(train_cache)))
                random.shuffle(order)
                for index in order:
                    item = train_cache[index]
                    if not item["reference_times"].numel():
                        continue
                    optimizer.zero_grad(set_to_none=True)
                    loss = boundary_loss(
                        predict(model, item, args.device), item["reference_times"]
                    )
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                    optimizer.step()
                    step += 1
                    log = dict(
                        step=step,
                        epoch=epoch,
                        train_loss_smooth_l1_ms=loss.item(),
                        lr=args.lr,
                    )
                    history.write(json.dumps(log) + "\n")
                    run.log(log)
                metrics = measure(model, dev_cache, args.device)
                dump(args.output / f"dev_epoch{epoch:03d}.json", metrics)
                log = dict(
                    step=step,
                    epoch=epoch,
                    dev_loss_smooth_l1_ms=metrics["loss_smooth_l1_ms"],
                    dev_mae_ms=metrics["k2_refined"]["mae_ms"],
                    dev_f1_20ms=metrics["k2_refined"]["boundaries"]["20"]["f1_pct"],
                    dev_r_value_20ms=metrics["k2_refined"]["boundaries"]["20"][
                        "r_value_pct"
                    ],
                )
                history.write(json.dumps(log) + "\n")
                history.flush()
                run.log(log)
                if metrics["k2_refined"]["mae_ms"] < best:
                    best = metrics["k2_refined"]["mae_ms"]
                    save(epoch)
                    dump(args.output / "dev_best.json", metrics)
                print(json.dumps(log), flush=True)
            run.summary["best_dev_mae_ms"] = best
    finally:
        run.finish()


def evaluate(args):
    cache = Cache(args.cache)
    saved = torch.load(args.checkpoint, map_location=args.device, weights_only=True)
    compatible(saved["cache_metadata"], cache.meta)
    if cache.meta["split"] not in ("dev", "test"):
        raise ValueError("Benchmark evaluation requires dev or test cache")
    model = CIFBoundaryRefiner(**saved["config"]).to(args.device)
    model.load_state_dict(saved["state"])
    fresh_directory(args.output)
    metrics = measure(model, cache, args.device, args.output / "predictions.jsonl")
    metrics.update(
        refiner_checkpoint=str(args.checkpoint),
        selected_epoch=saved["epoch"],
        selected_step=saved["step"],
        selection="minimum dev internal-boundary MAE, including identity epoch0",
        base_checkpoint_sha256=cache.meta["checkpoint_sha256"],
    )
    dump(args.output / "metrics.json", metrics)
    print(json.dumps(metrics, indent=2))


def evaluate_frozen(args):
    cache = Cache(args.cache)
    if cache.meta["split"] not in ("dev", "test"):
        raise ValueError("Benchmark evaluation requires dev or test cache")
    fresh_directory(args.output)
    model = FrozenOTRefiner(args.max_shift_ms)
    metrics = measure(model, cache, args.device, args.output / "predictions.jsonl")
    metrics.update(
        base_checkpoint=cache.meta["checkpoint"],
        base_checkpoint_sha256=cache.meta["checkpoint_sha256"],
        subset_limit=cache.meta["subset_limit"],
    )
    dump(args.output / "metrics.json", metrics)
    print(json.dumps(metrics, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    export = sub.add_parser("export")
    export.add_argument("--model", choices=MODELS, default="jsd_upstream")
    export.add_argument("--split", choices=("train", "dev", "test"), required=True)
    export.add_argument("--limit", type=int, default=0, help="0 means full manifest")
    fit = sub.add_parser("train")
    fit.add_argument("--train-cache", type=Path, required=True)
    fit.add_argument("--dev-cache", type=Path, required=True)
    fit.add_argument("--epochs", type=int, default=10)
    fit.add_argument("--lr", type=float, default=1e-3)
    fit.add_argument("--seed", type=int, default=42)
    fit.add_argument("--max-shift-ms", type=float, default=20.0)
    fit.add_argument(
        "--wandb-mode",
        choices=("online", "offline", "disabled"),
        default=os.environ.get("WANDB_MODE", "online"),
    )
    ev = sub.add_parser("evaluate")
    ev.add_argument("--cache", type=Path, required=True)
    ev.add_argument("--checkpoint", type=Path, required=True)
    frozen = sub.add_parser(
        "evaluate-frozen", help="Reuse original alpha; no new head or training"
    )
    frozen.add_argument("--cache", type=Path, required=True)
    frozen.add_argument("--max-shift-ms", type=float, default=20.0)
    for p in (export, fit, ev, frozen):
        p.add_argument("--device", default="cuda")
        p.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "export" and args.limit < 0:
        parser.error("--limit must be nonnegative")
    if args.command == "train" and (args.epochs < 1 or args.lr <= 0):
        parser.error("epochs and lr must be positive")
    {
        "export": export_cache,
        "train": train,
        "evaluate": evaluate,
        "evaluate-frozen": evaluate_frozen,
    }[args.command](args)


if __name__ == "__main__":
    main()
