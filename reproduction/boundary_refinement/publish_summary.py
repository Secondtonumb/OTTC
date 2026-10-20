"""Track complete frozen-model evaluation in separate W&B evaluation runs."""

import argparse
import json
import os
from pathlib import Path

import wandb

ROOT = Path(__file__).resolve().parents[2]


def main():
    from reproduction.wandb_policy import init_evaluation
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument(
        "--mode",
        choices=("online", "offline"),
        default=os.environ.get("WANDB_MODE", "online"),
    )
    args = parser.parse_args()
    summary = json.loads(args.summary.read_text())
    defaults = json.loads((ROOT / "reproduction/wandb_config.json").read_text())
    directory = args.summary.parent / "wandb_tracking"
    directory.mkdir(exist_ok=True)
    for key, subdir in (("WANDB_DATA_DIR", "data"), ("WANDB_CACHE_DIR", "cache")):
        path = directory / subdir
        path.mkdir(exist_ok=True)
        os.environ.setdefault(key, str(path.resolve()))
    for model in ("jsd_upstream", "ce"):
        identity_path = directory / f"{model}.json"
        identity = (
            json.loads(identity_path.read_text()) if identity_path.exists() else {}
        )
        audits = [a for a in summary["audit"] if a["model"] == model]
        config = dict(
            task="supplied_phone_forced_alignment",
            model=model,
            training=False,
            logging_origin="evaluation_summary",
            evaluation_root=summary["root"],
            max_shift_ms=20,
            checkpoint_sha256=audits[0]["checkpoint_sha256"],
            checkpoint=audits[0]["checkpoint"],
            full_set=True,
            dev_utterances=400,
            test_utterances=192,
            recognition_metrics="not measured: supplied phone transcript",
            protocol="All internal token boundaries incl sil/q/dx, endpoints excluded; directional matching",
        )
        options = dict(
            project=defaults["project"],
            entity=defaults.get("entity"),
            name=f'timit_frozen_cif_{model}_{Path(summary["root"]).name}',
            group=Path(summary["root"]).name,
            job_type="evaluation",
            config=config,
            dir=str(directory.resolve()),
            mode=args.mode,
            settings=wandb.Settings(init_timeout=35),
        )
        if identity:
            options.update(id=identity["id"], resume="allow")
        try:
            run = init_evaluation(**options)
        except (wandb.errors.CommError, wandb.errors.UsageError):
            if args.mode != "online":
                raise
            options.update(mode="offline")
            options.pop("resume", None)
            run = init_evaluation(**options)
        identity_path.write_text(
            json.dumps(
                dict(
                    id=run.id,
                    mode=options["mode"],
                    url=run.url if options["mode"] == "online" else None,
                    project=run.project,
                    entity=run.entity,
                ),
                indent=2,
            )
            + "\n"
        )
        try:
            values = {}
            for r in summary["metrics"]:
                if r["model"] != model:
                    continue
                stem = f"{r['split']}/{r['decoder']}"
                for key in ("mae_ms", "p90_ms"):
                    values[f"{stem}/{key}"] = r[key]
                for key in ("precision_pct", "recall_pct", "f1_pct", "r_value_pct"):
                    values[f"{stem}/{key}_{r['tolerance_ms']}ms"] = r[key]
            for r in summary["paired"]:
                if r["model"] == model:
                    for key in (
                        "delta_mae_ms",
                        "bootstrap95_lo_ms",
                        "bootstrap95_hi_ms",
                        "accepted",
                        "improved",
                        "worsened",
                        "unchanged",
                    ):
                        values[f"{r['split']}/paired/{key}"] = r[key]
            for r in summary["timing"]:
                if r["model"] == model:
                    for key in (
                        "refiner_mean_ms",
                        "refiner_median_ms",
                        "refiner_p90_ms",
                    ):
                        values[f"{r['split']}/readout_only/{key}"] = r[key]
            run.summary.update(values)
            artifact = wandb.Artifact(
                f'timit-frozen-cif-{model}-{Path(summary["root"]).name}',
                type="evaluation",
            )
            for name in (
                "summary.json",
                "report.md",
                "metrics.csv",
                "paired.csv",
                "timing.csv",
            ):
                artifact.add_file(str(args.summary.parent / name))
            run.log_artifact(artifact)
            identity_path.write_text(
                json.dumps(
                    dict(
                        id=run.id,
                        mode=options["mode"],
                        url=run.url if options["mode"] == "online" else None,
                        project=run.project,
                        entity=run.entity,
                    ),
                    indent=2,
                )
                + "\n"
            )
        finally:
            run.finish()


if __name__ == "__main__":
    main()
