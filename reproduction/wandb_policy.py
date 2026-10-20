"""Local-only evaluation by default; training scalar logging stays independent."""
import json
import os
from pathlib import Path


def evaluation_uploads_enabled():
    """Require an evaluation-specific opt-in, never the training WANDB_MODE."""
    override = os.environ.get("OTTC_EVAL_UPLOAD")
    if override is not None:
        return override.strip().lower() in {"1", "true", "yes"}
    config = Path(__file__).with_name("wandb_config.json")
    if not config.exists():
        return False
    return json.loads(config.read_text()).get("evaluation_uploads", False) is True


def init_evaluation(*args, **kwargs):
    """Keep existing evaluation logging calls usable without a cloud run."""
    import wandb
    if not evaluation_uploads_enabled():
        kwargs["mode"] = "disabled"
        kwargs.pop("resume", None)
    return wandb.init(*args, **kwargs)


def log_evaluation_artifact(run, *args, **kwargs):
    """Do not upload final evaluation files from training runs by default."""
    if evaluation_uploads_enabled():
        return run.log_artifact(*args, **kwargs)
    return None
