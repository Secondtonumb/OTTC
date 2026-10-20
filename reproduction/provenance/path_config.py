from pathlib import Path

REPRODUCTION_ROOT = Path(__file__).resolve().parents[2] / "reproduction"
DATASETS_ROOT_PATH = Path("/work/gm64/m64000/dataset")
HUGGINGFACE_CACHE = REPRODUCTION_ROOT / "cache"
LARGE_MODELS_PATH = REPRODUCTION_ROOT / "models"
TIMIT_DATASET_PATH = DATASETS_ROOT_PATH / "TIMIT"
