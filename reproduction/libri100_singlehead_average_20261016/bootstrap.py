"""Reuse the original singlehead dependencies and data without modifying them."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT = Path(__file__).resolve().parent
HERE = ROOT/'reproduction/libri100_singlehead_20261014'
sys.path[:0] = [str(ROOT/'reproduction/vendor'), str(HERE), str(ROOT/'reproduction'), str(ROOT)]
import transformers
transformers.utils.import_utils._peft_available = False
transformers.utils.import_utils._bitsandbytes_available = False
