"""Isolated Buckeye recipe using the reproduced upstream OTTC implementation."""
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT / 'reproduction/vendor'), str(ROOT / 'reproduction'), str(ROOT)]
import transformers
transformers.utils.import_utils._peft_available = False
transformers.utils.import_utils._bitsandbytes_available = False

