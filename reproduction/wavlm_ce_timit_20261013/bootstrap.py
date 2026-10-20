from pathlib import Path
import sys
import hashlib

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path[:0]=[str(ROOT/'reproduction/vendor'),str(ROOT),str(HERE),str(ROOT/'reproduction')]
import transformers
transformers.utils.import_utils._peft_available=False
transformers.utils.import_utils._bitsandbytes_available=False


def source_hashes():
    paths=[*sorted(HERE.glob('*.py')),ROOT/'run_exp.py',*sorted((ROOT/'ottc').rglob('*.py'))]
    return {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
