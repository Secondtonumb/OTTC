"""Reuse the existing frozen Zapformer recipe and original upstream OTTC loss."""
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
NATIVE = ROOT / 'reproduction/zapformer_s3t2_native_20261012'
sys.path.insert(0, str(NATIVE))
import runtime  # noqa: F401; adds the immutable native Zapformer code
sys.path.insert(0, str(HERE))
sys.path.extend([str(ROOT), str(ROOT / 'reproduction')])
