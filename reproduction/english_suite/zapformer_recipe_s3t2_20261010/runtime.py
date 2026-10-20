"""Load the isolated archived Zapformer recipe before any model imports."""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE/'vendor/zapformer'), str(HERE/'vendor'), str(HERE)]

