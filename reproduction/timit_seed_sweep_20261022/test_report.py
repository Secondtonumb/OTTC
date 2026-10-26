import importlib.util
from pathlib import Path
import sys

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
spec = importlib.util.spec_from_file_location('seed_sweep_report', HERE / 'report.py')
report = importlib.util.module_from_spec(spec)
spec.loader.exec_module(report)


def result(seed, value):
    return dict(cost='ce', seed=seed, **{key: value for key in report.METRICS})


def test_matched_seeds_exclude_supplemental_seed():
    rows = [result(42, 1.), result(43, 2.), result(44, 3.), result(45, 100.)]
    summary = report.summarize(rows, (42, 43, 44))[0]
    assert summary['seeds'] == [42, 43, 44]
    assert summary['complete']
    assert summary['metrics']['PER']['mean'] == 2.
    assert summary['metrics']['PER']['sample_std'] == 1.


def test_missing_seed_is_incomplete():
    summary = report.summarize([result(42, 1.)], (42, 43, 44))[0]
    assert not summary['complete']
    assert summary['n'] == 1
    assert summary['metrics']['PER']['sample_std'] is None


def test_duplicate_seed_rejected():
    with pytest.raises(ValueError, match='Duplicate'):
        report.summarize([result(42, 1.), result(42, 2.)], (42, 43, 44))
