import ast
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

HERE = Path(__file__).resolve().parent


def test_retention_keeps_only_three_best_and_best_link(tmp_path):
    namespace = dict(json=json, Path=Path)
    def write(path, value):
        path.write_text(json.dumps(value))
    def weights(model, path, metadata):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(metadata))
    namespace.update(write=write, atomic_weights=weights)
    module = ast.parse((HERE / 'common.py').read_text())
    node = next(node for node in module.body if isinstance(node, ast.FunctionDef) and node.name == 'retain_best')
    exec(compile(ast.Module(body=[node], type_ignores=[]), '<retention>', 'exec'), namespace)
    config = dict(name='jdiv__seed43', seed=43, source_sha256={})
    model = SimpleNamespace(config=SimpleNamespace(to_dict=lambda: {}))
    for step, score in enumerate([30., 20., 25., 15., 22., 40.], 1):
        namespace['retain_best'](tmp_path, model, score, step, step, config)
    index = json.loads((tmp_path / 'BEST_THREE.json').read_text())
    assert [item['PER'] for item in index['checkpoints']] == [15., 20., 22.]
    assert len(list((tmp_path / 'best_weights').glob('*.pt'))) == 3
    assert (tmp_path / 'best_dev.pt').is_symlink()
    assert (tmp_path / 'best_dev.pt').resolve().name == 'step-000004.pt'
    assert not (tmp_path / 'epoch_weights').exists()


def test_pfer_crosschecks_per_and_uses_audio_only_hypothesis(tmp_path):
    from reproduction.timit_seed_sweep_20261022.pfer import score_predictions
    path = tmp_path / 'predictions.jsonl'
    path.write_text(json.dumps(dict(utt='one', reference=[['sil', 0., .1], ['d', .1, .2], ['iy', .2, .3]],
                    text_free=dict(hypothesis=['dx', 'iy']), forced_alignment=dict(hypothesis=['wrong']))) + '\n')
    result = score_predictions(path, dict(utterances=1, modes=dict(text_free=dict(nosil_PER_percent=0.))))
    assert result['PFER_percent'] == pytest.approx(0.)
    assert result['reference_IPA_segments'] == 2
