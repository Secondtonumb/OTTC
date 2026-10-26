from contextlib import nullcontext
import json
import os
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest
import torch

from reproduction.libri100_timit_seed_sweep_20261022 import common, evaluate, train

HERE = Path(__file__).resolve().parent


def test_twelve_independent_worker_assignments():
    assignments = []
    for rank in range(12):
        result = subprocess.run(['bash', str(HERE / 'worker.sh'), '--dry-run'],
                                env=dict(os.environ, OMPI_COMM_WORLD_RANK=str(rank)),
                                text=True, capture_output=True, check=True)
        fields = dict(field.split('=', 1) for field in result.stdout.strip().split())
        assert fields['batch'] == '64'
        assert fields['GPUs'] == fields['world_size'] == '1'
        assert fields['epochs'] == '40'
        assignments.append(fields['recipe'])
    assert set(assignments) == {f'{cost}__seed{seed}' for cost in common.COSTS for seed in common.SEEDS}


def test_matrix_changes_only_cost_and_seed():
    configs = [common.recipe(f'{cost}__seed{seed}') for cost in common.COSTS for seed in common.SEEDS]
    excluded = {'name', 'cost', 'seed', 'bins_metric', 'objective'}
    controls = [{key: value for key, value in config.items() if key not in excluded} for config in configs]
    assert all(config == controls[0] for config in controls)
    config = configs[0]
    assert config['global_batch'] == 64 and config['epochs'] == 40
    assert config['corpus'] == 'libri100' and config['frame_ms'] == 10
    assert config['forward_type'] == 'regular'
    assert config['freeze_cnn_first_six'] and config['train_cnn_last_layer']
    assert config['evaluation_uploads'] is False


def test_training_and_selection_never_use_timit():
    audit = common.expected_data()
    assert audit['train']['corpus'] == audit['dev']['corpus'] == 'libri100'
    assert audit['train']['count'] == 28538 and audit['dev']['count'] == 5567
    assert audit['timit_dev']['count'] == 400 and audit['timit_test']['count'] == 192
    assert 'train-dev.jsonl' in audit['timit_dev']['manifest']
    assert 28538 % 64 == 58 and (28538 + 63) // 64 * 40 == 17840


def test_best_selection_uses_only_libri_canonical_metric(tmp_path, monkeypatch):
    retained = []
    monkeypatch.setattr(common, 'retain_best', lambda *args: retained.append(args))
    logged = []
    config = dict(selection_corpus='LibriSpeech dev-clean+dev-other')
    policy = train.Policy(tmp_path, config, [], {}, None, None, float('inf'), SimpleNamespace(log=logged.append))
    state = SimpleNamespace(epoch=1., global_step=446)
    policy.on_evaluate(None, state, None, model='model', metrics={'eval_nosil_PER_percent': 12.5})
    assert retained[0][2] == 12.5
    assert logged[0]['selection_corpus'] == config['selection_corpus']


@pytest.mark.parametrize('count', [64, 58])
def test_chunked_backward_matches_full_batch_mean(count):
    class SumLoss(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(2.))

        def forward(self, values):
            return SimpleNamespace(loss=((self.weight * values) ** 2).sum())

    values = torch.arange(count, dtype=torch.float32) / count
    full = SumLoss()
    (full(values).loss / count).backward()
    chunked = SumLoss()
    trainer = SimpleNamespace(_prepare_inputs=lambda batch: batch,
                              compute_loss_context_manager=nullcontext,
                              accelerator=SimpleNamespace(backward=lambda loss: loss.backward()))
    inputs = dict(microbatches=[dict(values=values[offset:offset + 8]) for offset in range(0, count, 8)],
                  logical_batch_size=count)
    train.QuietTrainer.training_step(trainer, chunked, inputs)
    assert torch.allclose(full.weight.grad, chunked.weight.grad)


def test_final_evaluates_timit_only_without_wandb(tmp_path, monkeypatch):
    class CPUModel(torch.nn.Linear):
        def cuda(self):
            return self

    model = CPUModel(1, 1)
    torch.save(dict(model=model.state_dict(), step=446), tmp_path / 'best_dev.pt')
    requested = []

    def fake_evaluate(config, model, rows, vocab, processor, topo, target):
        requested.append(len(rows))
        return dict(utterances=len(rows), modes=dict(text_free={}, forced_alignment=dict(alignment_failures=0)))

    monkeypatch.setattr(evaluate, 'evaluate', fake_evaluate)
    monkeypatch.setattr(evaluate, 'score_predictions', lambda *args: dict(PFER_percent=1.))
    config = dict(seed=42, cost='ce', selection_corpus='Libri dev', selection_reference='canonical PER',
                  expected_data={split: dict(corpus='timit') for split in ('timit_dev', 'timit_test')})
    rows = dict(dev=['not a final evaluation'], timit_dev=[None] * 400, timit_test=[None] * 192)
    evaluate.final_benchmarks(config, model, rows, {}, None, None, tmp_path, object())
    assert requested == [400, 192]
    complete = json.loads((tmp_path / 'BENCHMARK_COMPLETE.json').read_text())
    assert complete['evaluated_splits'] == ['timit_dev', 'timit_test']
    assert complete['timit_dev_used_for_selection'] is False
    assert not (tmp_path / 'benchmark_best_dev_dev.json').exists()
