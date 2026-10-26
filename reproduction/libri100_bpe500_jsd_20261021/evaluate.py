"""Local distributed recognition and separate human-reference TIMIT transfer scores."""
import json
from pathlib import Path

import torch
import torch.distributed as dist
from reproduction.libri100_bpe500_jsd_20261021 import decoder
from reproduction.timit_bpe500_stages_20261020 import train as parent
from reproduction.timit_bpe500_50ep_20261020 import segmentation

base = parent.base
COUNT_KEYS = ('word_errors', 'reference_words', 'char_errors', 'reference_characters', 'utterances')


def combined_metrics(records):
    counts = {key: sum(record[key] for record in records) for key in COUNT_KEYS}
    assert counts['reference_words'] and counts['reference_characters']
    return dict(counts, WER_percent=100 * counts['word_errors'] / counts['reference_words'],
                CER_percent=100 * counts['char_errors'] / counts['reference_characters'])


def evaluate_split(model, rows, processor, tokenizer, topology, vector, destination, word_references=None):
    rank = dist.get_rank()
    local_rows = rows[rank::dist.get_world_size()]
    modes = ['greedy'] + (['s3t2ss'] if topology is not None else [])
    destination.mkdir(parents=True, exist_ok=True)
    paths = {mode: destination / mode / f'rank-{rank}.jsonl' for mode in modes}
    for path in paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    streams = {mode: path.open('w') for mode, path in paths.items()}
    references, hypotheses = [], {mode: [] for mode in modes}
    model.eval()
    try:
        with torch.inference_mode():
            for offset in range(0, len(local_rows), 2):
                subset = local_rows[offset:offset + 2]
                batch = base.Collator(processor)(subset)
                batch.pop('labels')
                device = next(model.parameters()).device
                batch = {key: value.to(device) for key, value in batch.items()}
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    output = model(**batch)
                lengths = output.hidden_states[1].sum(-1).long().tolist()
                for index, (row, length) in enumerate(zip(subset, lengths)):
                    scores = output.logits[index, :length].float()
                    raw_path = scores.argmax(-1).tolist()
                    predictions = {'greedy': (base.decode_path(raw_path, tokenizer, topology), raw_path)}
                    if topology is not None:
                        tokens, path, _ = decoder.best_path(scores, topology, vector)
                        hypothesis = base.normalize_text(tokenizer.decode(tokens))
                        assert hypothesis == base.decode_path(path, tokenizer, topology)
                        predictions['s3t2ss'] = (hypothesis, path)
                    references.append(row['text'])
                    for mode, (hypothesis, path) in predictions.items():
                        hypotheses[mode].append(hypothesis)
                        streams[mode].write(json.dumps(dict(utt=row['utt'], reference=row['text'],
                            hypothesis=hypothesis, frame_ids=path, mode=mode, transcript_supplied=False)) + '\n')
    finally:
        for stream in streams.values():
            stream.close()
    for mode in modes:
        parent.write_json(paths[mode].with_suffix('.metrics.json'), base.text_metrics(references, hypotheses[mode]))
    dist.barrier()
    result = {}
    if rank == 0:
        for mode in modes:
            records = [parent.read_json(destination / mode / f'rank-{worker}.metrics.json') for worker in range(dist.get_world_size())]
            metrics = combined_metrics(records)
            assert metrics['utterances'] == len(rows)
            by_id = {}
            for worker in range(dist.get_world_size()):
                with (destination / mode / f'rank-{worker}.jsonl').open() as stream:
                    for line in stream:
                        record = json.loads(line)
                        assert record['utt'] not in by_id
                        by_id[record['utt']] = record
            merged = destination / (mode + '.jsonl')
            with merged.open('w') as stream:
                for row in rows:
                    stream.write(json.dumps(by_id[row['utt']]) + '\n')
            if word_references is not None:
                metrics['segmentation'] = segmentation.score_predictions(merged, word_references, tokenizer, topology)
            metrics['predictions_sha256'] = base.sha256(merged)
            metrics['evaluation_uploads'] = False
            result[mode] = metrics
        parent.write_json(destination / 'RESULTS.json', result)
    dist.barrier()
    return result


def benchmarks(model, rows, processor, tokenizer, topology, output, config):
    vector = decoder.build(tokenizer, next(model.parameters()).device)[1] if topology is not None else None
    members = sorted((output / 'epoch_weights').glob('epoch-*.pt'))
    best = parent.read_json(output / 'BEST_DEV.json')
    candidates = {'best_dev': dict(path=str(output / 'best_dev.pt'), step=best['step'],
                                  sha256=base.sha256(output / 'best_dev.pt'))}
    for count in [5, 10]:
        paths = members[-count:]
        metadata = [dict(path=str(path), **parent.read_json(path.with_suffix('.json'))) for path in paths]
        assert [record['epoch'] for record in metadata] == list(range(51 - count, 51))
        candidates['last' + str(count)] = dict(members=metadata, step=config['planned_optimizer_steps'])
    modes = ['greedy'] + (['s3t2ss'] if topology is not None else [])
    results = dict(candidates={}, selections={}, protocol=config, evaluation_uploads=False)
    for phase, splits in [('dev', ['dev-clean', 'dev-other']), ('test', ['test-clean', 'test-other'])]:
        for name, candidate in candidates.items():
            if name == 'best_dev':
                state = torch.load(candidate['path'], map_location='cpu', weights_only=False, mmap=True)['model']
            else:
                state, _ = base.average([Path(member['path']) for member in candidate['members']], mfa=True)
            model.load_state_dict(state, strict=True)
            del state
            model.step = candidate['step'] - 1
            results['candidates'].setdefault(name, dict(provenance=candidate))
            for split in splits:
                result = evaluate_split(model, rows[split], processor, tokenizer, topology, vector,
                                        output / 'evaluation' / name / split)
                if dist.get_rank() == 0:
                    results['candidates'][name][split] = result
                    parent.write_json(output / 'RESULTS.json', results)
        if phase == 'dev' and dist.get_rank() == 0:
            for mode in modes:
                results['selections'][mode] = min(candidates, key=lambda name: combined_metrics([
                    results['candidates'][name][split][mode] for split in splits])['WER_percent'])
            parent.write_json(output / 'DEV_SELECTION.json', dict(selections=results['selections'],
                criterion='Pooled full dev-clean/dev-other WER within each decoding protocol', test_used=False))
        dist.barrier()
    timit_rows = {split: [json.loads(line) for line in (parent.DATA / (split + '.jsonl')).open()]
                  for split in ['dev', 'test']}
    references = segmentation.load_references(timit_rows)
    if dist.get_rank() == 0:
        parent.write_json(output / 'timit_transfer/word_references.json', references)
    for name, candidate in candidates.items():
        if name == 'best_dev':
            state = torch.load(candidate['path'], map_location='cpu', weights_only=False, mmap=True)['model']
        else:
            state, _ = base.average([Path(member['path']) for member in candidate['members']], mfa=True)
        model.load_state_dict(state, strict=True)
        del state
        model.step = candidate['step'] - 1
        for split in ['dev', 'test']:
            result = evaluate_split(model, timit_rows[split], processor, tokenizer, topology, vector,
                output / 'timit_transfer' / name / split, references[split])
            if dist.get_rank() == 0:
                results['candidates'][name]['timit_transfer_' + split] = result
    if dist.get_rank() == 0:
        results['timit_transfer_protocol'] = 'Separate zero-shot TIMIT human-word-boundary results; not Libri alignment ground truth; no TIMIT training or selection.'
        parent.write_json(output / 'RESULTS.json', results)
        parent.write_json(output / 'BENCHMARK_COMPLETE.json', dict(epoch=50, step=config['planned_optimizer_steps'],
            results_sha256=base.sha256(output / 'RESULTS.json'), selections=results['selections'], evaluation_uploads=False))
    dist.barrier()