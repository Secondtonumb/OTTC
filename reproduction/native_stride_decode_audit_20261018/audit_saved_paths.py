"""Validate saved acoustic paths against original IF-MDD arcs without inference."""
import ast
from collections import defaultdict
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
EXPERIMENT = ROOT / 'reproduction/native_stride_ottc_20261018'
VENDOR = ROOT / 'reproduction/zapformer_ifmdd_topology_ottc_20261013/vendor/utils/subphonetic_topology.py'
ORIGINAL = ROOT.parent / 'IF-MDD/utils/subphonetic_topology.py'


def source_helpers():
    tree = ast.parse(VENDOR.read_text())
    names = {'normalize_topology_name', '_state_count', '_add_phone_arcs', 'token_fst_text'}
    namespace = {'dataclass': dataclass, 're': re}
    selected = [ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0)]
    for node in tree.body:
        if isinstance(node, ast.Assign):
            try:
                value = ast.literal_eval(node.value)
            except (ValueError, TypeError):
                continue
            for target in node.targets:
                if isinstance(target, ast.Name):
                    namespace[target.id] = value
        elif isinstance(node, ast.ClassDef) and node.name == 'Arc':
            selected.append(node)
        elif isinstance(node, ast.FunctionDef) and node.name in names:
            selected.append(node)
    module = ast.fix_missing_locations(ast.Module(body=selected, type_ignores=[]))
    exec(compile(module, str(VENDOR), 'exec'), namespace)
    old = {n.name: ast.dump(n, include_attributes=False) for n in ast.parse(ORIGINAL.read_text()).body if isinstance(n, ast.FunctionDef)}
    same = {n.name: old.get(n.name) == ast.dump(n, include_attributes=False)
            for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names}
    assert all(same.values()), same
    return namespace, same


def accepted(path, hypothesis, arcs):
    current = {(0, 0)}
    for label in path:
        following = set()
        for state, index in current:
            for target, phone in arcs.get((state, int(label)), []):
                if phone == '<eps>':
                    following.add((target, index))
                elif index < len(hypothesis) and phone == hypothesis[index]:
                    following.add((target, index + 1))
        current = following
        if not current:
            return False
    return (0, len(hypothesis)) in current


def main():
    helpers, parity = source_helpers()
    records = []
    for model in sorted((EXPERIMENT / 'models').glob('wav2vec2__timit__10ms__s3t2ss__*')):
        vocab = json.loads((model / 'vocab_used.json').read_text())
        phones = sorted(p for p in vocab if p not in ('<pad>', '<unk>'))
        labels = ['<blank>'] + [f'{p}_{i}' for p in phones for i in range(1 if p == 'sil' else 3)]
        index = {label: i for i, label in enumerate(labels)}
        text = helpers['token_fst_text'](phones, 's3_t2_star_star', split_silence=False, blank_label='<blank>')
        (HERE / (model.name + '.token_fst.txt')).write_text(text)
        arcs = defaultdict(list)
        for line in text.splitlines():
            parts = line.split()
            if len(parts) == 4:
                source, target = map(int, parts[:2])
                arcs[(source, index[parts[2]])].append((target, parts[3]))
        for path in sorted(model.glob('benchmark_*.predictions.jsonl')):
            total = fsa_accepted = greedy_accepted = fsa_failures = fa_failures = 0
            ids = set()
            for line in path.open():
                item = json.loads(line); total += 1
                assert item['utt'] not in ids; ids.add(item['utt'])
                assert item['reference_times_used_by_aligner'] is False
                decoded = item['fsa_text_free']
                if 'failure' in decoded:
                    fsa_failures += 1
                else:
                    good = accepted(decoded['frame_token_ids'], decoded['hypothesis'], arcs)
                    assert good, (path, item['utt'])
                    fsa_accepted += good
                greedy_accepted += accepted(item['text_free']['frame_token_ids'], item['text_free']['hypothesis'], arcs)
                fa_failures += 'failure' in item['forced_alignment']
            expected = 400 if '_dev.' in path.name else 192
            assert total == len(ids) == expected, (path, total)
            records.append(dict(recipe=model.name, predictions=str(path), utterances=total,
                                valid_topology_FSA_paths=fsa_accepted, FSA_failures=fsa_failures,
                                valid_unconstrained_greedy_paths=greedy_accepted,
                                forced_alignment_failures=fa_failures))
    report = dict(original_IFMDD_function_AST_parity=parity,
                  vendor_sha256=hashlib.sha256(VENDOR.read_bytes()).hexdigest(),
                  total_utterances=sum(r['utterances'] for r in records),
                  valid_FSA_paths=sum(r['valid_topology_FSA_paths'] for r in records),
                  valid_greedy_paths=sum(r['valid_unconstrained_greedy_paths'] for r in records),
                  forced_alignment_failures=sum(r['forced_alignment_failures'] for r in records),
                  records=records)
    (HERE / 'SAVED_PATH_AUDIT.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k != 'records'}, indent=2))


if __name__ == '__main__':
    main()
