"""Freeze existing TIMIT scorers, using official subsets and separate TF/native-FA metrics."""
import ast
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

import jiwer
import torch.distributed as dist

HERE = Path(__file__).resolve().parent
SUITE = HERE.parent/'english_suite'
SOURCE = SUITE/'mfa_metrics.py'
sys.path.insert(0, str(SUITE))
sys.path.insert(0, '/work/gm64/m64000/fa-bench/analysis/mfa_spottc_jsd_s3t2ss')
sys.path.insert(0, str(HERE.parent/'ipa_arpabet_comparison_20261012'))
from rescore import mapped


def evaluate(net, split, rank, world, output, limit=0):
    output.mkdir(parents=True, exist_ok=True)
    tree = ast.parse(SOURCE.read_text())
    fn = next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='evaluate')
    for node in ast.walk(fn):
        if isinstance(node,ast.ImportFrom) and node.module=='ifmdd_k2_alignment':
            node.module = 'native_alignment'
        if isinstance(node,ast.Call) and isinstance(node.func,ast.Name) and node.func.id=='align':
            node.args[0] = ast.parse('scores.log_softmax(-1)',mode='eval').body
            node.keywords.append(ast.keyword(arg='topology',value=ast.Constant('s3_t2')))
    ast.fix_missing_locations(tree)
    ns = {'__file__':str(SOURCE),'__name__':'framece_frozen_timit_metrics'}
    exec(compile(tree,str(SOURCE),'exec'),ns)
    ns['DATA'] = HERE/'eval_data'
    result = ns['evaluate'](net, 'spottc', split=split, rank=rank, world=world,
                            forced=True, output=output, limit=limit)
    references = {r['id']:r for r in (json.loads(line) for line in (HERE/'eval_data'/f'{split}.jsonl').open())}
    count = Counter()
    for row in (json.loads(line) for line in (output/f'{split}_predictions.jsonl').open()):
        ref = mapped(references[row['id']]['tokens'], 'english_collapsed')
        hyp = mapped(row['predicted_ids'], 'english_collapsed')
        edits = jiwer.process_words(' '.join(ref),' '.join(hyp))
        count.update(reference_phones=len(ref), substitutions=edits.substitutions,
                     deletions=edits.deletions, insertions=edits.insertions, utterances=1)
    if world > 1:
        gathered = [None]*world; dist.all_gather_object(gathered, dict(count))
        count = Counter()
        for local in gathered: count.update(local)
    n = count['reference_phones']
    result['mapped_arpabet'] = dict(count)
    for k in ('substitutions','deletions','insertions'):
        result['mapped_arpabet'][k+'_percent'] = 100*count[k]/n
    result['mapped_arpabet']['PER_percent'] = 100*sum(count[k] for k in ('substitutions','deletions','insertions'))/n
    result['protocol'].update(subset='official dev400' if split=='development' else 'official core-test192',
        fa='Exact ordinary S3T2 state FSA with supplied phones and silence; previous blank; duration/frames; no posterior merge',
        training_topology='s3_t2', mapped_recognition='IPA -> English-collapsed ARPAbet/TIMIT39; frozen native speech reference and jiwer',
        source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        adapter_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        test_checkpoint_selection=False)
    (output/'metrics.json').write_text(json.dumps(result,indent=2)+'\n')
    return result
