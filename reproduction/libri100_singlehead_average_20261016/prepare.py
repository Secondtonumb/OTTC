"""Freeze the retained-checkpoint pool and create a parameterized evaluator."""
import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
SOURCE = ROOT/'reproduction/libri100_singlehead_20261014'
OBJECTIVES = ['ce', 'jsd_upstream', 'jsd_standard']


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def replace_once(source, old, new, count=1):
    assert source.count(old) == count, (old, source.count(old))
    return source.replace(old, new)


def main():
    if (HERE/'plan.json').exists():
        plan = json.loads((HERE/'plan.json').read_text())
        assert sha(SOURCE/'evaluate.py') == plan['source_evaluator_sha256']
        print('Frozen checkpoint pool already exists; reuse')
        return
    original = (SOURCE/'evaluate.py').read_text()
    adapted = replace_once(original, "    parser.add_argument('--smoke',action='store_true')", """    parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--splits',nargs='+',choices=EVALUATION_SPLITS,default=EVALUATION_SPLITS)
    parser.add_argument('--label',required=True)""")
    start = adapted.index('    directory=model_directory(args.objective,args.smoke)')
    end = adapted.index("    if (output/'EVAL_COMPLETE.json').exists():", start)
    adapted = adapted[:start]+"""    directory=model_directory(args.objective,False)
    checkpoint=args.checkpoint.resolve()
    output=args.output.resolve()
    assert (checkpoint/'model.safetensors').exists()
"""+adapted[end:]
    adapted = replace_once(adapted, "        print('Evaluation already complete; reuse',output,flush=True);return", """        saved=json.loads((output/'EVAL_COMPLETE.json').read_text())
        assert saved['checkpoint']==str(checkpoint) and saved['smoke']==args.smoke
        assert set(saved['utterances'])==set(args.splits)
        print('Evaluation already complete; reuse',output,flush=True);return""")
    adapted = replace_once(adapted, "dict(objective=args.objective,checkpoint=str(checkpoint),", "dict(objective=args.objective,artifact_label=args.label,checkpoint=str(checkpoint),", count=2)
    adapted = replace_once(adapted, "for split in (['dev_clean'] if args.smoke else EVALUATION_SPLITS):", 'for split in args.splits:')
    adapted = replace_once(adapted, "selection='final epoch40, never select using test',reference_times_used_by_aligner=False,", "selection='Retained single checkpoint or uniform weight average; dev-only selection; see selection.json',source_evaluator_sha256='"+sha(SOURCE/'evaluate.py')+"',reference_times_used_by_aligner=False,")
    adapted = replace_once(adapted, "'libri100-singlehead-'+result['objective']+'-K2FA'", "'libri100-singlehead-'+result['objective']+'-'+result['artifact_label']+'-K2FA'", count=2)
    adapted = replace_once(adapted, "group='libri100_singlehead_20261014'", "group='libri100_singlehead_average_20261016'")
    # The scoring functions and inference loop body must remain bytecode-equivalent ASTs.
    before = ast.parse(original); after = ast.parse(adapted)
    for name in ['edges', 'new_counts', 'score', 'summary']:
        a = next(n for n in before.body if isinstance(n, ast.FunctionDef) and n.name == name)
        b = next(n for n in after.body if isinstance(n, ast.FunctionDef) and n.name == name)
        assert ast.dump(a) == ast.dump(b), name
    old_loop = next(n for n in ast.walk(before) if isinstance(n, ast.For) and isinstance(n.target, ast.Name) and n.target.id == 'split')
    new_loop = next(n for n in ast.walk(after) if isinstance(n, ast.For) and isinstance(n.target, ast.Name) and n.target.id == 'split')
    assert [ast.dump(n) for n in old_loop.body] == [ast.dump(n) for n in new_loop.body]
    compile(adapted, str(HERE/'evaluate_checkpoint.py'), 'exec')
    (HERE/'evaluate_checkpoint.py').write_text(adapted)
    (HERE/'source_evaluator_snapshot.py').write_text(original)
    pools = {}
    for objective in OBJECTIVES:
        directory = ROOT/'reproduction/models/large_models_results/wav2vec2-large-lv60'/('runexp_libri100_singlehead_'+objective+'_20261014')
        complete = json.loads((directory/'TRAIN_COMPLETE.json').read_text())
        assert complete['epoch'] == 40 and complete['global_step'] == 17840
        pool = []
        for p in sorted(directory.glob('checkpoint-*'), key=lambda p: int(p.name.split('-')[-1])):
            if not (p/'model.safetensors').exists():
                continue
            state = json.loads((p/'trainer_state.json').read_text())
            pool.append(dict(checkpoint=str(p), step=state['global_step'], epoch=state['epoch'],
                model_bytes=(p/'model.safetensors').stat().st_size, config_sha256=sha(p/'config.json')))
        assert len(pool) >= 3
        assert len({p['config_sha256'] for p in pool}) == 1
        pools[objective] = dict(directory=str(directory), candidates=pool, last3=[p['step'] for p in pool[-3:]],
            original_final_evaluation=str(SOURCE/'evaluation'/objective), original_final_checkpoint=complete['checkpoint'])
    plan = dict(objectives=OBJECTIVES, count=3, checkpoint_scope='Only retained weights; epochs39-40, not global best over the full training history',
        policies=['last3', 'dev_fa_best3', 'dev_per_best3'],
        selection_data=['dev_clean', 'dev_other'],
        selection_rules={'dev_fa_best3':'pooled K2 FA F1@20ms descending; ties lower FA MAE, lower pooled no-sil PER, later step',
                         'dev_per_best3':'pooled no-sil phone edit rate ascending; ties higher K2 FA F1@20ms, lower FA MAE, later step'},
        test_and_buckeye_used_for_selection=False, weighting='uniform arithmetic weight average, FP64 accumulation, original output dtype',
        optimizer_scheduler_rng_averaged=False, reuse_final_checkpoint_evaluation=True,
        splits=['dev_clean', 'dev_other', 'test_clean', 'test_other', 'buckeye_dev', 'buckeye_test'],
        source_evaluator_sha256=sha(SOURCE/'evaluate.py'), adapter_sha256=sha(HERE/'evaluate_checkpoint.py'),
        scoring_and_inference_loop_AST_identical=True, pools=pools)
    (HERE/'plan.json').write_text(json.dumps(plan, indent=2, ensure_ascii=False)+'\n')
    print(json.dumps({o:dict(candidates=[p['step'] for p in v['candidates']],last3=v['last3']) for o,v in pools.items()},indent=2))


if __name__ == '__main__':
    main()
