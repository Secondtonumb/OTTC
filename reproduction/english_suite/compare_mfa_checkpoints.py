"""Dev-only selection, individual tests and last/best-N MFA weight averages."""
import argparse
import gc
import hashlib
import json
import os
import sys
from pathlib import Path

SUITE = Path(__file__).resolve().parent
REPRO = SUITE.parent
BASE = Path('/work/gm64/m64000/zipa/mfa_miyabi_20260922/zipa')
sys.path[:0] = [str(BASE/'experiments/mfa_zapformer'), str(REPRO), str(SUITE)]
import torch
from average_checkpoints import average, sha256
from model_10ms import MFAZapformer
from mfa_metrics import evaluate, DATA


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.partial')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')
    temporary.replace(path)


def load(path, arm):
    state = torch.load(path, map_location='cpu', weights_only=False, mmap=True)
    net = MFAZapformer(arm)
    net.load_state_dict(state['model'], strict=True)
    del state
    return net.cuda().eval()


def main():
    from reproduction.wandb_policy import init_evaluation
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--limit', type=int, default=0)
    args = parser.parse_args()
    torch.set_num_threads(4)
    plan = json.loads(args.plan.read_text())
    arm = plan['objective']
    count = plan['count']
    output = Path(plan['output'])
    if args.limit:
        output = output/'validation'
    output.mkdir(parents=True, exist_ok=True)
    if (output/'COMPARE_COMPLETE.json').exists():
        print('Already complete', output, flush=True)
        return
    candidates = plan['candidates']
    if args.limit:
        candidates = candidates[-count:]
    provenance = {'plan_sha256': sha256(args.plan), 'evaluator_sha256': sha256(Path(__file__)),
                  'metric_source_sha256': sha256(SUITE/'mfa_metrics.py'),
                  'development_manifest_sha256': sha256(DATA/'development.jsonl'),
                  'test_manifest_sha256': sha256(DATA/'test.jsonl'),
                  'phone_mapping_sha256': json.loads((DATA/'PREPARE_COMPLETE.json').read_text())['mapping_source_sha256'],
                  'intermediate_diagnostic': True, 'limit': args.limit,
                  'selection': 'dev text-free PER ascending; tie -> newer step; test unused',
                  'candidate_scope': 'all saved model checkpoints frozen at plan creation',
                  'lastN_definition': 'last N saved optimizer-step checkpoints, not N epochs', 'count': count}
    dev = {}
    for candidate in candidates:
        step = candidate['step']
        directory = output/'individual'/str(step)
        path = directory/'dev.json'
        if path.exists():
            result = json.loads(path.read_text())
            assert result['checkpoint_sha256'] == sha256(candidate['checkpoint'])
        else:
            directory.mkdir(parents=True, exist_ok=True)
            checkpoint = Path(candidate['checkpoint'])
            result = {'step': step, 'checkpoint': str(checkpoint),
                      'checkpoint_sha256': sha256(checkpoint), **provenance}
            prior = SUITE/'mfa_evaluation'/arm/('step-'+str(step))/'metrics.json'
            if not args.limit and prior.exists() and (prior.parent/'EVAL_COMPLETE.json').exists():
                earlier = json.loads(prior.read_text())
                if earlier.get('checkpoint_sha256') == result['checkpoint_sha256'] and earlier['metrics_source_sha256'] == provenance['metric_source_sha256']:
                    result['development'] = earlier['splits']['development']
                    result['evaluation_reused_from'] = str(prior)
            if 'development' not in result:
                net = load(checkpoint, arm)
                result['development'] = evaluate(net, arm, limit=args.limit, output=directory)
                del net; gc.collect(); torch.cuda.empty_cache()
            write(path, result)
        dev[step] = result['development']['text_free']['PER_percent']
        print('DEV', arm, step, dev[step], flush=True)
    last = sorted(candidates, key=lambda c: c['step'])[-count:]
    best = sorted(candidates, key=lambda c: (dev[c['step']], -c['step']))[:count]
    if len(last) != count or len(best) != count:
        raise RuntimeError('Not enough available checkpoints')
    selected = {'last'+str(count): [c['step'] for c in last], 'best'+str(count): [c['step'] for c in best],
                'best_single': best[0]['step'], 'development_scores': dev, **provenance}
    write(output/'selection.json', selected)  # Freeze dev selection before any tests.
    tests = {}
    for candidate in {c['step']: c for c in last+best}.values():
        step = candidate['step']; directory = output/'individual'/str(step)
        path = directory/'test.json'
        if path.exists():
            result = json.loads(path.read_text())
        else:
            net = load(Path(candidate['checkpoint']), arm)
            result = evaluate(net, arm, 'test', forced=True, limit=args.limit, output=directory)
            del net; gc.collect(); torch.cuda.empty_cache()
            write(path, result)
        tests[str(step)] = result
    averages = {}
    for name, chosen in [('last'+str(count)+'_weight_average', last), ('best'+str(count)+'_weight_average', best)]:
        directory = output/name; directory.mkdir(exist_ok=True)
        path = directory/'metrics.json'
        if path.exists():
            averages[name] = json.loads(path.read_text())
            continue
        chosen = sorted(chosen, key=lambda c: c['step'])
        state, policy = average([Path(c['checkpoint']) for c in chosen], mfa=True)
        torch.save({'model': state, 'inference_only': True,
                    'source_steps': [c['step'] for c in chosen], 'averaging': policy}, directory/'model.pt')
        net = MFAZapformer(arm); net.load_state_dict(state, strict=True); del state
        net = net.cuda().eval()
        result = {'source_steps': [c['step'] for c in chosen], 'averaging': policy,
                  'averaged_model_sha256': sha256(directory/'model.pt'), **provenance,
                  'development': evaluate(net, arm, limit=args.limit, output=directory),
                  'test': evaluate(net, arm, 'test', forced=True, limit=args.limit, output=directory)}
        del net; gc.collect(); torch.cuda.empty_cache()
        write(path, result); averages[name] = result
    # Metric means describe variation between individual models, distinct from weight averaging.
    import statistics
    individual_summary = {}
    for name, chosen in [('last'+str(count)+'_individual', last), ('best'+str(count)+'_individual', best)]:
        values = [tests[str(c['step'])]['text_free']['PER_percent'] for c in chosen]
        individual_summary[name] = {'steps': [c['step'] for c in chosen],
                                    'test_PER_mean_percent': statistics.mean(values),
                                    'test_PER_population_std_percent': statistics.pstdev(values)}
    result = {**provenance, 'objective': arm, 'selection': selected, 'individual_tests': tests,
              'weight_averages': averages, 'individual_metric_summaries': individual_summary}
    write(output/'comparison.json', result)
    import wandb
    identity_path = output/'wandb_run.json'
    identity = json.loads(identity_path.read_text()) if identity_path.exists() else {}
    options = dict(project='ottc-mfa-english', entity='kevingenghaopeng-',
                   name=arm+'-checkpoint-last'+str(count)+'-best'+str(count)+('-validation' if args.limit else ''),
                   group='mfa_checkpoint_comparison', job_type='evaluation', dir=str(output),
                   id=identity.get('id'), resume='allow' if identity else None,
                   config={**plan, **provenance}, settings=wandb.Settings(init_timeout=45))
    try:
        run = init_evaluation(**options)
    except wandb.errors.CommError:
        options.update(mode='offline'); options.pop('resume', None); run=init_evaluation(**options)
    run.define_metric('candidate_step')
    run.define_metric('dev/*', step_metric='candidate_step')
    for step, per in sorted(dev.items()):
        run.log({'candidate_step': step, 'dev/PER_percent': per})
    rows = []
    for step, modes in tests.items():
        rows.append(['individual', step, dev[int(step)], modes['text_free']['PER_percent'],
                     modes['text_free']['boundary_F1_20ms_percent'], modes['forced_k2']['boundary_F1_20ms_percent']])
    for name, value in averages.items():
        rows.append([name, ','.join(map(str,value['source_steps'])), value['development']['text_free']['PER_percent'],
                     value['test']['text_free']['PER_percent'], value['test']['text_free']['boundary_F1_20ms_percent'],
                     value['test']['forced_k2']['boundary_F1_20ms_percent']])
        run.summary.update({name+'/'+key: val for key,val in value['test']['text_free'].items() if isinstance(val,(int,float))})
        run.summary.update({name+'/forced_k2/'+key: val for key,val in value['test']['forced_k2'].items() if isinstance(val,(int,float))})
    run.summary.update({'individual/'+name+'/'+key:value for name,record in individual_summary.items()
                        for key,value in record.items() if isinstance(value,(int,float))})
    run.log({'checkpoint_comparison': wandb.Table(columns=['strategy','steps','dev_PER','test_PER','text_free_F1_20ms','K2_FA_F1_20ms'],data=rows)})
    artifact = wandb.Artifact(arm+'-checkpoint-comparison'+('-validation' if args.limit else ''), type='evaluation')
    artifact.add_file(str(output/'comparison.json')); artifact.add_file(str(output/'selection.json'))
    run.log_artifact(artifact)
    write(identity_path, {'id':run.id,'url':run.url,'mode':run.settings.mode}); run.finish()
    write(output/'COMPARE_COMPLETE.json', {'objective':arm,'limit':args.limit,'selection':selected})
    import subprocess
    subprocess.run([sys.executable,str(REPRO/'summarize_checkpoint_comparison.py')],check=True)
    print('COMPLETE',output,flush=True)


if __name__ == '__main__':
    main()
