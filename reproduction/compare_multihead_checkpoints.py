"""Compare saved Wav2Vec2 multihead checkpoints without changing training artifacts."""
import argparse
import gc
import json
import os
import statistics
import subprocess
import sys
from pathlib import Path

import torch
from average_checkpoints import average, sha256

REPRO = Path(__file__).resolve().parent


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.partial')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')
    temporary.replace(path)


def main():
    from reproduction.wandb_policy import init_evaluation
    parser=argparse.ArgumentParser()
    parser.add_argument('--plan',type=Path,required=True)
    parser.add_argument('--limit',type=int,default=0)
    args=parser.parse_args();torch.set_num_threads(4)
    plan=json.loads(args.plan.read_text());count=plan['count'];objective=plan['objective']
    output=Path(plan['output'])/('validation' if args.limit else 'results')
    output.mkdir(parents=True,exist_ok=True)
    if (output/'COMPARE_COMPLETE.json').exists():
        print('Already complete',output,flush=True);return
    candidates=plan['candidates'];scores={};dev_results={}
    if len(candidates)<count:raise ValueError('Not enough saved checkpoints')
    provenance={'plan_sha256':sha256(args.plan),'evaluator_sha256':sha256(Path(__file__)),
                'metric_source_sha256':sha256(REPRO/'evaluate_multihead_modes.py'),
                'limit':args.limit,'selection_metric':'dev PER_percent/ottc_arpabet ascending; tie -> newer step',
                'test_used_for_selection':False,'candidate_scope':'saved weights only; deleted historical weights unavailable',
                'intermediate_diagnostic':plan['intermediate_diagnostic']}

    def run_eval(checkpoint, destination, splits, text_only=False, label=None):
        actual=destination/'validation' if args.limit else destination
        if (actual/'EVAL_COMPLETE.json').exists():
            result=json.loads((actual/'metrics.json').read_text())
            assert result['checkpoint_sha256']==sha256(checkpoint/'pytorch_model.bin')
            return result
        command=[sys.executable,'-u',str(REPRO/'evaluate_multihead_modes.py'),
                 '--objective',objective,'--experiment',plan['experiment'],
                 '--checkpoint',str(checkpoint),'--output',str(destination),
                 '--splits',*splits,'--no-wandb','--no-native-ot']
        if text_only:command.append('--text-free-only')
        if args.limit:command+=['--limit',str(args.limit)]
        if label:command+=['--comparison-label',label]
        subprocess.run(command,check=True)
        return json.loads((actual/'metrics.json').read_text())

    # Evaluate dev for every retained candidate before selecting or opening test.
    for candidate in candidates:
        step=candidate['step'];checkpoint=Path(candidate['checkpoint'])
        result=run_eval(checkpoint,output/'individual'/str(step)/'dev',['dev'],True)
        scores[step]=result['splits']['dev']['text_free']['ottc_arpabet']['PER_percent']
        dev_results[str(step)]=result
    last=sorted(candidates,key=lambda c:c['step'])[-count:]
    best=sorted(candidates,key=lambda c:(scores[c['step']],-c['step']))[:count]
    selection={'last'+str(count):[c['step'] for c in last],
               'best'+str(count):[c['step'] for c in best],
               'best_single':best[0]['step'],'dev_scores':scores,**provenance}
    write(output/'selection.json',selection)
    tests={}
    for candidate in {c['step']:c for c in last+best}.values():
        step=candidate['step']
        tests[str(step)]=run_eval(Path(candidate['checkpoint']),output/'individual'/str(step)/'test',['test'])
    averages={};aliases={};seen={}
    for name,chosen in [('last'+str(count)+'_weight_average',last),('best'+str(count)+'_weight_average',best)]:
        chosen=sorted(chosen,key=lambda c:c['step']);steps=tuple(c['step'] for c in chosen)
        if steps in seen:
            aliases[name]=seen[steps];averages[name]=averages[seen[steps]];continue
        seen[steps]=name;artifact=output/name/'model';artifact.mkdir(parents=True,exist_ok=True)
        weights=artifact/'pytorch_model.bin'
        if not weights.exists():
            configs=[json.loads((Path(c['checkpoint'])/'config.json').read_text()) for c in chosen]
            fields=['encoder_config','vocabs','objective','freeze_alignment']
            for config in configs[1:]:
                if any(config.get(k)!=configs[0].get(k) for k in fields):raise ValueError('Model configuration differs')
            state,policy=average([Path(c['checkpoint'])/'pytorch_model.bin' for c in chosen])
            torch.save(state,weights);del state;gc.collect()
            write(artifact/'config.json',configs[-1])
            write(artifact/'AVERAGED.json',{'steps':steps,'policy':policy,
                  'source_sha256':{str(c['step']):sha256(Path(c['checkpoint'])/'pytorch_model.bin') for c in chosen},**provenance})
        averages[name]=run_eval(artifact,output/name/'evaluation',['dev','test'],label=objective+'-'+name)
    metric_summaries={}
    for name,chosen in [('last'+str(count)+'_individual',last),('best'+str(count)+'_individual',best)]:
        per=[tests[str(c['step'])]['splits']['test']['text_free']['ottc_arpabet']['PER_percent'] for c in chosen]
        metric_summaries[name]={'steps':[c['step'] for c in chosen],
                               'test_PER_mean_percent':statistics.mean(per),
                               'test_PER_population_std_percent':statistics.pstdev(per)}
    result={'dataset':plan['dataset'],'objective':objective,'selection':selection,
            'individual_dev':dev_results,'individual_tests':tests,'weight_averages':averages,
            'identical_average_aliases':aliases,'individual_metric_summaries':metric_summaries,**provenance}
    write(output/'comparison.json',result)
    import wandb
    options=dict(project='ottc-english-multiphoneset',entity='kevingenghaopeng-',
                 name=plan['dataset']+'-'+objective+'-last'+str(count)+'-best'+str(count)+('-validation' if args.limit else ''),
                 group=plan['dataset']+'_checkpoint_comparison',job_type='evaluation',dir=str(output),
                 config={**plan,**provenance},settings=wandb.Settings(init_timeout=45))
    try:run=init_evaluation(**options)
    except wandb.errors.CommError:options['mode']='offline';run=init_evaluation(**options)
    rows=[]
    for step,value in tests.items():
        for head,metrics in value['splits']['test']['text_free'].items():
            fa=value['splits']['test']['forced_k2'][head]
            rows.append(['individual',step,head,metrics.get('PER_percent',metrics.get('token_error_percent')),
                         metrics.get('boundaries',{}).get('20',{}).get('f1_percent'),
                         fa.get('boundaries',{}).get('20',{}).get('f1_percent')])
    for name,value in averages.items():
        for head,metrics in value['splits']['test']['text_free'].items():
            fa=value['splits']['test']['forced_k2'][head]
            rows.append([name,','.join(map(str,selection['last'+str(count) if name.startswith('last') else 'best'+str(count)])),
                         head,metrics.get('PER_percent',metrics.get('token_error_percent')),
                         metrics.get('boundaries',{}).get('20',{}).get('f1_percent'),
                         fa.get('boundaries',{}).get('20',{}).get('f1_percent')])
            run.summary.update({name+'/'+head+'/'+key:val for key,val in metrics.items() if isinstance(val,(int,float))})
    run.log({'comparison':wandb.Table(columns=['strategy','steps','head','test_PER_or_token_error','text_free_F1_20ms','K2_FA_F1_20ms'],data=rows)})
    run.summary.update({'individual/'+name+'/'+key:value for name,record in metric_summaries.items()
                        for key,value in record.items() if isinstance(value,(int,float))})
    artifact=wandb.Artifact(plan['dataset']+'-'+objective+'-checkpoint-comparison'+('-validation' if args.limit else ''),type='evaluation')
    artifact.add_file(str(output/'comparison.json'));artifact.add_file(str(output/'selection.json'));run.log_artifact(artifact)
    write(output/'wandb_run.json',{'id':run.id,'url':run.url,'mode':run.settings.mode});run.finish()
    write(output/'COMPARE_COMPLETE.json',{'selection':selection,'limit':args.limit})
    subprocess.run([sys.executable,str(REPRO/'summarize_checkpoint_comparison.py')],check=True)


if __name__=='__main__':main()
