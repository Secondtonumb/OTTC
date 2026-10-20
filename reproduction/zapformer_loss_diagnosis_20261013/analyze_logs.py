"""Backfill derived summaries without editing original histories or timestamps."""
import collections
import hashlib
import json
import math
from pathlib import Path
import statistics

HERE=Path(__file__).resolve().parent


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    inputs=json.loads((HERE/'inputs.json').read_text())
    report=dict(backfill=True,original_steps_retained=True,interpolation=False,
                source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),arms={})
    fig,axs=plt.subplots(2,2,figsize=(13,9))
    for arm,info in inputs.items():
        out=Path(info['output'])
        files={n:out/n for n in ['updates.jsonl','evaluation_loss.jsonl','timit_metrics.jsonl']}
        loaded={n:[json.loads(l) for l in f.read_text().splitlines()] for n,f in files.items()}
        rows=loaded['updates.jsonl'];epochs=collections.defaultdict(list)
        for r in rows:epochs[r['epoch']].append(r)
        summaries=[]
        for e,records in sorted(epochs.items()):
            utt=sum(r['utterances'] for r in records)
            summary=dict(epoch=e,first_step=records[0]['optimizer_step'],last_step=records[-1]['optimizer_step'],
                         updates=len(records),complete_epoch=len(records)==106,
                         total_loss_per_utterance=sum(r['loss'] for r in records)/utt,
                         ottc_per_utterance=sum(r.get('ottc_sum',r.get('jsd_sum',0)) for r in records)/utt,
                         forward_sum_local_mean_averaged_over_ranks_and_updates=(sum(r.get('forward_sum_raw',0) for r in records)/8/len(records)) if arm!='origin' else None,
                         lr_first=records[0]['lr'],lr_last=records[-1]['lr'])
            summaries.append(summary)
        dev=loaded['evaluation_loss.jsonl']
        devsummaries=[dict(step=r['optimizer_step'],epoch=r['epoch'],
                          ottc_per_utterance=r.get('ottc_sum',r.get('jsd_sum',0))/r['utterances'],
                          forward_sum_per_utterance=r.get('forward_sum_raw',0)/r['utterances']) for r in dev]
        metrics=[r for r in loaded['timit_metrics.jsonl'] if r['split']=='development' and 'checkpoint' not in r]
        metricsummary=[dict(step=r['optimizer_step'],epoch=r['epoch'],PER_percent=r['mapped_arpabet']['PER_percent'],
                            insertions_percent=r['mapped_arpabet']['insertions_percent'],deletions_percent=r['mapped_arpabet']['deletions_percent'],
                            forced_boundary_F1_20ms=r['forced_k2']['boundary_F1_20ms_percent'],
                            forced_R_value_20ms=r['forced_k2']['boundary_R_value_20ms_percent']) for r in metrics]
        complete=[r for r in summaries if r['complete_epoch']]
        trend=dict(first=complete[0],last=complete[-1],ottc_relative_drop_percent=100*(1-complete[-1]['ottc_per_utterance']/complete[0]['ottc_per_utterance']))
        report['arms'][arm]=dict(input_log_sha256={n:hashlib.sha256(f.read_bytes()).hexdigest() for n,f in files.items()},
                                completed_epoch_trend=trend,epoch_summaries=summaries,development_loss=devsummaries,
                                development_metrics=metricsummary,
                                raw_utterance_range=[min(r['utterances'] for r in rows),max(r['utterances'] for r in rows)])
        if arm=='origin':continue
        axs[0,0].plot([r['optimizer_step'] for r in rows],[r['loss'] for r in rows],lw=.5,alpha=.7,label=arm)
        axs[0,1].plot([r['epoch'] for r in complete],[r['ottc_per_utterance'] for r in complete],marker='o',label=arm)
        axs[1,0].plot([r['step'] for r in devsummaries],[r['ottc_per_utterance'] for r in devsummaries],marker='.',label=arm)
        axs[1,1].plot([r['step'] for r in metricsummary],[r['PER_percent'] for r in metricsummary],marker='.',label=arm)
    titles=['Raw global batch-sum loss (variable batch size; mixed reductions)',
            'Completed-epoch OTTC sum / utterances (CE and JSD differ)',
            'Fixed TIMIT dev400 OTTC / utterances (CE and JSD differ)',
            'TIMIT dev400 mapped phone error rate']
    xlabels=['Original optimizer step','Original epoch','Original optimizer step','Original optimizer step']
    for ax,title,x in zip(axs.flat,titles,xlabels):
        ax.set_title(title,fontsize=10);ax.set_xlabel(x);ax.grid(alpha=.2);ax.legend(fontsize=8)
    axs[1,1].set_ylabel('PER (%)')
    fig.tight_layout();fig.savefig(HERE/'loss_diagnosis.png',dpi=150);plt.close(fig)
    (HERE/'log_report.json').write_text(json.dumps(report,indent=2)+'\n')
    for a,x in report['arms'].items():
        t=x['completed_epoch_trend']
        print(a,'completed epochs',t['first']['epoch'],t['last']['epoch'],'OTTC/utt',round(t['first']['ottc_per_utterance'],4),round(t['last']['ottc_per_utterance'],4),'drop%',round(t['ottc_relative_drop_percent'],2))


if __name__=='__main__':main()
