"""Audit recorded CIF losses; no inference or training and no interpolation."""
from collections import defaultdict
import csv
import datetime
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent


def main():
    runs = json.loads((ROOT/'reproduction/cif_timit_20261016/active_runs.json').read_text())
    report = {'created_at': datetime.datetime.now().astimezone().isoformat(),
        'origin': 'derived from original recorded epochs/steps; no missing points interpolated',
        'train_total': 'mean of recorded per-batch objectives within epoch',
        'components': 'pooled sums / phones for CE, quantity and CTC; native OTTC sums / utterances',
        'dev_total': 'pooled CE/phones + quantity_weight*L1/phones + ctc_weight*CTC/phones + ottc_weight*OTTC/utterances',
        'caution': 'Total objectives differ across variants. Train uses dropout; dev is teacher-forced with oracle CIF count for loss only. Free segmentation uses neither labels nor counts.',
        'runs': {}, 'historical': {}}
    for name, directory in runs.items():
        p=Path(directory); config=json.loads((p/'run_config.json').read_text())
        rows=[json.loads(line) for line in (p/'history.jsonl').read_text().splitlines()]
        train={r['step']:r for r in rows if 'train_loss' in r}
        dev={r['epoch']:r for r in rows if 'dev/ce_per_phone' in r}
        assert set(train)==set(range(1,2921)) and set(dev)==set(range(1,41))
        assert all(math.isfinite(r['train_loss']) and math.isfinite(r['gradient_norm']) for r in train.values())
        by_epoch=defaultdict(list)
        for step,row in sorted(train.items()):
            by_epoch[(step-1)//73+1].append(row)
        epochs=[]
        for epoch,rr in sorted(by_epoch.items()):
            assert len(rr)==73 and sum(r['utterances'] for r in rr)==4620
            phones=sum(r['train_target_phones'] for r in rr)
            record={'epoch':epoch,'step':epoch*73,
                'train_total':sum(r['train_loss'] for r in rr)/len(rr)}
            for comp in ['ce','quantity','ctc','ottc']:
                denominator=4620 if comp=='ottc' else phones
                record['train_'+comp]=sum(r['train_'+comp+'_sum'] for r in rr)/denominator
                record['dev_'+comp]=dev[epoch]['dev/'+comp+('_per_utterance' if comp=='ottc' else '_per_phone')]
            record['dev_total']=record['dev_ce']+config['quantity_weight']*record['dev_quantity']+config['ctc_weight']*record['dev_ctc']+config['ottc_weight']*record['dev_ottc']
            record['dev_quantity_MAE']=dev[epoch]['dev/quantity_MAE']
            record['dev_R20']=dev[epoch]['dev/text_free/r_value_pct_20ms']
            epochs.append(record)
        best=min(epochs,key=lambda x:x['dev_total'])
        best_r=max(epochs,key=lambda x:x['dev_R20'])
        a,b=epochs[29],epochs[39]
        delta={k:100*(b[k]/a[k]-1) for k in ['train_total','dev_total','train_ce','dev_ce']}
        report['runs'][name]={'source_history':str(p/'history.jsonl'),
            'history_sha256':hashlib.sha256((p/'history.jsonl').read_bytes()).hexdigest(),
            'weights':{k:config[k] for k in ['quantity_weight','ctc_weight','ottc_weight','label_smoothing']},
            'complete_training_updates':len(train),'dev_epochs':len(dev),'nonfinite_loss_or_norm':0,
            'epoch30_to40_relative_change_pct':delta,
            'best_dev_total_epoch':best['epoch'],'best_dev_total':best['dev_total'],
            'best_dev_R20_epoch':best_r['epoch'],'best_dev_R20':best_r['dev_R20'],'epochs':epochs}
        with (HERE/(name+'_epochs.csv')).open('w') as f:
            writer=csv.DictWriter(f,fieldnames=list(epochs[0]));writer.writeheader();writer.writerows(epochs)
    # Historical Trainer validation used kaldi_test240, not the CIF dev400.
    for name in ['runexp_ctc_fuse_phoneme','runexp_ottc_fuse_phoneme','runexp_ottc_fuse_phoneme_jsd']:
        p=ROOT/'reproduction/models/large_models_results/wav2vec2-large-lv60'/name/'checkpoint-2920/trainer_state.json'
        state=json.loads(p.read_text());tr=[r for r in state['log_history'] if 'loss' in r];dv=[r for r in state['log_history'] if 'eval_loss' in r]
        report['historical'][name]={'source':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),
            'recording':'original intervals; only14 train-loss records, 40 validation records, no interpolation',
            'eval_set':'historical kaldi_test240; not CIF dev400; values not directly comparable',
            'last_training_records':tr[-4:],'last_validation_records':dv[-4:]}
    (HERE/'metrics.json').write_text(json.dumps(report,indent=2)+'\n')
    lines=['# CIF loss 收敛审计','','2920个训练更新、40轮验证均完整，三组无非有限loss或梯度范数。总loss的定义不同，不跨组比较绝对大小。', '',
        '|实验|train epoch30→40|相对变化|dev epoch30→40|相对变化|最低dev loss epoch|',
        '|---|---:|---:|---:|---:|---:|']
    for name,x in report['runs'].items():
        a,b=x['epochs'][29],x['epochs'][39];d=x['epoch30_to40_relative_change_pct']
        lines.append(f"|{name}|{a['train_total']:.4f}→{b['train_total']:.4f}|{d['train_total']:+.2f}%|{a['dev_total']:.4f}→{b['dev_total']:.4f}|{d['dev_total']:+.2f}%|{x['best_dev_total_epoch']}|")
    lines += ['', '判断：训练稳定并已进入后期缓慢下降阶段；不能宣称所有训练项已完全收敛。验证已大体平台，CTC组后期回升，存在过拟合迹象。OTTC验证总loss第40轮仍为最低，但30→40轮只改善约0.2%，同时训练OTTC项仍下降。', '',
        'OTTC辅助项 train/utterance：epoch30为1.8279、epoch40为1.6499；dev为2.5618→2.5582。CTC辅助项 train/phone：0.1504→0.1175；dev为0.3295→0.3656。', '',
        'quantity的dev每句绝对计数误差在epoch40约0.97/0.97/1.02个phone（无辅助/CTC/OTTC）。quantity稳定不保证每个边界准确。CE使用0.1 label smoothing，其目标最小值不是0。', '',
        '分段R-value最优发生于epoch10/24/11；dev loss最优发生于22/21/40。训练目标与分段指标的最优点不同，不能把较低loss等同于更好的分段。', '',
        '历史CTC/CE/upstream-JSD的原始记录已一并保存到metrics.json。它们最后数轮eval_loss也趋于平台；但历史验证为kaldi_test240、loss定义和reduction也不同，不同CIF dev400做绝对值比较。', '',
        '复现：`python -m reproduction.cif_loss_audit_20261016.analyze`。逐epoch CSV为原始记录的明确聚合，非平滑或插值；原始训练、checkpoint和W&B运行均未改变。']
    (HERE/'REPORT_ZH.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines[:10]))
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print('Matplotlib unavailable; JSON/CSV/report saved.')
        return
    titles={'no_ctc':'CIF: CE + quantity','ctc_0p5':'CIF + 0.5 CTC','ottc_jsd_0p5':'CIF + 0.5 upstream OTTC'}
    fig,axes=plt.subplots(1,3,figsize=(12,3.8),constrained_layout=True)
    for ax,(name,x) in zip(axes,report['runs'].items()):
        rr=x['epochs'];e=[r['epoch'] for r in rr]
        ax.plot(e,[r['train_total'] for r in rr],label='Train: mean batch objective',lw=1.6)
        ax.plot(e,[r['dev_total'] for r in rr],label='Dev: pooled objective',lw=1.6)
        ax.axvspan(30,40,color='gray',alpha=.12)
        ax.set(title=titles[name],xlabel='Original epoch',ylabel='Own objective (log scale)',yscale='log')
        ax.grid(alpha=.2);ax.legend(fontsize=7)
    fig.suptitle('Recorded CIF loss curves; objectives differ across columns')
    fig.savefig(HERE/'loss_curves.png',dpi=180);fig.savefig(HERE/'loss_curves.pdf');plt.close(fig)
    fig,axes=plt.subplots(3,3,figsize=(12,9),constrained_layout=True)
    for col,(name,x) in enumerate(report['runs'].items()):
        for row,comp in enumerate(['ce','quantity','ctc' if name=='ctc_0p5' else 'ottc']):
            ax=axes[row,col];rr=x['epochs'];e=[r['epoch'] for r in rr]
            if row==2 and name=='no_ctc':ax.text(.5,.5,'No auxiliary loss',ha='center',va='center',transform=ax.transAxes);ax.set_axis_off();continue
            ax.plot(e,[r['train_'+comp] for r in rr],label='Train')
            ax.plot(e,[r['dev_'+comp] for r in rr],label='Dev')
            ax.axvspan(30,40,color='gray',alpha=.12);ax.grid(alpha=.2);ax.legend(fontsize=8)
            ax.set(title=titles[name] if row==0 else '',xlabel='Original epoch',ylabel=comp.upper()+('/utterance' if comp=='ottc' else '/phone'))
    fig.suptitle('Unweighted component losses, pooled with matching denominators')
    fig.savefig(HERE/'loss_components.png',dpi=180);fig.savefig(HERE/'loss_components.pdf');plt.close(fig)


if __name__=='__main__':
    main()
