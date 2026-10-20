"""Highlight upstream OTTC jsd results; reuse completed CE for comparison."""
import json
from datetime import datetime
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from segmentation_metrics import boundary_r_value,flat_boundary_r_value

from track import live_progress

SUITE=Path(__file__).resolve().parent
REPRO=SUITE.parent


def read(path):
    return json.loads(path.read_text()) if path.exists() else {}


def fmt(value):
    return '—' if value is None else f'{value:.2f}'


def main():
    lines=['# OTTC 上游 JSD：重点结果','',
           f'更新：{datetime.now().astimezone().isoformat(timespec="seconds")}。','',
           '重点查看 OTTC 上游 jsd 的识别、text-free segmentation 和 IF-MDD K2-FSA forced alignment。CE 作为已有对照，同数据与模型配置已训练完成的 CE 不重训。这里保留用户指定的上游 jsd 实现，区别于数学标准 Jensen–Shannon divergence。','',
           '| 数据 / 四头 JSD | 状态 | 最近 train loss | Dev PER % | Dev F1@20ms % | Dev R-value % | W&B |','|---|---|---:|---:|---:|---:|---|']
    for dataset in ['timit','libri100','libri960']:
        directory=REPRO/('multihead_'+dataset)/'models/jsd_upstream'
        complete=read(directory/'TRAIN_COMPLETE.json');live=live_progress(directory)
        train=live.get('last_train',{});dev=live.get('last_dev',{})
        status='✅ 完成40轮' if complete else f'🟡 训练中 {train["epoch"]:.2f}/40' if train else '⏳ 尚未启动'
        identity=read(directory/'wandb_run.json');url=identity.get('url')
        lines.append('| '+' | '.join([dataset,status,fmt(train.get('loss')),fmt(dev.get('eval_PER_percent/ottc_arpabet')),
                       fmt(dev.get('eval_boundary_F1_20ms/ottc_arpabet')),fmt(flat_boundary_r_value(dev,'eval_','ottc_arpabet')),'[W&B]('+url+')' if url else '—'])+' |')
    lines+=['','dev 表使用原始 OTTC ARPAbet head；不同阶段的 dev 不当作最终比较。','',
            '## LibriSpeech 最终 text-free test：JSD 与已有对照','',
            '| 数据 | 目标 | Test PER % ↓ | Text-free F1@20ms % ↑ | Text-free R-value % ↑ | K2 FA |',
            '|---|---|---:|---:|---:|---|']
    for dataset in ['libri100','libri960']:
        for objective,label in [('jsd_upstream','JSD'),('ce','CE（复用）'),('ctc','CTC')]:
            directory=REPRO/('multihead_'+dataset)/'models'/objective
            if not (directory/'TRAIN_COMPLETE.json').exists():continue
            metrics=read(directory/'test_metrics.json')
            if not metrics:continue
            k2=REPRO/('multihead_'+dataset)/'evaluation_modes'/objective/'EVAL_COMPLETE.json'
            lines.append('| '+' | '.join([dataset,label,fmt(metrics.get('test_PER_percent/ottc_arpabet')),
                         fmt(metrics.get('test_boundary_F1_20ms/ottc_arpabet')),fmt(flat_boundary_r_value(metrics,'test_','ottc_arpabet')),'已完成' if k2.exists() else '待测'])+' |')
    lines+=['','表内为原始 OTTC ARPAbet head；LibriSpeech 分段参考来自自动对齐。最终 epoch40 模型与 dev 选择/权重平均结果分开报告。','',
            '## 三个 checkpoint：JSD 与已有 CE 对照','',
            '| 数据 | 目标 / 策略 | step | Test PER % ↓ | Text-free F1@20ms % ↑ | Text-free R-value % ↑ | K2 FA F1@20ms % ↑ | K2 FA R-value % ↑ |',
            '|---|---|---|---:|---:|---:|---:|---:|']
    checkpoint_links=[]
    for dataset in ['timit','libri100','libri960']:
        for objective,label in [('jsd_upstream','JSD'),('ce','CE（复用）')]:
            directory=REPRO/'checkpoint_comparison'/(dataset+'_'+objective)/'results'
            if not (directory/'COMPARE_COMPLETE.json').exists():continue
            result=read(directory/'comparison.json');best=result['selection']['best_single']
            values=[('dev最好单个',str(best),result['individual_tests'][str(best)])]
            for strategy,key in [('最后3个权重平均','last3'),('dev最好3个权重平均','best3')]:
                values.append((strategy,','.join(map(str,result['selection'][key])),result['weight_averages'][key+'_weight_average']))
            for strategy,steps,metrics in values:
                modes=metrics['splits']['test'];tf=modes['text_free']['ottc_arpabet'];fa=modes['forced_k2']['ottc_arpabet']
                lines.append('| '+' | '.join([dataset,label+' / '+strategy,steps,fmt(tf['PER_percent']),
                             fmt(tf['boundaries']['20']['f1_percent']),fmt(boundary_r_value(tf['boundaries']['20'])),fmt(fa['boundaries']['20']['f1_percent']),fmt(boundary_r_value(fa['boundaries']['20']))])+' |')
            identity=read(directory/'wandb_run.json')
            if identity.get('url'):checkpoint_links.append('['+dataset+' '+label+' checkpoint 比较 W&B]('+identity['url']+')')
    if checkpoint_links:lines+=['',' · '.join(checkpoint_links),'']
    lines+=['','TIMIT 仅保留最后三个 checkpoint，因此最好三个与最后三个组成相同集合，权重平均结果相同。选最好单个和最好三个仅使用 dev PER；上表每个模型在完整 test 上分别评分。','',
            '## 原始单头 JSD（已完成）','',
            '| 目标 | README test240 PER % | Text-free F1@20ms % | Text-free R-value % |','|---|---:|---:|---:|']
    for name,label in [('runexp_ottc_fuse_phoneme_jsd','上游 JSD'),('runexp_ottc_fuse_phoneme','CE（复用）')]:
        record=read(REPRO/'results'/name/'metrics.json')
        metrics=record.get('splits',{}).get('readme_kaldi_test_240')
        if metrics:lines.append(f'| {label} | {fmt(100*metrics["raw_PER"])} | {fmt(100*metrics["boundaries"]["20"]["f1"])} | {fmt(boundary_r_value(metrics["boundaries"]["20"]))} |')
    lines+=['','单头与四头数据范围及训练配置不同，分别报告。MFA 标签 Zapformer 是另一实验系列，不列作本次四头 JSD 的结果。','',
            '[Zapformer JSD 支持与验证](mfa_jsd/README.md)：复用原 OTTC JSD 归档实现，完整模型 GPU 前向/反向与 loss/梯度一致性检查已通过；新 JSD 长训尚未提交。','',
            '[论文 MFA 3.0 与归档 CTM 身份核对](MFA_TOPLINE.md)：论文 test TSE=31.95ms、R-value=87.44%；原生 MFA topline 的旧标注已撤回，归档 CTM 身份仍待核实。','',
            '[CE/JSD/CTC 与归档 CTM 的统一重计分](mfa_protocol_comparison/README.md)：冻结 scorer、raw PHN、同一 test192 和各系统共同等长 TSE 子集；复用既有 K2 输出。这不是论文 MFA 3.0 的同口径复现。','',
            '[全部四头与 checkpoint 结果](../checkpoint_comparison/README.md) · [实时 dev 曲线](https://wandb.ai/kevingenghaopeng-/ottc-english-multiphoneset/reports/English-phonemes:-dev-PER-and-segmentation--VmlldzoxODA3OTU2MQ==) · [实验状态](EXPERIMENT_STATUS.md)。','']
    (SUITE/'JSD_RESULTS.md').write_text('\n'.join(lines))


if __name__=='__main__':main()
