"""One-seed frame supervision report, separate from OT-only three-seed results."""
import json
from reproduction.buckeye_frame_ag_20261027 import common
from reproduction.timit_frame_ce_20261026.report import metrics


def main():
    output=common.HERE/'models'/common.NAME
    done=(output/'BENCHMARK_COMPLETE.json').exists()
    lines=['# Buckeye full40：单组 frame-wise AG（seed 42）','',
        '使用 gold 时间标注训练，监督强于 OT-only；不能将差异归因于散度本身。',
        '仅一个 seed，不报告三种子均值或标准差。主比较为 single best-dev；final5/final10 另列。',
        '所有模型及平均权重 FP32，encoder BF16；评估 FP32，FA 为无剪枝精确 CTC Viterbi。',
        'Buckeye dev PER 选模；Buckeye test 和 TIMIT 均不参与选择；评估仅本地。','',
        f'训练完成：{(output/"TRAIN_COMPLETE.json").exists()}；全部评估完成：{done}。','']
    results=[]
    if done:
        summary=json.loads((output/'FINAL_SUMMARY.json').read_text())
        lines += [f'开发集选择的候选：`{summary["selected"]}`。','',
            '| 候选 | 集合 | PER % | PFER % | Text-free F1 % | Text-free R-value % | FA F1 % | FA R-value % | FA MAE ms |',
            '|---|---|---:|---:|---:|---:|---:|---:|---:|']
        for candidate,splits in summary['candidates'].items():
            for split,result in splits.items():
                values=metrics(result)
                results.append(dict(candidate=candidate,split=split,**values))
                lines.append('| '+' | '.join([candidate,split]+[f'{values[k]:.2f}' for k in
                    ['PER','PFER','text_free_F1','text_free_R','FA_F1','FA_R','FA_MAE_ms']])+' |')
    else:
        lines += ['尚未完成，不能将提交、profile 或训练完成当作最终结果。']
    lines += ['', '边界评分：20ms 容差、唯一端点、排除句首尾20ms、一对一匹配、冻结 IF-MDD R-value。',
        'FA 以给定音素序列为条件；模型没有 blank 监督。gold 时间仅用于训练标签、loss 诊断和评分，不用于对齐搜索。','']
    common.write(common.HERE/'RESULTS.json',dict(complete=done,seed=42,results=results,evaluation_uploads=False))
    (common.HERE/'REPORT_ZH.md').write_text('\n'.join(lines))
    print('REPORT',done,flush=True)


if __name__=='__main__':
    main()
