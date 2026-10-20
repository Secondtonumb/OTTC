"""Report independent FP32 K2 metrics, matched clocks, and separate FA."""
import ast
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
EXPERIMENT=HERE.parent/'native_stride_ottc_20261018'
sys.path.insert(0,str(ROOT/'reproduction'))
from segmentation_metrics import r_value_percent,SOURCE as R_SOURCE


def boundary_helpers():
    namespace=dict(Counter=Counter,r_value_percent=r_value_percent)
    for path,names in [(ROOT/'reproduction/libri100_singlehead_20261014/train.py',{'boundary_matches'}),
                       (ROOT/'reproduction/libri960_continue5_20261017/evaluate.py',{'edges','new_counts','score','summary'})]:
        nodes=[n for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef) and n.name in names]
        exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),namespace)
    return namespace


def native_clock(path,helpers):
    counts=helpers['new_counts']()
    for line in path.open():
        row=json.loads(line);decoded=row['fsa_text_free'];frames=len(decoded['frame_token_ids']);duration=row['duration']
        starts=[round(start*frames/duration) for _,start,_ in decoded['predicted_spans']]
        helpers['score'](counts,row['reference'],[start*.01 for start in starts[1:]],duration)
        counts['utterances']+=1
    return helpers['summary'](counts)


def main():
    helpers=boundary_helpers();results=[]
    for head in ['mono','s3t2ss']:
        for cost in ['jsd_upstream','ce','jsd_standard']:
            name=f'wav2vec2__timit__10ms__{head}__{cost}';out=HERE/'results'/name
            if not (out/'CORRECTED_EVALUATION_COMPLETE.json').exists():continue
            selection=json.loads((out/'DEV_SELECTION.json').read_text());chosen=selection['selected']
            dev=json.loads((out/f'benchmark_{chosen}_dev.json').read_text())
            test=json.loads((out/f'benchmark_{chosen}_test.json').read_text())
            mode='fsa_text_free' if head=='s3t2ss' else 'text_free'
            common=native_clock(out/f'benchmark_{chosen}_test.predictions.jsonl',helpers) if head=='s3t2ss' else test['modes'][mode]
            fa=test['modes']['forced_alignment']
            assert fa['alignment_failures']==0
            results.append(dict(recipe=name,head=head,cost=cost,selected=chosen,
                dev_PER=dev['modes'][mode]['nosil_PER_percent'],test_PER=test['modes'][mode]['nosil_PER_percent'],
                original_boundary_clock_metrics=test['modes'][mode]['boundaries'],
                common_native_10ms_metrics=common['boundaries'],forced_alignment=fa,
                dev_selection=selection,predictions=str(out/f'benchmark_{chosen}_test.predictions.jsonl')))
    complete=sum(row['head']=='s3t2ss' for row in results)==3
    report=dict(updated_at_UTC=datetime.now(timezone.utc).isoformat(),topology_metrics_complete=complete,
        precision='Fresh standalone FP32 batch1, no retained Trainer AMP wrapper',
        topology='Original IF-MDD S3T2** token FST, K2 dense intersection/shortest path, output_beam8',
        inventory='119 states: blank1, sil1, ordinary phones including q each3; q/sil removed only for PER',
        selection='Dev-only K2 no-sil PER among fixed best_dev/final/final5/final10 candidates; not all epoch checkpoints',
        boundary_protocol='unique endpoints; exclude utterance edges20ms; one-to-one matching10/20/30ms',
        boundary_clock='Common native10ms for direct text-free comparison; original duration/frame_count K2 scores also preserved',
        R_value_source=str(R_SOURCE),text_free_vs_FA='Text-free graph uses no supplied transcript; FA uses supplied reference phones, never reference timestamps',
        training_restarted=False,rows=results)
    (HERE/'RESULTS.json').write_text(json.dumps(report,indent=2)+'\n')
    lines=['# TIMIT 原生10ms：K2 S3T2** 重新解码',f'\n更新 UTC：{report["updated_at_UTC"]}。dev400/core-test192。原权重不变，无新训练。',
        '\n本次独立FP32模型解码，不沿用Trainer训练时的BF16 autocast包装。旧评测的“FP32”标签不准确；旧数值保留在原目录，本报告采用新结果。',
        '\n解码直接复用IF-MDD原始token FST：S0与S2必经、S1可跳过、状态允许self-loop，sil单状态，保留重复音素；K2 output_beam=8，无phone LM、插入惩罚或给定转录。',
        '\n在既定best-dev/final/final5/final10四个候选中，仅用dev K2 PER选择。训练期best-dev候选本身按greedy选出；本报告不将其称为所有epoch中的K2最优。',
        '\n无文本core-test指标（PER及P/R/F1/R-value均为%；边界统一固定10ms时间轴、20ms容差）：',
        '\n| 输出/解码 | OTTC代价 | dev所选权重 | dev PER | test PER | P | R | F1 | R-value |',
        '|---|---|---|---:|---:|---:|---:|---:|---:|']
    for row in results:
        b=row['common_native_10ms_metrics']['20'];decoder='S3T2** K2' if row['head']=='s3t2ss' else '单音素 greedy'
        lines.append(f'| {decoder} | {row["cost"]} | {row["selected"]} | {row["dev_PER"]:.2f} | {row["test_PER"]:.2f} | {b["precision_percent"]:.2f} | {b["recall_percent"]:.2f} | {b["f1_percent"]:.2f} | {b["r_value_percent"]:.2f} |')
    lines+=['\nS3T2** 原 duration/frame_count 时间轴的无文本分割分数（与原解码导出兼容，不能与固定10ms时间轴混用）：',
        '\n| 代价 | F1@20 | R-value@20 |','|---|---:|---:|']
    for row in results:
        if row['head']=='s3t2ss':
            b=row['original_boundary_clock_metrics']['20'];lines.append(f'| {row["cost"]} | {b["f1_percent"]:.2f} | {b["r_value_percent"]:.2f} |')
    lines+=['\n给定音素FA单独报告（非无文本分割；时间轴duration/frame_count）：',
        '\n| 输出 | 代价 | P | R | F1 | R-value | MAE ms | 失败/core192 |','|---|---|---:|---:|---:|---:|---:|---:|']
    for row in results:
        fa=row['forced_alignment'];b=fa['boundaries']['20']
        lines.append(f'| {row["head"]} | {row["cost"]} | {b["precision_percent"]:.2f} | {b["recall_percent"]:.2f} | {b["f1_percent"]:.2f} | {b["r_value_percent"]:.2f} | {fa["boundary_MAE_ms"]:.2f} | {fa["alignment_failures"]}/192 |')
    lines+=['\n验证：SAVED_PATH_AUDIT.json保存原7104条无文本路径审查；CORRECTED_PATH_AUDIT.json审查重新解码的7104条无文本路径和7104条FA路径，全部符合原始拓扑且覆盖完整。',
        '\n修复边界：在独立新评测入口中，K2图构建放到inference_mode(False)，模型从普通FP32参数重建。原训练源、epoch权重、平均权重和历史结果均保留。旧FA全失败对应分数无效。',
        '\nLibri100仍未提交，8卡只用于未来经结果审查放行后的Libri100。WavLM已完成训练的九组评测错误另在evaluation_dispatch.json记录恢复作业，不重训。']
    (HERE/'REPORT_ZH.md').write_text('\n'.join(lines)+'\n')
    print('S3 complete',complete,'reported recipes',len(results))
    for row in results:
        b=row['common_native_10ms_metrics']['20'];print(row['head'],row['cost'],row['selected'],round(row['test_PER'],2),'P/R/F1/Rvalue',*[round(b[k],2) for k in ['precision_percent','recall_percent','f1_percent','r_value_percent']])


if __name__=='__main__':main()
