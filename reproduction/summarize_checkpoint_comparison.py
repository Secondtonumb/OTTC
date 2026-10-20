"""Publish only completed comparisons; retain exact candidate scope and source steps."""
import json
from datetime import datetime
from pathlib import Path
from segmentation_metrics import boundary_r_value,flat_boundary_r_value

ROOT=Path(__file__).resolve().parent/'checkpoint_comparison'


def main():
    lines=['# 三个 checkpoint 的比较','',f'更新：{datetime.now().astimezone().isoformat()}。','',
           '比较最后三个逐个测试、最后三个权重平均，以及 dev PER 最好的三个权重平均。最后三个指已保存 checkpoint，可能相隔一两个更新，并非三个 epoch。权重均匀平均；不平均 optimizer/RNG。四头按原始 OTTC ARPAbet dev PER 选，MFA 按原生 phone dev PER 选；test 不参与选择。','',
           'TIMIT 只剩最后三个权重，因此“最好三个”只能在这三个中选，与最后三个平均相同，不能称为全程最好三个。Libri100 现存及以后权重已增加独立保存；中途快照与最终结果单独标记。MFA 对全部现存 checkpoint 补 dev 后选三个。','',
           '| 数据 / 目标 | 状态 | 候选数 | Job |','|---|---|---:|---|']
    submissions=ROOT/'submissions.json'
    jobs=json.loads(submissions.read_text()) if submissions.exists() else {}
    tracking=ROOT.parent/'english_suite/tracking.json'
    snapshot=json.loads(tracking.read_text()) if tracking.exists() else {}
    for plan_path in sorted(ROOT.glob('*/plan.json')):
        p=json.loads(plan_path.read_text());d=plan_path.parent
        marker=(d/'results/COMPARE_COMPLETE.json') if p['family']=='multihead' else d/'COMPARE_COMPLETE.json'
        state=snapshot.get('pbs',{}).get('三个 checkpoint 比较 '+d.name,{}).get('job_state')
        status='✅ 已完成' if marker.exists() else '🟡 正在评估' if state=='R' else '⏳ 排队' if state in ['H','Q'] else '已提交' if d.name in jobs else '准备完成'
        lines.append(f'| {d.name} | '+status+f' | {len(p["candidates"])} | {jobs.get(d.name,"—")} |')
    for path in sorted([*ROOT.glob('*/results/comparison.json'),*ROOT.glob('*/comparison.json')]):
        if not (path.parent/'COMPARE_COMPLETE.json').exists():continue
        r=json.loads(path.read_text());name=path.parent.parent.name if path.parent.name=='results' else path.parent.name
        identity=path.parent/'wandb_run.json';url=json.loads(identity.read_text()).get('url') if identity.exists() else None
        lines+=['', '## '+name+('（中途 checkpoint）' if r['intermediate_diagnostic'] else '（训练完成）'),'']
        if name=='mfa_spottc' and (path.parent/'report_recovery.json').exists():
            lines+=['评测与 W&B 上传已完成；原 PBS 因 Markdown 汇总字段冲突 exit1。已读取冻结 selection 文件恢复汇总，未重跑 GPU 或修改评分。','']
        sel=r['selection']
        # Older MFA records merged provenance after selection, overwriting
        # this field with the selection-policy description. The frozen file
        # retains the actual dev-only ranking; do not rerun or alter metrics.
        if not isinstance(sel,dict):
            sel=json.loads((path.parent/'selection.json').read_text())
        lines.append('最后三个：'+str(sel['last3'])+'；最好三个：'+str(sel['best3'])+'；dev 最好单个：'+str(sel['best_single'])+'。')
        if url:lines+=['','[W&B]('+url+')']
        lines+=['','| 策略 / step | head | Test PER / token error % ↓ | Text-free F1@20ms % ↑ | Text-free R-value % ↑ | K2 FA F1@20ms % ↑ | K2 FA R-value % ↑ |','|---|---|---:|---:|---:|---:|---:|']
        values=[('step '+step,m) for step,m in r['individual_tests'].items()]+list(r['weight_averages'].items())
        def fmt(v):return '—' if v is None else f'{v:.2f}'
        for label,m in values:
            if 'splits' in m:
                test=m['splits']['test']
                for head,t in test['text_free'].items():
                    f=test['forced_k2'][head]
                    lines.append('| '+' | '.join([label,head,fmt(t.get('PER_percent',t.get('token_error_percent'))),fmt(t.get('boundaries',{}).get('20',{}).get('f1_percent')),fmt(boundary_r_value(t.get('boundaries',{}).get('20'))),fmt(f.get('boundaries',{}).get('20',{}).get('f1_percent')),fmt(boundary_r_value(f.get('boundaries',{}).get('20')))])+' |')
            else:
                test=m.get('test',m);t=test['text_free'];f=test['forced_k2']
                lines.append('| '+' | '.join([label,'MFA phones',fmt(t['PER_percent']),fmt(t['boundary_F1_20ms_percent']),fmt(flat_boundary_r_value(t)),fmt(f['boundary_F1_20ms_percent']),fmt(flat_boundary_r_value(f))])+' |')
        lines+=['','逐个模型的指标均值及标准差见 comparison.json；它们与权重平均模型的重新测试分开记录。']
    temporary=ROOT/'README.md.partial';temporary.write_text('\n'.join(lines)+'\n');temporary.replace(ROOT/'README.md')
    import sys
    sys.path.insert(0,str(ROOT.parent/'english_suite'))
    from summarize_jsd import main as summarize_jsd
    summarize_jsd()


if __name__=='__main__':main()
