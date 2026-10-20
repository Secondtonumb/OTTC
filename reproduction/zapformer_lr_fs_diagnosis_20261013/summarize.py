"""Summarize completed isolated tests and untouched formal-history snapshots."""
import hashlib
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent


def main():
    assert (HERE/'COMPLETE.json').exists(), 'The four controlled tests must finish first.'
    report=json.loads((HERE/'report.json').read_text())
    formal=json.loads((HERE/'formal_snapshot.json').read_text())
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axs=plt.subplots(2,2,figsize=(12,8))
    table=[]
    for label,case in report['cases'].items():
        rows=case['history']
        axs[0,0].plot([r['diagnostic_step'] for r in rows],[r['ottc_sum']/4 for r in rows],label=label)
        if case['fs_weight']:
            axs[0,1].plot([r['diagnostic_step'] for r in rows],[r['forward_sum_raw'] for r in rows],label=label)
        first,last=case['first'],case['last']
        table.append(f"| {case['lr']:.5f} | {case['fs_weight']:.0f} | {first['ottc_sum']/4:.6f} | {last['ottc_sum']/4:.6f} | {100*(1-last['ottc_sum']/first['ottc_sum']):.2f}% | {first['forward_sum_raw']:.4f} → {last['forward_sum_raw']:.4f} |")
    for label,arm in formal['arms'].items():
        if label.startswith('jsd'):
            axs[1,0].plot([r['epoch'] for r in arm['epochs']],[r['ottc_per_utterance'] for r in arm['epochs']],marker='.',label=label)
        axs[1,1].plot([r['optimizer_step'] for r in arm['metrics']],[r['mapped_arpabet']['PER_percent'] for r in arm['metrics']],marker='.',label=label)
    for ax,title,x in zip(axs.flat,
        ['Fixed four-utterance JSD / utterance (120 updates)','Fixed four-utterance forward-sum (local target-normalized mean)',
         'Formal runs: completed-epoch JSD / utterance','Formal runs: mapped TIMIT dev400 PER'],
        ['Diagnostic update','Diagnostic update','Original epoch','Original optimizer step']):
        ax.set_title(title,fontsize=10);ax.set_xlabel(x);ax.grid(alpha=.2);ax.legend(fontsize=8)
    axs[1,1].set_ylabel('PER (%)')
    fig.tight_layout();fig.savefig(HERE/'lr_fs_diagnosis.png',dpi=160);plt.close(fig)
    metrics=[]
    for label,value in formal['matched_development_metrics'].items():
        mapped=value['mapped_arpabet'];fa=value['forced_k2'];tf=value['text_free']
        metrics.append(f"| {label} | {mapped['PER_percent']:.2f} | {mapped['insertions_percent']:.2f} | {mapped['deletions_percent']:.2f} | {fa['boundary_F1_20ms_percent']:.2f} | {fa['boundary_R_value_20ms_percent']:.2f} | {tf['boundary_F1_20ms_percent']:.2f} | {tf['boundary_R_value_20ms_percent']:.2f} |")
    gradients=[]
    for label,value in report['task_gradients'].items():
        a=value['state_logits'];b=value['shared_encoder_output']
        gradients.append(f"| {label} | {a['cosine']:.6f} | {a['fs_to_ottc_norm']:.4f} | {b['cosine']:.6f} | {b['fs_to_ottc_norm']:.4f} |")
    labels={'jsd_fs':'JSD + FS','jsd_only':'JSD only','ce_fs':'CE + FS','ce_only':'CE only'}
    epoch_lines=chr(10).join(f"- {labels[k]}: {v['ottc_per_utterance']:.6f}" for k,v in formal['matched_epoch_values'].items())
    text=f'''LR 与 forward-sum 的可复现诊断

原始 OTTC divergence、SP 三状态目标、mass head、四组正式训练均未修改。诊断代码与结果仅写入本目录。

实际 LR 与更新

原生模型只有一个 LR 参数组（958 个参数张量），包含 encoder/frontend/state head/mass head，没有意外的低 LR head。原第10轮 checkpoint 的 LR 为 0.0005；用户选择重规划50轮后，第11轮 LR 为 {report['cases']['lr1x_fs0']['lr']:.9f}。正式训练的实际 LR 见 formal_snapshot.json，不能继续用旧的0.0005解释当前曲线。

固定四句、相同第10轮模型/优化器、相同 seed42、120次更新：

| LR | FS weight | 初始 JSD/句 | 最终 JSD/句 | JSD降幅 | FS local mean |
| --- | --- | --- | --- | --- | --- |
{chr(10).join(table)}

将 LR 翻倍，FS=0 时最终 JSD/句从 {report['cases']['lr1x_fs0']['last']['ottc_sum']/4:.6f} 改善到 {report['cases']['lr2x_fs0']['last']['ottc_sum']/4:.6f}；FS=1 时从 {report['cases']['lr1x_fs1']['last']['ottc_sum']/4:.6f} 改善到 {report['cases']['lr2x_fs1']['last']['ottc_sum']/4:.6f}。本次短测支持提高LR有帮助，且翻倍LR+FS组合下降最多，不能认定FS直接把JSD卡住。降幅仍有限，不能宣称JSD已完全收敛，也不能证明0.03424是正式训练的最优LR。

四组初始20步均值/最终20步均值见 report.json；结果方向与最终单点相同。原LR无FS组的第一步 encoder/frontend/state head/mass head 参数相对更新分别为 {report['cases']['lr1x_fs0']['parameter_updates']['1']['encoder']['relative_delta']:.6f} / {report['cases']['lr1x_fs0']['parameter_updates']['1']['frontend']['relative_delta']:.6f} / {report['cases']['lr1x_fs0']['parameter_updates']['1']['state_head']['relative_delta']:.6f} / {report['cases']['lr1x_fs0']['parameter_updates']['1']['mass_head']['relative_delta']:.6f}，证实优化器并未停止更新。这些数值属于隔离诊断，不是正式DDP训练的参数差值。

所有组合的初始 OTTC loss 逐位一致。完整 first/last20均值和每步曲线见 report.json，不能只用最后一个点下结论。模型使用原生 BatchedRubik 和恢复的优化器状态，LR 在诊断中固定；未使用 SpecAugment。BF16未乘AMP loss scale，因此 auxiliary scale 为未缩放的0.05。原生 batch-count 固定在step1060，正交约束保持原生实现。本实验属于短程过拟合诊断，不等于正式8卡训练的50轮LR扫描。

四句诊断中，OTTC是local sum、FS是local mean；正式批次远大于四句，因此 FS 在诊断里的相对比例更大。这里测得的最优权重不能直接搬到正式训练。对照保留同样的原生 auxiliary 正则。

梯度关系（同一个 forward，排除 backbone 内部 auxiliary backward）：

| checkpoint | logits cosine | logits FS/OTTC norm | encoder output cosine | encoder output FS/OTTC norm |
| --- | --- | --- | --- | --- |
{chr(10).join(gradients)}

两处梯度都接近正交，没有发现显著相反方向或 FS 梯度压倒 OTTC 的证据。这个结论只覆盖这两份 checkpoint 和四句固定真实样本；不代表全训练集梯度永远一致。

正式训练 matched epoch {formal['matched_epoch']} 的 OTTC/句：

{epoch_lines}

JSD only 与 JSD + FS 可互比；CE only 与 CE + FS 可互比；CE 与 JSD 的 raw loss 定义不同。

固定 TIMIT dev400、相同 optimizer step {formal['matched_metric_step']} 的诊断（不是 LibriSpeech WER）：

| arm | mapped PER% | insertion% | deletion% | supplied-phone FA F1@20ms% | supplied-phone FA R@20ms% | text-free F1@20ms% | text-free R@20ms% |
| --- | --- | --- | --- | --- | --- | --- | --- |
{chr(10).join(metrics)}

完整 P/R/F1/R-value、原始记录区间、协议与 scorer/source hash 保存在 formal_snapshot.json 和 formal_snapshot/。强制对齐提供目标音素；text-free recognition 使用音频及原生 greedy complete。边界协议为保存的 native phone-level变化/10ms grid、blank previous，以及原有冻结 R-value 公式；不改 scorer、不与 MFA 论文 topline做结论比较。本记录是 backfill，保留原始 epoch/step，不插值、不把导入时间当训练时间。

forward-sum 的具体限制

现有适配先合并 S0/S1/S2，再按目标位置取分数并重新归一化。因此它不能约束原模型 acoustic blank 的绝对概率，也不能区分三状态的顺序；CTC的blank使用独立固定分数-1。此前的 forward_sum_invariance.json已验证这两项不变性。这是经确认的现有输出适配的性质，不是修改原生OTTC的理由。

正式训练中，FS两组的OTTC下降均略快；JSD+FS同时有大量识别插入，CE+FS的FA和识别则好于CE-only。因此不能把四组loss不易下降统一归因于FS，但JSD+FS确有需要检查的识别副作用。梯度近正交也不能排除目标与解码约束不一致。

复现命令：从干净的新输出目录复制 diagnose.py 与对应PBS输出路径，再提交 `qsub diagnose.pbs`。脚本显式拒绝向已有结果追加一个从step0重跑的新实验。原始 checkpoint SHA256 为 {report['origin_sha256']}。每组独立W&B run，run ID与配置记录在 cases/*/wandb_run.json；已有结果不会被重跑覆盖。
'''
    (HERE/'REPORT.md').write_text(text)
    files=[p for p in HERE.glob('*') if p.is_file() and p.name!='manifest.json']
    files.extend(HERE.glob('cases/*/updates.jsonl'));files.extend(HERE.glob('cases/*/wandb_run.json'))
    (HERE/'manifest.json').write_text(json.dumps({str(p.relative_to(HERE)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},indent=2)+'\n')
    print(chr(10).join(table))


if __name__=='__main__':main()
