"""Summarize only recorded metrics and enumerate missing tests for all formal runs."""
import ast
import csv
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2];SUITE=Path(__file__).resolve().parent
sys.path.insert(0,str(SUITE))
sys.path.insert(0,str(ROOT/'reproduction'))
from segmentation_metrics import boundary_r_value,flat_boundary_r_value

def pct(value):return '—' if value is None else f'{value:.2f}'

def mfa_protocol_summary():
    directory=SUITE/'mfa_protocol_comparison'
    complete=directory/'RESCORE_COMPLETE.json'
    if not complete.exists():return []
    coverage=json.loads(complete.read_text())
    full_count=coverage['full_test'];common_count=coverage['common_test_TSE']
    with (directory/'matrix.csv').open(newline='') as source:
        rows=[r for r in csv.DictReader(source) if r['dataset']=='timit_test']

    def select(family,scope,suffix):
        matches=[r for r in rows if r['family']==family and r['scope']==scope
                 and (family=='mfa' or r['head']=='ottc_arpabet')
                 and r['model'].endswith(suffix)]
        if len(matches)!=1:
            raise ValueError(f'Expected one MFA-protocol row: {family}/{scope}/{suffix}, got {len(matches)}')
        row=matches[0]
        expected=full_count if scope=='all' else common_count
        if int(row['records'])!=expected or (scope=='common_TSE' and int(row['eval_utts'])!=expected):
            raise ValueError(f'MFA-protocol coverage mismatch: {row["model"]}/{scope}')
        return row

    lines=['## TIMIT：OTTC 与系统身份待核实的归档 CTM 重计分','',
           f'复用已训练的四头模型和已有 K2-FSA 对齐输出，CE 没有重训。下表展示原始 OTTC ARPAbet head；边界 F1/R-value 使用完整 core test {full_count} 条，TSE 使用所有系统、两个 ARPAbet head 共同等长的 {common_count} 条子集。归档 CTM 行不是论文 MFA 3.0 的复现。','',
           f'| 模型 | checkpoint 选择 | FA F1@20ms %（{full_count}条）↑ | R-value %（{full_count}条）↑ | TSE ms（共同{common_count}条）↓ |',
           '|---|---|---:|---:|---:|']
    for family,label in [('jsd_upstream','OTTC 上游 JSD'),('ce','OTTC CE'),('ctc','CTC')]:
        for suffix in ['_dev_best','/last3_best3_weight_average']:
            boundary=select(family,'all',suffix);tse=select(family,'common_TSE',suffix)
            if boundary['model']!=tse['model']:
                raise ValueError('MFA-protocol boundary/TSE checkpoint mismatch')
            strategy=('dev PER 最佳单个（step '+boundary['model'].split('/step_')[1].removesuffix('_dev_best')+'）'
                      if suffix=='_dev_best' else '最后3个 = 最佳3个权重平均')
            lines.append(f'| {label} | {strategy} | {pct(float(boundary["boundary_20_f1_pct"]))} | {pct(float(boundary["boundary_20_r_value_pct"]))} | {pct(float(tse["phone_tse_ms"]))} |')
    boundary=select('mfa','all','TIMIT_MFA');tse=select('mfa','common_TSE','TIMIT_MFA')
    lines.append(f'| 归档 TIMIT_MFA CTM（系统身份待核实） | 当前 scorer 重计分；不是论文 MFA 3.0 | {pct(float(boundary["boundary_20_f1_pct"]))} | {pct(float(boundary["boundary_20_r_value_pct"]))} | {pct(float(tse["phone_tse_ms"]))} |')
    drift=next(r for r in coverage['boundary_version_drift'] if r['dataset']=='timit_test')
    lines+=['',
            '单个 checkpoint 沿用已冻结的 dev PER 选择，不按 test 结果挑选。仅保留 step2774、2847、2920 三个 checkpoint，因此 last3 与 best3 是同一组权重平均；全部逐个测试见完整矩阵。上游 JSD 沿用原 OTTC 实现，区别于数学标准 JSD。','',
            '直接复用并冻结 IF-MDD raw-PHN scorer 和原始 PHN，源码 hash 已记录。评分采用 dx→d、closure→silence、过滤 silence，以相邻非静音 phone 间前一 phone 的 end 作为边界。TSE 是每句平均的 |start误差|+|end误差|，再对句子平均；它与后文单侧 start MAE 的定义不同，也尚未核实与论文在样本、边界提取和聚合方式上完全一致。','',
            f'归档 CTM 历史 F1 为 {pct(drift["historical_F1_20ms"])}%，同一输出经当前冻结 scorer 重算为 {pct(drift["rescored_F1_20ms"])}%；差异来自边界候选匹配顺序的版本变化，历史值仍保留。表中所有系统均用本次 scorer 重算。归档 CTM 自身等长子集为156条；共同子集进一步排除原 OTTC head 因映射/相邻重复融合而音素数量不等的4条。','',
            '归档导出记录的 target 为 word_transcript，模型与输入构图来源尚待核实；OTTC/CTC K2 FA 使用参考音素序列。归档 CTM 没有 text-free PER。论文 MFA 3.0、这里的归档 CTM、后文使用 MFA 标签训练的 Zapformer 分开报告。','',
            '[两个 ARPAbet head 的完整对照与 W&B](mfa_protocol_comparison/README.md) · [评分源码、输入与子集来源](mfa_protocol_comparison/provenance.json)。后文保留原评测协议的 text-free 与 K2 FA 指标，其边界定义不同，应在各自表内比较。','']
    verification=directory/'MFA_ROW_VERIFIED.json'
    if verification.exists():
        verified=json.loads(verification.read_text())
        if verified['verification']!='PASS':raise ValueError('MFA row verification failed')
        for scope,current in [('all',boundary),('common_TSE',tse)]:
            replay=next(r for r in verified['recomputed_results'] if r['scope']==scope)
            for key in ['records','eval_utts','phone_tse_ms','boundary_20_f1_pct','boundary_20_r_value_pct']:
                if abs(float(current[key])-replay[key])>1e-10:
                    raise ValueError(f'MFA row verification is stale: {scope}/{key}')
        lines+=['**归档 CTM 数值核验与身份限制：** 原始 `IF-MDD/TIMIT_MFA/ctm.phone.timit.test` 的192条输出与归档 JSONL 的标签、时间一致；冻结 scorer 和原始 PHN 的 hash 一致，重新计分复现上表。此核验只证明数值，不能证明它来自 MFA；论文身份对照见上节。','',
                f'共同{common_count}条的归档 CTM 起点平均绝对误差为 {pct(verified["common_mean_start_absolute_error_ms"])}ms，终点为 {pct(verified["common_mean_end_absolute_error_ms"])}ms，相加才是表中的 {pct(verified["independent_TSE_common_152_ms"])}ms TSE。历史完整等长156条的 TSE 为 {pct(verified["independent_TSE_all_156_ms"])}ms。等长子集按位置配对，不要求 phone 标签序列完全相同；归档 CTM 在共同{common_count}条中只有{verified["common_identical_normalized_phone_sequence_utts"]}条归一化标签序列与参考完全一致，TSE仍包含音素序列差异。','',
                '[原始 CTM、hash 与数值重算证据（不验证模型身份）](mfa_protocol_comparison/MFA_ROW_VERIFIED.json)。','']
    return lines

def main():
    lines=['# 当前实验结果与测试覆盖','',f'更新：{datetime.now().astimezone().isoformat(timespec="seconds")}。','',
           '[三个 checkpoint 的逐个测试与权重平均比较](../checkpoint_comparison/README.md)。','',
           '[✅ / 🟡 / ⏳ 已跑实验标记](EXPERIMENT_STATUS.md) · [声学前端差异](FRONTENDS.md)。','',
           '[论文 MFA 3.0 与归档 CTM 身份核对](MFA_TOPLINE.md) · [当前重点：OTTC 上游 JSD](JSD_RESULTS.md)。','',
           '[CE/JSD/CTC 与归档 CTM 统一重计分](mfa_protocol_comparison/README.md)：完整 test192 的 F1/R-value 与各系统共同等长子集的 TSE。','',
           '四头 TIMIT 三组已完成最终 core test 和 K2-FSA forced alignment；单头五组已完成原 README、core test、Buckeye、L2-ARCTIC。LibriSpeech 各组完成状态按实际文件标记；MFA 中途 dev/test 与最终结果分开。不同 phoneset、测试范围与 loss 定义不可混成同一排名。','']
    from paper_mfa_report import paper_mfa_notice
    lines+=paper_mfa_notice()
    lines+=mfa_protocol_summary()
    backfill_identity=SUITE/'rvalue_backfill/wandb_run.json'
    if backfill_identity.exists():
        url=json.loads(backfill_identity.read_text())['url']
        lines+=['[W&B：历史 dev R-value 曲线与 test/checkpoint 补算表]('+url+') · [保留原 epoch/step 的 CSV](rvalue_backfill/dev_rvalue.csv)。历史曲线为 P/R 派生回填，时间轴不使用回填时间。','']
    lines+=['R-value 已补齐：历史缺失值由同一条记录的 Precision/Recall 计算，直接复用冻结 IF-MDD 公式；不修改边界提取或匹配，也不重训/重推理。训练 dev、最终 test、text-free 与 K2 FA 分别报告；常规 scorer 和上方 raw-PHN scorer 的边界协议仍有区别。','']
    lines+=['## 四头 TIMIT：最终 core test 192','',
           '| 目标 | head | 有标签条数 | Text-free PER / token error % ↓ | Text-free F1@20ms % ↑ | Text-free R-value % ↑ | K2 FA F1@20ms % ↑ | K2 FA R-value % ↑ | K2 start MAE ms ↓ |',
           '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    audit=[]
    for objective in ['ctc','ce','jsd_upstream']:
        directory=ROOT/'reproduction/multihead_timit/models'/objective
        metrics=json.loads((directory/'test_metrics.json').read_text())
        modes=json.loads((ROOT/'reproduction/multihead_timit/evaluation_modes'/objective/'metrics.json').read_text())['splits']['test']
        for head in ['ottc_arpabet','ifmdd_arpabet','mfa_ipa','zipa_ipa']:
            tf=modes['text_free'][head];fa=modes['forced_k2'][head]
            error=tf.get('PER_percent',tf.get('token_error_percent'))
            f1=tf.get('boundaries',{}).get('20',{}).get('f1_percent');ff=fa.get('boundaries',{}).get('20',{}).get('f1_percent')
            lines.append(f'| {objective} | {head} | {tf["supervised_utterances"]} | {pct(error)} | {pct(f1)} | {pct(boundary_r_value(tf.get("boundaries",{}).get("20")))} | {pct(ff)} | {pct(boundary_r_value(fa.get("boundaries",{}).get("20")))} | {pct(fa.get("start_MAE_ms"))} |')
        audit.append({'experiment':'timit-fourhead/'+objective,'test':'complete','k2':'complete','metrics':str(directory/'test_metrics.json')})
    lines+=['','JSD 组保留上游反向 KL 混合实现，区别于数学标准 JSD。MFA/ZIPA 四头没有真实音素区间参考，仅导出 spans，不填写分段准确率。MFA 四头按词典完整覆盖子集 158 条计分。','',
            'TIMIT 四头最近真实 train loss：CTC 0.4765、CE 0.5157、上游 JSD 1.3907；三组均完成 40 轮 / 2920 updates。loss 定义不同，不按量级排名。','',
            '[完整 TIMIT 四头报告与 loss 曲线](../multihead_timit/notion_update/README.md)。','',
            '## 单头 fused phoneme：已完成的测试','',
            '| 目标 | 测试集 | 条数 | Raw PER % ↓ | F1@20ms % ↑ | R-value % ↑ |','|---|---|---:|---:|---:|---:|']
    for name,label in [('runexp_ctc_fuse_phoneme','CTC'),('runexp_ottc_fuse_phoneme','OTTC CE'),('runexp_ottc_fuse_phoneme_jsd','OTTC upstream jsd'),('runexp_ottc_fuse_phoneme_jsd_standard','OTTC standard JSD'),('runexp_softdtw_fuse_phoneme','Soft-DTW CE gamma0.1')]:
        p=ROOT/'reproduction/results'/name/'metrics.json';r=json.loads(p.read_text())
        for split,m in r['splits'].items():
            lines.append(f'| {label} | {split} | {m["utterances"]} | {100*m["raw_PER"]:.2f} | {100*m["boundaries"]["20"]["f1"]:.2f} | {pct(boundary_r_value(m["boundaries"]["20"]))} |')
        audit.append({'experiment':name,'test':'complete','splits':list(r['splits']),'k2':'not performed for historical single-head series','metrics':str(p)})
    lines+=['','单头与四头训练设置和评测范围不同。单头分段为已有 text-free 协议，未冒充 K2 FA 结果。Soft-DTW 此配置出现严重删除、结果较差，不能用 loss 下降断言收敛良好。','',
            '## LibriSpeech 当前 dev 与最终 test 状态','',
            '[W&B dev PER 与分段曲线]('+json.loads((SUITE/'dev_wandb_report.json').read_text())['url']+')。','',
            '| 规模 / 目标 | dev epoch | head | PER / token error % | Precision@20ms % | Recall@20ms % | F1@20ms % | R-value % | 最终 test |',
            '|---|---:|---|---:|---:|---:|---:|---:|---|']
    from track import live_progress
    for dataset in ['libri100','libri960']:
        for objective in ['ctc','ce','jsd_upstream']:
            directory=ROOT/f'reproduction/multihead_{dataset}/models'/objective
            dev=live_progress(directory).get('last_dev',{})
            test=(directory/'test_metrics.json').exists();live=live_progress(directory)
            status='✅ 已完成' if test else '🟡 训练结束后自动 test' if live.get('last_train') else '⏳ 训练尚未启动'
            for head in ['ottc_arpabet','ifmdd_arpabet','mfa_ipa','zipa_ipa']:
                per=dev.get('eval_PER_percent/'+head,dev.get('eval_token_error_percent/'+head))
                values=[pct(dev.get('eval_boundary_'+k+'_20ms/'+head)) for k in ['precision','recall','F1']]
                values.append(pct(flat_boundary_r_value(dev,'eval_',head)))
                lines.append('| '+' | '.join([dataset+'/'+objective,str(dev.get('epoch','—')),head,pct(per),*values,status])+' |')
            audit.append({'experiment':dataset+'/'+objective,'test':'complete' if test else 'pending-training','k2':'complete' if (directory.parent.parent/'evaluation_modes'/objective/'EVAL_COMPLETE.json').exists() else 'scheduled-after-complete'})
    lines+=['','## LibriSpeech 已完成的最终 text-free test','',
            '| 数据 / 目标 | head | Test PER / token error % | F1@20ms % | R-value % |','|---|---|---:|---:|---:|']
    for dataset in ['libri100','libri960']:
        for objective in ['ctc','ce','jsd_upstream']:
            path=ROOT/f'reproduction/multihead_{dataset}/models/{objective}/test_metrics.json'
            if not path.exists():continue
            metrics=json.loads(path.read_text())
            for head in ['ottc_arpabet','ifmdd_arpabet','mfa_ipa','zipa_ipa']:
                per=metrics.get('test_PER_percent/'+head,metrics.get('test_token_error_percent/'+head))
                lines.append(f'| {dataset}/{objective} | {head} | {pct(per)} | {pct(metrics.get("test_boundary_F1_20ms/"+head))} | {pct(flat_boundary_r_value(metrics,"test_",head))} |')
    lines+=['','LibriSpeech 时间参考来自自动对齐；中途 dev 在不同 epoch，不能作为最终公平比较。本地原名称为 eval_PER_percent/<head>、eval_boundary_F1_20ms/<head>；W&B 显示 eval/PER_percent/<head>、eval/boundary_F1_20ms/<head>。新进程同时写明确的 dev/* 别名。每组最终训练自带 test，另由 evaluate.pbs 补 K2 FA。','',
            '| Libri100 目标 | 最近 train loss | 记录 epoch | 最新训练步数 |','|---|---:|---:|---:|']
    for objective in ['ctc','ce','jsd_upstream']:
        live=live_progress(ROOT/f'reproduction/multihead_libri100/models/{objective}');train=live.get('last_train',{})
        lines.append(f'| {objective} | {pct(train.get("loss"))} | {train.get("epoch","—")} | {live.get("step","—")} |')
    lines+=['',
            '## MFA English 外部 TIMIT dev/test','',
            '| 目标 | checkpoint step | 阶段 | Split | PER % | Text-free F1@20ms % | Text-free R-value % | K2 FA F1@20ms % | K2 FA R-value % |','|---|---:|---|---|---:|---:|---:|---:|---:|']
    for arm in ['ctc','ottc','spottc','film_k4']:
        latest=SUITE/'mfa_models'/arm/'latest_dev_metrics.json'
        if latest.exists():
            record=json.loads(latest.read_text());metrics=record['text_free']
            lines.append(f'| {arm} | {record["optimizer_step"]} | 正式逐轮 dev | development | {pct(metrics["PER_percent"])} | {pct(metrics["boundary_F1_20ms_percent"])} | {pct(flat_boundary_r_value(metrics))} | — | — |')
        paths=sorted((SUITE/'mfa_evaluation'/arm).glob('step-*/metrics.json'),key=lambda p:int(p.parent.name[5:]))
        if paths:
            r=json.loads(paths[-1].read_text())
            for split,m in r['splits'].items():
                lines.append(f'| {arm} | {r["progress"]["global_step"]} | '+('最终' if r['final'] else '固定中途 checkpoint 诊断')+f' | {split} | {pct(m["text_free"]["PER_percent"])} | {pct(m["text_free"]["boundary_F1_20ms_percent"])} | {pct(flat_boundary_r_value(m["text_free"]))} | {pct(m.get("forced_k2",{}).get("boundary_F1_20ms_percent"))} | {pct(flat_boundary_r_value(m.get("forced_k2",{})))} |')
        elif not latest.exists():lines.append('| '+arm+' | — | 待补 dev/test'+' | — | — | — | — | — | — |')
        audit.append({'experiment':'mfa/'+arm,'test':'current-diagnostic-complete' if paths and 'test' in r['splits'] else 'queued-current-checkpoint' if arm!='film_k4' else 'training-not-started','final_test':'automatic at epoch10','epoch_dev':'initial-resume + every completed epoch (new processes)'})
    lines+=['', '| MFA 当前 test 目标 | K2 start MAE ms | K2 end MAE ms | K2 token IoU % | W&B |',
            '|---|---:|---:|---:|---|']
    for arm in ['ctc','ottc','spottc']:
        paths=sorted((SUITE/'mfa_evaluation'/arm).glob('step-*/metrics.json'),key=lambda p:int(p.parent.name[5:]))
        if paths:
            path=paths[-1];r=json.loads(path.read_text());k=r.get('splits',{}).get('test',{}).get('forced_k2',{})
            identity=path.parent/'wandb_run.json';url=json.loads(identity.read_text()).get('url') if identity.exists() else None
            if k:lines.append(f'| {arm} | {pct(k["start_MAE_ms"])} | {pct(k["end_MAE_ms"])} | {pct(k["mean_token_IoU_percent"])} | '+('[W&B]('+url+')' if url else '—')+' |')
    lines+=['','MFA 原先没有 dev/test logging，已经增加下一进程恢复时 dev 和每轮 dev。development 使用原配置的 TIMIT 4096 条，core test 192 条；它们没有进入 MFA 全量训练。按冻结的 TIMIT→MFA 标签映射计 PER，闭塞/释放相邻融合；该结果包含标注体系差异，与四头词典 MFA IPA 目标不同。SPOTTC/FiLM 采用原 S3T2 greedy complete；K2 FA 在三状态边缘化的 phone posterior 上执行，与原 substate topology FA 区分。','',
            '当前固定中途 checkpoint 的 MFA test PER 约 CTC 45.12%、OTTC 61.14%、SPOTTC 45.16%，效果仍差。SPOTTC 的 K2 起止 MAE 约 305/303ms、IoU 接近 0，需要单独调查时序和 substate 解释；CTC 为 32/34ms、OTTC 为 53/56ms。未对预测拟合或修正时间偏移，未基于 test 改超参或挑选 checkpoint。','',
            '## 自动保存、切换与补测','',
            'MFA 当前片段在预算到时保存 optimizer/sampler/RNG，正常退出后续训提交优化配置；新片段同时启用逐轮外部 dev。Libri100 保持当前设置完成，保存及 test 后进入 microbatch8 的 Libri960。所有新 run 沿用 W&B；test 不参与 checkpoint 选择。','',
            'MFA 数据准备 job 3512374、四模型输出一致性 gate 3512417，以及三组完整 dev4096 / core test192 / K2 test192 jobs 3512419 / 3512434 / 3512435 均已成功完成。当前诊断 test 和最终 test 分开存放，不覆盖。FiLM 正式训练尚未启动；现有 30-step 验证权重仅做 2 条推理校验，不当作正式指标。MFA CTC 已正常保存 step26266，并自动提交优化续训 job3512447；其余组在片段结束时自动切换。新的八 rank 逐轮 dev hook 等待优化续训启动后验证。最终 epoch10 自动 test。','',
            '[最新进度与 W&B](TRACKING.md) · [吞吐实测](THROUGHPUT.md) · [实验配置](README.md)。','']
    (SUITE/'RESULTS.md').write_text('\n'.join(lines));(SUITE/'test_coverage.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2)+'\n')
    from mark_experiments import main as mark_experiments
    mark_experiments()
    from summarize_jsd import main as summarize_jsd
    summarize_jsd()

if __name__=='__main__':main()
