"""Describe the bounded probe without presenting it as full-set results."""
import csv,json
from pathlib import Path
HERE=Path(__file__).resolve().parent


def main():
 complete=json.loads((HERE/'EVAL_COMPLETE.json').read_text())
 records=[json.loads(s) for s in (HERE/'predictions.jsonl').read_text().splitlines()]
 assert len(records)==20 and complete['jobs']==9 and complete['utterances']==4
 lookup={(r['recording_id'],r['model_label'],r['method']):r for r in records}
 labels=['CIF CE+Quantity','CIF + CTC','CIF + OTTC']
 ids=list(dict.fromkeys(r['recording_id'] for r in records))
 lines=['# CIF 条件分段与辅助头 K2 FA：四例实测','',
 '本次实际运行9个已训练checkpoint，覆盖4条语音、12个模型×语音组合，未重训。共12条CIF长度条件分段和8条辅助头K2 FA；仅为可视化四例probe，不是完整测试集。','',
 '输入条件：K2辅助头给定音素序列；CIF长度条件只给定音素数量，音素内容不参与边界搜索；二者均不输入参考时间。纯CE+Quantity没有逐帧辅助头，不虚构K2结果。','',
 '共同目标沿用图中参考音素序列：Libri省去CIF训练目标中的首尾sil但不裁剪音频；native dx保留，检查共同标签时映为d。所有新预测时间均保留duration/valid_frames时钟，无拟合偏移。','',
 '下表为当前整句20ms容差的 **F1 / R-value（%）**。自由边界也从同一checkpoint在同一duration/frames时钟下重新计算，便于隔离长度条件的作用；因此不承诺逐位等于此前固定20ms图示数字。','',
 '|语音|模型|自由CIF边界|CIF给定数量|辅助头K2 FA|','|---|---|---:|---:|---:|']
 def cell(metrics):return f'{metrics["f1_pct"]:.2f} / {metrics["r_value_pct"]:.2f}'
 rows=[]
 for rid in ids:
  for label in labels:
   native=lookup[rid,label,'cif_length_conditioned'];aux=lookup.get((rid,label,'auxiliary_k2'))
   aux_cell='无逐帧头' if aux is None else '失败' if aux.get('alignment_failed') else cell(aux['metrics']['20'])
   lines.append(f'|{rid}|{label}|{cell(native["text_free_boundary_baseline"]["metrics"]["20"])}|{cell(native["metrics"]["20"])}|{aux_cell}|')
   for mode,by_tol in [('text_free',native['text_free_boundary_baseline']['metrics']),('cif_length_conditioned',native['metrics'])]+([('auxiliary_k2',aux['metrics'])] if aux and not aux.get('alignment_failed') else []):
    for ms,metric in by_tol.items():
     rows.append(dict(recording_id=rid,model=label,method=mode,tolerance_ms=int(ms),**{k:metric[k] for k in ['precision_pct','recall_pct','f1_pct','r_value_pct']}))
 lines+=['',f'辅助头K2对齐失败：{complete["alignment_failures"]}/8。K2路径逐条验证折叠后等于给定目标音素序列；权重严格加载、epoch/step核对、推理前后hash一致。',
 '', '边界协议：共同图示参考的唯一端点，距语音两端20ms内剔除，一对一时间匹配；10/20/25ms的P/R/F1/R-value见BOUNDARY_METRICS.csv。R-value使用冻结IF-MDD实现。TIMIT/Buckeye参考为人工标注，Libri参考为模型身份未核实的既有自动对齐。',
 '', 'TIMIT另按历史冻结raw-PHN scorer复算，保存于predictions.jsonl的timit_raw_PHN_separate，scorer/PHN hash保存在EVAL_COMPLETE.json。raw-PHN与图示参考协议不同；没有执行共同等音素数TSE比较，不把样例数值与论文MFA基线直接比较。',
 '', '文件：INPUTS.json记录输入与源模型；runs/各模型目录保存完整区间、逐帧K2路径、计时、显存、源hash和完成标记；telemetry.csv/job.log为实际GPU运行证据。checkpoints/是既有权重的硬链接，无重复权重副本。',
 '', '复现：在PBS分配的GH200上执行 `bash reproduction/cif_fa_probe_20261017/evaluate.pbs`。已完成项跳过；没有优化器更新。W&B只记录数值摘要，不上传音频/转写/TextGrid附件。',
 '', '[查看FA与CIF条件分段](../segmentation_figures_20261017/interactive/forced_alignment.html)。']
 (HERE/'RESULTS_ZH.md').write_text('\n'.join(lines)+'\n')
 with (HERE/'BOUNDARY_METRICS.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 print('\n'.join(lines[10:26]))

if __name__=='__main__':main()
