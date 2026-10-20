"""Report cited paper values without assigning its MFA identity to old CTMs."""
import json
from pathlib import Path

SUITE=Path(__file__).resolve().parent


def paper_mfa_notice(prefix=''):
    path=SUITE/'paper_mfa_audit/audit.json'
    if not path.exists():return []
    audit=json.loads(path.read_text())
    lines=['## 论文 MFA 3.0：引用值与归档身份纠正','',
           '此前将 `IF-MDD/TIMIT_MFA` 目录中的归档 CTM 标为“原生 MFA topline”是错误的：文件数值可重算，不代表模型身份已核实。下面是用户指定论文 Table II 的原文数值。','',
           '| 论文系统 | Dev TSE ms ↓ | Dev R-value % ↑ | Test TSE ms ↓ | Test R-value % ↑ |',
           '|---|---:|---:|---:|---:|']
    for row in sorted(audit['paper_rows'],key=lambda r:0 if r['system'].startswith('MFA') else 1):
        label='MFA 3.0' if row['system'].startswith('MFA') else 'Kaldi TDNN-F'
        lines.append('| '+label+' | '+' | '.join(f'{row[k]:.2f}' for k in
                     ['dev_TSE_ms','dev_R_value_percent','test_TSE_ms','test_R_value_percent'])+' |')
    lines+=['',
            '归档 CTM 使用 closure→stop 合并和当前冻结 scorer 后，dev/test 四项数值均在两位小数上与论文 Kaldi TDNN-F 行一致（45.39/63.59、45.41/63.53）。它的模型来源仍待核实，不能沿用目录名称认定为 MFA 3.0。','',
            '论文 MFA 的 F1 未报告，原始输出尚未找到，未完成同一评分口径的本地复现。表中论文值仅为引用；本地 OTTC 使用 TIMIT 训练并按 TIMIT dev PER 选 checkpoint，论文神经模型使用 LibriSpeech 训练，因此这些数值不作为严格公平排名。此前“OTTC 超过原生 MFA topline”的解释撤回。','',
            '[论文 Table II]('+audit['table_url']+') · [网页快照与归档 CTM 身份核对]('+prefix+'paper_mfa_audit/audit.json)。','']
    return lines
