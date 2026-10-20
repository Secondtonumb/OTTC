"""Export a timestamped experiment report; does not publish to Notion."""
import ast
import hashlib
import json
import re
from datetime import datetime,timezone,timedelta
from pathlib import Path

ROOT=Path(__file__).resolve().parent
REPRO=ROOT.parent
HEADS=['ottc_arpabet','ifmdd_arpabet','mfa_ipa','zipa_ipa']
NAMES=['原始 OTTC ARPAbet','IF-MDD ARPAbet','MFA IPA','ZIPA IPA127']
OBJECTIVES={'ctc':'CTC','ce':'OTTC + CE','jsd_upstream':'OTTC + 上游 jsd'}
now=datetime.now(timezone(timedelta(hours=9))).isoformat(timespec='seconds')
snapshot={'captured_at':now,'published_to_notion':False,'runs':{}}
blocks=[];lines=[]

def rich(text):return [{'type':'text','text':{'content':str(text)}}]
def paragraph(text):
    lines.extend([text,''])
    blocks.append({'object':'block','type':'paragraph','paragraph':{'rich_text':rich(text)}})
def heading(text):
    lines.extend(['## '+text,''])
    blocks.append({'object':'block','type':'heading_2','heading_2':{'rich_text':rich(text)}})
def table(header,rows):
    lines.extend(['| '+' | '.join(header)+' |','| '+' | '.join(['---']*len(header))+' |'])
    lines.extend('| '+' | '.join(map(str,row))+' |' for row in rows);lines.append('')
    blocks.append({'object':'block','type':'table','table':{'table_width':len(header),
                  'has_column_header':True,'has_row_header':False,'children':[
                  {'object':'block','type':'table_row','table_row':{'cells':[rich(c) for c in row]}}
                  for row in [header,*rows]]}})

heading('OTTC 英语音素实验：单头复现与四 phoneset 联合训练')
paragraph('快照时间：'+now+'。本文为本地实验记录，尚未发布到 Notion。')
paragraph('当前范围：先用 TIMIT train，之后扩到 LibriSpeech 100h。'
          '当前三组模型分别为 CTC、OTTC+CE、OTTC+原始上游 jsd；'
          '每组共享一个编码器，同时训练四个 phoneset 头。')
heading('四头配置及监督来源')
metadata=json.loads((ROOT/'data/metadata.json').read_text())
table(['Head','输出数（含特殊标签）','Train / dev / test 有标签条数'],[
    [NAMES[i],str(metadata['head_vocab_sizes'][h]),counts] for i,(h,counts) in enumerate(zip(HEADS,
    ['4620 / 400 / 192','4620 / 400 / 192','3909 / 339 / 158','4620 / 400 / 192']))])
paragraph('原始 OTTC 头沿用 fused mapping，保留 q、dx；IF-MDD 头沿用旧输出表，'
          'TIMIT 目标做 60→39、删除 q、dx→d，完整输出表保留未监督的 ao/zh。'
          'MFA 头采用官方 US IPA v3.1 词典的最高概率发音，不是 MFA 声学对齐；'
          '词典覆盖不足的语音仅屏蔽 MFA loss，训练覆盖率 84.61%。'
          'ZIPA 使用原生 IPA127 输出表及显式 ARPAbet→IPA 标签映射，未使用 ZIPA 模型伪标签。')
paragraph('底座为官方 facebook/wav2vec2-large-lv60 原始权重。TIMIT train 4620'
          '（包含 SA1/SA2）、dev 400、core test 192，三个 split 说话人不重叠。'
          'seed 42；40 epochs；有效 batch 64、microbatch 16、梯度累积 4；'
          'CNN 冻结；BF16；AdamW weight decay 0.005、warmup 500、linear decay。'
          'CTC lr=5e-5，OTTC lr=2e-4；四头等权；OTTC 完成第 30 轮后冻结对齐分支。')
heading('当前训练进度及 W&B')
progress=[];metrics=[];test_rows=[];loss_rows=[];head_loss_rows=[]
for objective,label in OBJECTIVES.items():
    directory=ROOT/'models'/objective
    entries=[];text=(directory/'train.log').read_text(errors='replace')
    for line in text.splitlines():
        if line.startswith('{') and line.endswith('}'):
            try:entry=ast.literal_eval(line)
            except (ValueError,SyntaxError):continue
            if isinstance(entry,dict):entries.append(entry)
    train=next((e for e in reversed(entries) if 'loss' in e),{})
    first_train=next((e for e in entries if 'loss' in e),{})
    evaluation=next((e for e in reversed(entries) if 'eval_loss' in e),{})
    complete_path=directory/'TRAIN_COMPLETE.json'
    completed=json.loads(complete_path.read_text()) if complete_path.exists() else None
    test_path=directory/'test_metrics.json'
    final=json.loads(test_path.read_text()) if completed and not completed['smoke'] and completed['epochs']>=40 and test_path.exists() else None
    steps=re.findall(r'(\d+)/2920',text)
    step=int(steps[-1]) if steps else 0
    if completed:step=completed['steps']
    identity=json.loads((directory/'wandb_run.json').read_text())
    runtimes=[e['train_runtime'] for e in entries if 'train_runtime' in e]
    snapshot['runs'][objective]={'step':step,'total_steps':2920,'latest_training':train,'first_training':first_train,
                               'latest_validation':evaluation,'completion':completed,'final_test':final,
                               'trainer_runtime_segments_seconds':runtimes,
                               'wandb_url':identity['url'],'log_sha256':hashlib.sha256(text.encode()).hexdigest()}
    status='完成' if final else '已保存断点，等待续训' if (directory/'CHECKPOINT_PAUSED.json').exists() else '训练中'
    progress.append([label,status,f'{step}/2920',str(evaluation.get('epoch','—')),f'{train.get("loss",0):.4f}',identity['url']])
    if first_train and train:
        loss_rows.append([label,f'{first_train["loss"]:.4f}',f'{train["loss"]:.4f}',
                          f'{100*(1-train["loss"]/first_train["loss"]):.2f}%',f'{sum(runtimes)/60:.2f}'])
    for h,name in zip(HEADS,NAMES):
        error_kind='token_error_percent' if h=='zipa_ipa' else 'PER_percent'
        error=evaluation.get(f'eval_{error_kind}/{h}')
        f1=evaluation.get(f'eval_boundary_F1_20ms/{h}')
        metrics.append([label,name,str(evaluation.get('epoch','—')),f'{error:.2f}' if error is not None else '—',
                        f'{f1:.2f}' if f1 is not None else '无真实参考边界'])
        loss_key='head_loss/'+h
        if loss_key in first_train and loss_key in train:
            head_loss_rows.append([label,name,f'{first_train[loss_key]:.4f}',f'{train[loss_key]:.4f}'])
        if final:
            error=final[f'test_{error_kind}/{h}'];f1=final.get(f'test_boundary_F1_20ms/{h}')
            precision=final.get(f'test_boundary_precision_20ms/{h}')
            recall=final.get(f'test_boundary_recall_20ms/{h}')
            test_rows.append([label,name,str(final[f'test_supervised_utterances/{h}']),f'{error:.2f}',
                              f'{precision:.2f}' if precision is not None else '—',
                              f'{recall:.2f}' if recall is not None else '—',
                              f'{f1:.2f}' if f1 is not None else '无真实参考边界'])
all_complete=all(r['final_test'] for r in snapshot['runs'].values())
snapshot['all_runs_complete']=all_complete
table(['Objective','状态','更新步数','最新 dev epoch','最近 train loss','W&B'],progress)
paragraph('三组原始 loss 的定义/reduction 不同，不能按数值高低排名；'
          +('三组均完成 40 轮、2920 个更新步，最终 core test 已完成。' if all_complete else
            '当前验证来自不同 epoch，不能作为公平最终性能对比。')+
          '四头扩展实验也不是未修改的单头 README 复现。')
heading('Loss 下降情况')
if (ROOT/'notion_update/figures/train_losses.png').exists():
    paragraph('![CTC、OTTC CE 与上游 jsd 的已记录训练 loss](figures/train_losses.png)')
    paragraph('三幅图分别使用各自纵轴，仅绘制真实记录点；Markdown 是本地快照，W&B 链接用于查看在线曲线。')
table(['Objective','首个记录 train loss','末个记录 train loss','下降比例','累计 Trainer runtime（分钟）'],loss_rows)
paragraph('首个记录为 step 50 / epoch 0.68，末个记录为 step 2900 / epoch 39.73；'
          '不补造 step 0 或 2920 的训练 loss。runtime 为各训练调用实测时间之和，'
          '包含每轮评估及保存开销，不包含排队；断点续训段已相加，未使用续训时被总步数稀释的 train_loss。'
          '这些下降比例描述各自曲线，不等同于不同目标之间的性能排名或已充分收敛的证明。')
table(['Objective','Head','首个 train head loss','末个 train head loss'],head_loss_rows)
if test_rows:
    heading('Text-free inference：最终 TIMIT core test 192')
    table(['Objective','Head','有标签语音','原生识别错误率 % ↓','边界 Precision %','边界 Recall %','F1@20ms % ↑'],test_rows)
    paragraph('测试仅在最终第 40 轮 checkpoint 上进行，未按测试结果选择 checkpoint。'
              'MFA 使用其中 158 条完整词典标签语音，其余三个头均评估全部 192 条。')
    if all_complete:
        paragraph('本次 seed 42 的最终结果：原始 OTTC ARPAbet 头的 PER 为 '
                  'CTC 10.26%、CE 9.91%、上游 jsd 9.68%；边界 F1 为 52.92%、63.72%、67.62%。'
                  'IF-MDD ARPAbet 头的识别以 CTC 最低（9.00%）；MFA 词典 IPA 目标的 '
                  'PER 以两 OTTC 组较低（CE 11.25%、jsd 11.21%，CTC 19.55%）；'
                  'ZIPA token error 以 CTC 最低（9.49%，CE 12.97%，jsd 19.91%）。'
                  '因此不同 head 的最佳 objective 不同；单 seed 不提供统计显著性结论。')
heading('Forced alignment 与 text-free inference 的评估区别')
paragraph('Text-free inference 只输入音频，参考 phones/timestamps 仅在预测完成后用于评分；'
          '识别报告 PER 或 ZIPA token error，分段报告自由预测边界。'
          'Forced alignment 提供有序参考 phone sequence，不提供真实时间戳；'
          '它评估已知内容条件下的对齐，而不是自由识别，因此不报告 forced PER。')
paragraph('统一 forced_k2：三组复用 IF-MDD 的 native-k2 FSA 实现：'
          'ctc_graph、DenseFsaVec、intersect_dense_pruned、shortest_path；'
          '保留原 beam retries，关闭 torchaudio/PyTorch 对齐回退。'
          'blank_policy=previous，blank 间隙归到前一个音素；时间戳按原实现 duration / frames 转换。'
          '原生 forced_native_ot：仅两个 OTTC 组，使用音频预测的冻结 frame masses，'
          '按参考 token 数的等质量目标做 inverse-CDF 对齐，保留分数帧边界。'
          '两种 FA 算法分开报告，禁止利用真实音素持续时间或拟合时间偏移。'
          '当前 FA 输入为各头的参考音素序列，不是只给单词文本再做 G2P 的评估。')
fa_rows=[];mode_runs=[];mode_results={}
for objective,label in OBJECTIVES.items():
    directory=ROOT/'evaluation_modes'/objective
    if not (directory/'EVAL_COMPLETE.json').exists():continue
    complete=json.loads((directory/'EVAL_COMPLETE.json').read_text())
    if complete['limit']:continue
    result=json.loads((directory/'metrics.json').read_text())
    if 'forced_k2' not in result['splits']['test']:continue
    mode_results[objective]=result
    for mode,heads in result['splits']['test'].items():
        if mode=='text_free':continue
        for h,name in zip(HEADS,NAMES):
            entry=heads[h];boundary=entry.get('boundaries',{}).get('20',{})
            fa_rows.append([label,mode,name,str(entry['supervised_utterances']),
                           f'{boundary["f1_percent"]:.2f}' if boundary else '无真实参考边界',
                           f'{entry["start_MAE_ms"]:.2f}' if 'start_MAE_ms' in entry else '—',
                           f'{entry["end_MAE_ms"]:.2f}' if 'end_MAE_ms' in entry else '—',
                           f'{entry["mean_token_IoU_percent"]:.2f}' if 'mean_token_IoU_percent' in entry else '—'])
    identity=directory/'wandb_run.json'
    if identity.exists():mode_runs.append([label,json.loads(identity.read_text()).get('url') or '离线待同步'])
snapshot['evaluation_modes']=mode_results
if fa_rows:
    heading('Forced alignment：最终 test 对齐结果')
    table(['Objective','FA decoder','Head','参考序列条数','F1@20ms % ↑','Start MAE ms ↓','End MAE ms ↓','Mean token IoU % ↑'],fa_rows)
    paragraph('K2 phone spans 使用 IF-MDD 的 previous-blank 规则，首尾覆盖语音；'
              'native OT spans 为 transport 连续分区。两者的分段算法及帧时间定义不同，'
              '应主要在同一 decoder 内比较三组。MAE/IoU 按参考序列位置逐 token 计算；'
              '两 IPA 头只导出预测 spans，没有真实 reference intervals，不报告边界准确率。')
    if mode_runs:table(['Objective','FA / text-free 评估 W&B'],mode_runs)
    paragraph('重放检查确认：显式不传 labels 的 forward 与传 labels 仅计算 loss 的 logits 完全一致；'
              '新评估的全部 text-free frame paths 与原 dev/test 预测逐帧一致。'
              '新指标存储于 evaluation_modes/{ctc,ce,jsd_upstream}/，单独 W&B evaluation run 保留原训练曲线。')
else:
    paragraph('新增 K2-FSA / text-free 双模式评估已准备；当前上表为完成的 text-free 结果。'
              'FA 指标将在评估成功后追加，不能把尚未计算的 FA 分数当作结果。')
heading('最终 dev 指标（40 epochs）' if all_complete else '最新 dev 识别和分段指标（中途快照）')
table(['Objective','Head','Dev epoch','原生识别错误率 % ↓','边界 F1@20ms % ↑'],metrics)
paragraph('两个 ARPAbet 头和 MFA 完整 phone token 报告各自标签上的 PER；'
          'MFA 指标以词典发音为目标，ZIPA 按 token error 计算，不是同粒度音素 PER。'
          '分段仅两个 ARPAbet 头有真实 PHN 时间参考；按 20ms argmax 变化（含 blank）、'
          '无拟合偏移、一对一 20ms 容差匹配。MFA/ZIPA 不伪造 token 时间边界。'
          '最终测试仅在全部 40 轮完成后进行，测试集不用于选 checkpoint。')
heading('并行执行记录' if all_complete else '并行作业与续训')
if all_complete:
    paragraph('CTC 在 3506874.opbs / mg0022 完成。CE、JSD 首段 '
              '3506924.opbs / mg0038、mg0039（12:57–13:23）成功保存断点；'
              '第二段 3506938.opbs / mg0007、mg0030（13:23–13:36）恢复至 40 轮，'
              '两个 debug 作业 Exit_status=0。每配置独立 GPU、world_size=1，'
              '1500s runtime budget 保留原完整 scheduler，恢复沿用同一 W&B ID。'
              '三组全部完成后，已取消未启动的常规兜底 3506940.opbs / 3506941.opbs，并释放交互节点。')
else:
    paragraph('CTC：3506874.opbs / mg0022。CE、JSD：3506924.opbs，'
              '分别 mg0038 / mg0039，独立 GPU、world_size=1。debug 最大 30min，'
              '1500s train-loop budget 后保存 optimizer/RNG，保持原 40-epoch scheduler。'
              '第二段 debug 3506938.opbs 自动续训；常规兜底 CE 3506940.opbs、JSD 3506941.opbs。'
              '已完成配置直接跳过，恢复沿用同一 W&B ID。')
heading('此前已完成的单头 fused-phoneme 实验')
paragraph('以下为此前单头模型的最终 40-epoch 测试结果，与本次四头实验采用不同训练及评估范围。'
          'README kaldi_test 为含 SA 的 240 条，core test 为 192 条。'
          '原仓库 jsd 的实际代价为 0.5 KL(m||p)+0.5 KL(m||qε)，不是标准 JSD；'
          '标准 JSD 为 0.5 KL(p||m)+0.5 KL(q||m)。')
historical=[]
for line in (REPRO/'RESULTS_SOFTDTW_ZH.md').read_text().splitlines():
    if not line.startswith('| ') or 'readme_kaldi_test_240' not in line:continue
    cells=[c.strip() for c in line.strip('|').split('|')]
    historical.append([cells[0],cells[1],cells[2],cells[4]])
table(['单头模型','测试集','Raw PER % ↓','边界 F1@20ms % ↑'],historical)
paragraph('单 seed，未做学习率/gamma 调参或置信区间。Soft-DTW CE gamma=0.1、'
          'normalize=False、lr=5e-5 在当前配置中出现严重删除，不能解释为调参后的最佳性能。'
          '分段指标不等同于论文未明确实现协议的 starting-frame F1/IDR。')
paragraph('历史 loss 曲线报告：https://wandb.ai/kevingenghaopeng-/ottc-timit-reproduction/reports/'
          'TIMIT-fused-phonemes:-loss下降与识别及分段对比--VmlldzoxODA3NDE2NA== 。'
          '五组历史曲线标记 backfill，保留原记录点，不插值；补录时间不是训练时间。')
heading('可追溯产物与下一步')
paragraph('仓库：/work/gm64/m64000/OTTC。四头配置：reproduction/multihead_timit/README.md；'
          '模型：reproduction/multihead_timit/models/{ctc,ce,jsd_upstream}/；'
          '历史报告：reproduction/RESULTS_JS_ZH.md、RESULTS_SOFTDTW_ZH.md、WANDB_ZH.md。'
          +('三组已保存 final checkpoint、dev/test metrics、每组 400 条 dev / 192 条 test 逐句预测及 W&B evaluation artifact；'
            '全部 12 个识别指标已从逐句预测重新计算核验，见 validation/final_results_audit.json。'
            '下一阶段可扩到 LibriSpeech 100h，目前尚未启动。' if all_complete else
            '训练结束保存 final checkpoint、dev/test metrics、逐句预测及 W&B evaluation artifact。'
            '下一步先完成并比较 TIMIT 三组，再扩到 LibriSpeech 100h。'))
if (REPRO/'english_suite/submission_status.json').exists():
    paragraph('后续 LibriSpeech 100h / 960h 与 MFA English 全套已安排，作业、资源和恢复协议见 '
              '[全套实验调度 README](../../english_suite/README.md)。正式训练是否开始以对应 PBS/W&B 状态为准。')
output=ROOT/'notion_update';output.mkdir(exist_ok=True)
markdown='\n'.join(lines)+'\n'
(output/'EXPERIMENT_UPDATE_ZH.md').write_text(markdown)
(output/'README.md').write_text(markdown.replace('## ', '# ', 1))
(output/'snapshot.json').write_text(json.dumps(snapshot,ensure_ascii=False,indent=2)+'\n')
(output/'notion_blocks.json').write_text(json.dumps({'children':blocks},ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'captured_at':now,'output':str(output),'notion_blocks':len(blocks),'published':False},ensure_ascii=False))
