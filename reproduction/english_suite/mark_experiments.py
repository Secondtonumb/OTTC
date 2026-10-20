"""Mark actual training/evaluation evidence, separately from queued plans."""
import json
from datetime import datetime
from pathlib import Path

from track import live_progress

SUITE=Path(__file__).resolve().parent
REPRO=SUITE.parent


def read(path):
    return json.loads(path.read_text()) if path.exists() else {}


def main():
    rows=[]
    snapshot=read(SUITE/'tracking.json')
    for name,label in [('runexp_ctc_fuse_phoneme','CTC'),('runexp_ottc_fuse_phoneme','OTTC CE'),
                       ('runexp_ottc_fuse_phoneme_jsd','OTTC 上游 JSD'),
                       ('runexp_ottc_fuse_phoneme_jsd_standard','OTTC 标准 JSD'),
                       ('runexp_softdtw_fuse_phoneme','Soft-DTW CE')]:
        metrics=REPRO/'results'/name/'metrics.json'
        rows.append(['原始单头 TIMIT',label,'Wav2Vec2 LV60 / 20ms',
                     '✅ 已完成训练与 test' if metrics.exists() else '待核对',
                     '✅ README240、core192、Buckeye、L2-ARCTIC' if metrics.exists() else '—',
                     '历史系列未做 K2 / checkpoint 平均'])
    for dataset in ['timit','libri100','libri960']:
        for objective in ['ctc','ce','jsd_upstream']:
            directory=REPRO/('multihead_'+dataset)/'models'/objective
            complete=read(directory/'TRAIN_COMPLETE.json');live=live_progress(directory)
            step=live.get('step');epoch=live.get('last_train',{}).get('epoch')
            paused=read(directory/'CHECKPOINT_PAUSED.json')
            state=snapshot.get('production_runs',{}).get(dataset+'/'+objective,{}).get('training_state')
            if paused and state=='已保存，续训排队':epoch=paused['epochs']
            status=(f'✅ 已完成 {complete["epochs"]:.0f} 轮' if complete else
                    f'🟡 {state or "已跑，状态待核对"} {epoch:.2f}/40' if epoch is not None else '⏳ 尚未启动')
            test='✅ 最终 text-free test' if (directory/'test_metrics.json').exists() else '训练完成后 test'
            k2=REPRO/('multihead_'+dataset)/'evaluation_modes'/objective/'EVAL_COMPLETE.json'
            comparison=REPRO/'checkpoint_comparison'/(dataset+'_'+objective)/'results/COMPARE_COMPLETE.json'
            extras=('✅ K2 FA；' if k2.exists() else 'K2 FA 待测；')+('✅ 三个 checkpoint 比较' if comparison.exists() else 'checkpoint 比较未完成' if dataset!='libri960' else '—')
            if dataset=='libri100' and objective=='jsd_upstream' and (REPRO/'fabench_jsd_20261009/timit/full/EVAL_COMPLETE.json').exists():
                extras+='；✅ FA-bench clean TIMIT / processed Buckeye'
            rows.append([dataset+' 四头',{'ctc':'CTC','ce':'OTTC CE','jsd_upstream':'OTTC 上游 JSD'}[objective],
                         'Wav2Vec2 LV60 / 20ms',status,test,extras])
    directories={arm:SUITE/'mfa_models'/arm for arm in ['ctc','ottc','spottc','film_k4']}
    directories.update({arm:Path(record['directory']) for arm,record in read(SUITE/'additional_tracking_runs.json').items()})
    for arm,directory in directories.items():
        record=snapshot.get('production_runs',{}).get('mfa/'+arm,{})
        progress=read(directory/'progress.json')
        live=record.get('live',{}).get('last_update',{})
        complete=read(directory/'TRAIN_COMPLETE.json')
        step=live.get('step',progress.get('global_step'))
        epoch=live.get('epoch',progress.get('epoch'))
        status=('✅ 已完成 10 轮' if complete else
                f'🟡 {record.get("training_state","状态待核对")}；step {step}，第 {int(epoch)+1}/10 轮' if step is not None else
                '⏳ 正式训练尚未启动；仅 smoke')
        tests=list((SUITE/'mfa_evaluation'/arm).glob('step-*/EVAL_COMPLETE.json'))
        test='✅ 中途 dev4096 / test192 / K2 FA' if tests else '✅ 逐轮 dev；最终 test 待训练' if record.get('development') else '最终 test 待训练'
        compare=REPRO/'checkpoint_comparison'/('mfa_'+arm)/'COMPARE_COMPLETE.json'
        rows.append(['MFA 标签英语全量',arm,'Fbank→Linear+LayerNorm→Zapformer / 10ms',status,test,
                     '✅ 三个 checkpoint 比较' if compare.exists() else 'checkpoint 比较未完成' if arm!='film_k4' else '—'])
    now=datetime.now().astimezone().isoformat(timespec='seconds')
    lines=['# 已跑实验标记','',f'更新：{now}。','',
           '✅ 完成；🟡 已实际训练但尚未完成；⏳ 尚未开始正式训练。MFA 标签英语全量指 Zapformer 系列，四头里的 MFA IPA head 共享 Wav2Vec2，属于另一系列。','',
           'CE：已完成且数据/模型配置相同的实验直接复用权重，不重训；未完成的从断点续训。checkpoint 比较仅做推理、参数平均和评分，不执行训练更新。','',
           '| 实验系列 | 目标 | 前端 / 帧步长 | 训练状态 | 已有测试 | 补充评估 |',
           '|---|---|---|---|---|---|']
    lines+=['| '+' | '.join(row)+' |' for row in rows]
    snapshot=read(SUITE/'tracking.json');jobs=read(REPRO/'checkpoint_comparison/submissions.json')
    lines+=['','## 本次三个 checkpoint 比较','',
            '| 实验 | 状态 | Job |','|---|---|---|']
    for name,job in jobs.items():
        directory=REPRO/'checkpoint_comparison'/name
        marker=directory/('COMPARE_COMPLETE.json' if name.startswith('mfa_') else 'results/COMPARE_COMPLETE.json')
        pbs=snapshot.get('pbs',{}).get('三个 checkpoint 比较 '+name,{})
        state='✅ 完整比较已完成' if marker.exists() else '🟡 正在评估' if pbs.get('job_state')=='R' else '⏳ 排队' if pbs.get('job_state') in ['H','Q'] else '未完成，见最新 PBS'
        lines.append(f'| {name} | {state} | {job} |')
    lines+=['','[前端差异与来源](FRONTENDS.md) · [全部结果](RESULTS.md) · [checkpoint 平均结果](../checkpoint_comparison/README.md) · [PBS / W&B](TRACKING.md)。','']
    (SUITE/'EXPERIMENT_STATUS.md').write_text('\n'.join(lines))
    (SUITE/'experiment_status.json').write_text(json.dumps({'captured_at':now,'rows':rows},ensure_ascii=False,indent=2)+'\n')


if __name__=='__main__':main()
