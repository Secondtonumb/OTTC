"""Validate coverage, report paired changes, and publish evaluation summaries."""
from evaluate import HERE, ROOT, CONT, OBJECTIVES, VARIANTS, SPLITS, digest, save
from datetime import datetime, timezone
import json
import os
from pathlib import Path


def main():
    prepared = json.loads((HERE/'PREPARE_COMPLETE.json').read_text())
    results = {}
    common = {}
    for objective in OBJECTIVES:
        results[objective] = {}
        records = {}
        for variant in VARIANTS:
            directory = HERE/'evaluation'/objective/variant
            done = json.loads((directory/'EVAL_COMPLETE.json').read_text())
            assert not done['smoke'] and done['checkpoint_unchanged']
            assert done['metrics_sha256'] == digest(directory/'metrics.json')
            metrics = json.loads((directory/'metrics.json').read_text())
            checkpoint = Path(metrics['checkpoint'])
            assert digest(checkpoint/'model.safetensors') == metrics['checkpoint_sha256']
            complete = json.loads((CONT/'models'/objective/'TRAIN_COMPLETE.json').read_text())
            average = json.loads((CONT/'models'/objective/'AVERAGE_LAST5.json').read_text())
            if variant == 'libri100_epoch40' or (variant == 'best_dev' and metrics['identity']['stage_epoch'] == 0):
                expected_hash = complete['parent_sha256']
            elif variant == 'average_last5':
                expected_hash = average['weights_sha256']
            else:
                expected_hash = next(m['sha256'] for m in average['members']
                                     if m['stage_epoch'] == metrics['identity']['stage_epoch'])
            assert metrics['checkpoint_sha256'] == expected_hash
            assert metrics['label_independence_verified']
            assert not metrics['protocol']['optimizer_updated']
            assert not metrics['protocol']['training_on_timit']
            records[variant] = {}
            for split, (_, expected) in SPLITS.items():
                path = directory/(split+'_predictions.jsonl')
                assert done['prediction_sha256'][split] == digest(path)
                rows = [json.loads(line) for line in path.open()]
                assert len(rows) == len({r['utt'] for r in rows}) == expected
                assert metrics['manifest_sha256'][split] == prepared['splits'][split]['manifest_sha256']
                assert all(not r['reference_times_used_by_aligner'] for r in rows)
                records[variant][split] = {r['utt']: r for r in rows}
                failures = sum('failure' in r['forced_k2'] for r in rows)
                assert failures == metrics['splits'][split]['forced_k2']['alignment_failures']
                assert metrics['splits'][split]['text_free']['utterances'] == expected
            results[objective][variant] = metrics
        common[objective] = {}
        for split in SPLITS:
            baseline = records['libri100_epoch40'][split]
            assert all(set(records[v][split]) == set(baseline) for v in VARIANTS)
            valid = [utt for utt in baseline if all('failure' not in records[v][split][utt]['forced_k2'] for v in VARIANTS)]
            common[objective][split] = dict(utterances=len(valid), models={})
            for variant in VARIANTS:
                absolute = 0.0
                tokens = 0
                for utt in valid:
                    row = records[variant][split][utt]
                    reference = baseline[utt]['reference']
                    assert row['reference'] == reference
                    spans = row['forced_k2']['predicted_spans']
                    assert [s[0] for s in spans] == [s[0] for s in reference]
                    for (_, rs, re), (_, ps, pe) in zip(reference, spans):
                        absolute += abs(rs-ps)+abs(re-pe)
                        tokens += 1
                common[objective][split]['models'][variant] = dict(tokens=tokens,
                    boundary_MAE_ms=500*absolute/tokens if tokens else None)
    deltas = {}
    for objective in OBJECTIVES:
        deltas[objective] = {}
        for variant in VARIANTS:
            deltas[objective][variant] = {}
            for split in SPLITS:
                base = results[objective]['libri100_epoch40']['splits'][split]['text_free']
                current = results[objective][variant]['splits'][split]['text_free']
                bb, cb = base['boundaries']['20'], current['boundaries']['20']
                deltas[objective][variant][split] = dict(
                    PER_percentage_points=current['nosil_PER_percent']-base['nosil_PER_percent'],
                    relative_PER_reduction_percent=100*(base['nosil_PER_percent']-current['nosil_PER_percent'])/base['nosil_PER_percent'],
                    F1_percentage_points=cb['f1_percent']-bb['f1_percent'],
                    R_value_percentage_points=cb['r_value_percent']-bb['r_value_percent'])
    result = dict(created_at_UTC=datetime.now(timezone.utc).isoformat(), complete=True,
        interpretation='Paired zero-shot transfer before/after Libri960 continuation. Preserve Libri-selected best checkpoint; no TIMIT selection.',
        results=results, common_successful_FA=common, changes_from_parent=deltas,
        source_sha256={str(p): digest(p) for p in [HERE/'evaluate.py', Path(__file__),
            ROOT/'reproduction/libri100_singlehead_20261014/train.py', CONT/'evaluate.py',
            ROOT/'reproduction/segmentation_metrics.py',
            ROOT/'reproduction/english_suite/mfa_protocol_comparison/scorer_source/boundary_rvalue.py']})
    save(HERE/'RESULTS.json', result)
    primary = deltas['jsd_upstream']['best_dev']['core_test192']
    lines = ['# Libri100 → Libri960 加训：TIMIT 零样本评测', '',
        f"原上游 JSD 的既定 best-dev 在 core-test192 上，相对 Libri100 父模型：PER 变化 {primary['PER_percentage_points']:+.3f} 个百分点（相对错误率降低 {primary['relative_PER_reduction_percent']:.2f}%），无文本 F1 变化 {primary['F1_percentage_points']:+.3f} 个百分点，R-value 变化 {primary['R_value_percentage_points']:+.3f} 个百分点。", '',
        '全部复用现有单头 Wav2Vec2-large-LV60 权重，没有 TIMIT 训练或参数更新。三目标各比较父模型、Libri dev 选出的 best-dev、final 和新增五轮平均。TIMIT dev/test 均不重新选择 checkpoint。', '',
        'dev400 与 core-test192，不含 SA。人工 PHN 未取整参考与历史 core192 逐句一致；closure 作 sil、q 去除并保留时间间隙、sil 参考保留；识别去 sil/q 并将 dx→d。参考连续重复音素不合并。', '',
        '无文本分割使用音频 argmax 的全部类别变化及固定20ms帧时钟；唯一参考端点、排除句首句尾20ms、一对一匹配、冻结 IF-MDD R-value。给定音素 K2 FA 包含 sil、仅输入音素顺序，不输入参考时间；使用 duration/帧数 时钟及 blank previous。两种模式分别评分。', '',
        '下表 Δ 为相对同目标 Libri100 父模型的百分点差；PER 负值较好，F1/R 正值较好。单次实验，不据此声称统计显著性。', '']
    for objective in OBJECTIVES:
        lines += ['## '+objective+'：识别和无文本分割', '',
            '| Split | Variant | PER % ↓ | Δ PER | Precision % | Recall % | F1@20ms % ↑ | Δ F1 | R-value % ↑ | Δ R |',
            '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
        for split in SPLITS:
            base = results[objective]['libri100_epoch40']['splits'][split]['text_free']
            bb = base['boundaries']['20']
            for variant in VARIANTS:
                m = results[objective][variant]['splits'][split]['text_free']
                b = m['boundaries']['20']
                lines.append(f"| {split} | {variant} | {m['nosil_PER_percent']:.3f} | {m['nosil_PER_percent']-base['nosil_PER_percent']:+.3f} | {b['precision_percent']:.3f} | {b['recall_percent']:.3f} | {b['f1_percent']:.3f} | {b['f1_percent']-bb['f1_percent']:+.3f} | {b['r_value_percent']:.3f} | {b['r_value_percent']-bb['r_value_percent']:+.3f} |")
        lines += ['', '### 给定音素的 K2 强制对齐', '',
            '| Split | Variant | P % | Recall % | F1@20ms % | R-value % | FA failures | 共同成功子集 MAE ms ↓ |',
            '|---|---|---:|---:|---:|---:|---:|---:|']
        for split in SPLITS:
            for variant in VARIANTS:
                m = results[objective][variant]['splits'][split]['forced_k2']
                b = m['boundaries']['20']
                mae = common[objective][split]['models'][variant]['boundary_MAE_ms']
                lines.append(f"| {split} | {variant} | {b['precision_percent']:.3f} | {b['recall_percent']:.3f} | {b['f1_percent']:.3f} | {b['r_value_percent']:.3f} | {m['alignment_failures']} | {mae:.3f} |")
        lines += ['']
    lines += ['FA P/R/F1/R 覆盖全集，失败记零命中；MAE 在每目标四个变体的共同成功句集上计算，详见 RESULTS.json。10/20/30ms 分数、逐句预测、原始 OTTC evaluation loss、checkpoint 身份及哈希在各 metrics.json 中。不同目标的原始 loss 不直接比较。', '',
        '复现：`bash reproduction/libri960_timit_20261018/run.sh`（在 GPU allocation 内运行，完成结果校验后复用）。', '']
    (HERE/'REPORT_ZH.md').write_text('\n'.join(lines))
    save(HERE/'COMPARE_COMPLETE.json', dict(complete=True, objectives=list(OBJECTIVES), variants=VARIANTS,
        expected_utterances_per_variant={s: e for s, (_, e) in SPLITS.items()},
        validated_predictions=12*592, results_sha256=digest(HERE/'RESULTS.json'),
        checkpoint_selection='unchanged Libri development selection; no TIMIT model selection'))
    publish(result)


def publish(result):
    from reproduction.wandb_policy import init_evaluation, evaluation_uploads_enabled
    if not evaluation_uploads_enabled():
        return
    import wandb
    defaults = json.loads((ROOT/'reproduction/wandb_config.json').read_text())
    runs = {}
    for objective in OBJECTIVES:
        output = HERE/'wandb_runs'/objective
        output.mkdir(parents=True, exist_ok=True)
        identity = output/'wandb_run.json'
        options = dict(project=os.environ.get('WANDB_PROJECT', defaults['project']),
            entity=os.environ.get('WANDB_ENTITY', defaults['entity']),
            name='libri100-to960-TIMIT-zero-shot-'+objective, group='libri960_timit_20261018',
            job_type='evaluation', tags=['timit', 'zero-shot', 'libri960', objective, 'evaluation-only'],
            config=dict(objective=objective, protocol=result['results'][objective]['best_dev']['protocol'],
                checkpoint_identity={v: m['identity'] for v, m in result['results'][objective].items()},
                checkpoint_sha256={v: m['checkpoint_sha256'] for v, m in result['results'][objective].items()}),
            dir=str(output), mode=os.environ.get('WANDB_MODE', defaults.get('mode', 'online')),
            settings=wandb.Settings(init_timeout=45, disable_code=True))
        if identity.exists():
            options['id'] = json.loads(identity.read_text())['id']
            if options['mode'] == 'online':
                options['resume'] = 'allow'
        try:
            run = init_evaluation(**options)
        except wandb.errors.CommError:
            options['mode'] = 'offline'
            options.pop('resume', None)
            run = init_evaluation(**options)
        values = {}
        for variant, metrics in result['results'][objective].items():
            for split in SPLITS:
                for mode in ['text_free', 'forced_k2']:
                    m = metrics['splits'][split][mode]
                    for key in ['nosil_PER_percent', 'alignment_failures']:
                        if key in m:
                            values[f'{variant}/{split}/{mode}/{key}'] = m[key]
                    for key, value in m['boundaries']['20'].items():
                        values[f'{variant}/{split}/{mode}/{key}_20ms'] = value
                values[f'{variant}/{split}/raw_evaluation_loss'] = metrics['splits'][split]['raw_evaluation_loss']
                values[f'{variant}/{split}/forced_k2/common_subset_MAE_ms'] = result['common_successful_FA'][objective][split]['models'][variant]['boundary_MAE_ms']
        run.log(values); run.summary.update(values)
        artifact = wandb.Artifact('libri960-timit-'+objective, type='evaluation')
        for filename in ['REPORT_ZH.md', 'RESULTS.json', 'PREPARE_COMPLETE.json', 'COMPARE_COMPLETE.json', 'evaluate.py', 'summarize.py']:
            artifact.add_file(str(HERE/filename), name=filename)
        for variant in VARIANTS:
            for path in (HERE/'evaluation'/objective/variant).glob('*.json*'):
                artifact.add_file(str(path), name=variant+'/'+path.name)
        run.log_artifact(artifact)
        info = dict(id=run.id, url=run.url, mode=run.settings.mode)
        save(identity, info); runs[objective] = info
        run.finish()
        print('WANDB_COMPLETE', objective, info, flush=True)
    save(HERE/'WANDB_RUNS.json', runs)


if __name__ == '__main__':
    main()
