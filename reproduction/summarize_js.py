"""Report CE, upstream jsd and standard JS on the same audio-only tests."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    variants = [('CTC', 'ctc', 'crossentropy'), ('OTTC CE', 'ottc', 'crossentropy'),
                ('OTTC upstream jsd', 'ottc', 'jsd'), ('OTTC standard JS', 'ottc', 'jsd_standard')]
    rows, evidence = [], {}
    for label, loss, metric in variants:
        suffix = f'runexp_{loss}_fuse_phoneme'+('' if metric == 'crossentropy' else '_'+metric)
        target = ROOT/'reproduction/results'/suffix
        assert (target/'TRAIN_COMPLETE.json').exists() and (target/'EVAL_COMPLETE.json').exists()
        data = json.loads((target/'metrics.json').read_text())
        assert data['epoch'] == 40 and data['global_step'] == 2920
        checkpoint = Path(data['checkpoint'])
        config = json.loads((checkpoint.parent/'run_config.json').read_text())
        assert config['bins_metric'] == metric
        assert config['dataset'] == 'timit_fused_phoneme' and config['batch_size'] == 64
        assert config['num_train_epochs'] == 40 and config['oracle_weights_model'] is None
        assert config['learning_rate'] == (2e-4 if loss == 'ottc' else 5e-5)
        assert not any(data['checkpoint_loading_info'].get(k)
                       for k in ['missing_keys', 'unexpected_keys', 'mismatched_keys', 'error_msgs'])
        for split, result in data['splits'].items():
            records = [json.loads(line) for line in (target/(split+'.jsonl')).read_text().splitlines()]
            assert len(records) == result['utterances']
            assert len({r['wav'] for r in records}) == len(records)
            assert sum(r['raw_errors'] for r in records) == result['counts']['raw_errors']
            assert sum(r['normalized_errors'] for r in records) == result['counts']['nosil_errors']
            boundary = result['boundaries']['20']
            rows.append(f"| {label} | {split} | {100*result['raw_PER']:.2f} | "
                        f"{100*result['normalized_nosil_PER']:.2f} | {100*boundary['f1']:.2f} | "
                        f"{100*boundary['r_value']:.2f} | {100*result['matched_interval_overlap_ratio']:.2f} |")
        history = json.loads((checkpoint.parent/'trainer_state.json').read_text())['log_history']
        train = next(h for h in reversed(history) if 'train_runtime' in h)
        evidence[label] = dict(checkpoint=str(checkpoint), minutes=train['train_runtime']/60,
                               eval_wer=data['trainer_eval']['eval_wer'], bins_metric=metric)
        if metric != 'crossentropy':
            weights = checkpoint/'model.safetensors'
            sha = hashlib.sha256()
            with weights.open('rb') as handle:
                for block in iter(lambda: handle.read(8*1024*1024), b''):
                    sha.update(block)
            (ROOT/'reproduction/provenance'/f'{suffix}_checkpoint.json').write_text(json.dumps(
                dict(checkpoint=str(checkpoint), model_sha256=sha.hexdigest(), epoch=40, global_step=2920), indent=2))
    report = """# TIMIT fused-phoneme：CE 与 JS 代价对比

原 README 复现使用 CE（`bins_metric=crossentropy`）。本次追加两个独立实验，
均从相同官方 Wav2Vec2-large-LV60 底座、seed 42 开始，未从 CE 模型续训。
除音素 transport cost 外，TIMIT 数据、40 音素映射、regfrozen 2165 steps、
batch 64、lr 2e-4、40 epochs、warmup 500 和最终 checkpoint 选择均保持一致。

- `jsd`：保留原仓库实现。其实际代价为 0.5 KL(m || p) + 0.5 KL(m || qε)，
  m=(p+qε)/2；q 使用原代码 1e-12 平滑。这不是标准 JS，并且没有 log(2) 上界。
- `jsd_standard`：标准 0.5 KL(p || m) + 0.5 KL(q || m)，m=(p+q)/2。
  采用自然对数；不对 one-hot 目标平滑，0 log 0 按 0 处理。
  数值与 SciPy JS distance 的平方交叉核对，并验证梯度及极端 logits。

结果均为百分比；PER 越低越好，其余越高越好。分段采用原评估器协议：
音频独立推理、argmax 类别变化含 blank、无时间偏移、20ms 一对一边界匹配。
Overlap 为正确识别音素区间交集 / 全部参考时长；它与论文 starting-frame F1/IDR
不等价。补充 benchmark 只用于测试。CTC 的学习率沿用其 README 5e-5。

| 模型 | 测试集 | raw PER | no-sil PER | boundary F1@20ms | R-value | interval overlap |
|---|---|---:|---:|---:|---:|---:|
""" + '\n'.join(rows) + '\n\n'
    for label, d in evidence.items():
        report += f"- {label}：训练 {d['minutes']:.2f} 分钟，Trainer eval_wer {100*d['eval_wer']:.2f}%；checkpoint `{d['checkpoint']}`。\n"
    report += """
## 解读

在这次 seed 42 的实验中，原代码 `jsd` 在 README TIMIT 240 条上保持与 CE
相同的 raw PER（8.61%），边界 F1 提升 2.94 个百分点，区间 overlap 提升
2.40 个百分点；其边界 F1 在四个评估集上均高于 CE。
标准 JS 在相同超参数下的 TIMIT PER 为 15.53%，F1 为 61.53%，区间 overlap
为 70.14%，没有优于 CE。它收敛更慢；本实验没有调整学习率、loss 缩放或
训练时长，不能据此判断标准 JS 调参后的最佳性能。以上为单 seed 对比，
没有多 seed 置信区间；分段指标也不等同于论文 starting-frame F1/IDR。

统一 benchmark 图、收敛曲线和三个同语音分段对照图（PNG/PDF）位于
`reproduction/results/js_comparison/`。
"""
    report += "\nTrainer 的 eval_wer 是批量 padding 下的音素编辑率，与上述单音频 raw PER 分开保存。\n"
    report += "完整 metrics、逐音频 JSONL、后验图与 checkpoint 加载验证在各自 `reproduction/results/runexp_*` 目录。\n"
    (ROOT/'reproduction/RESULTS_JS_ZH.md').write_text(report)
    (ROOT/'reproduction/results/JS_COMPLETE.json').write_text(json.dumps(evidence, indent=2))
    print(report)


if __name__ == '__main__':
    main()
