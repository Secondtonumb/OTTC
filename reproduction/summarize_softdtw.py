"""Verify all Soft-DTW predictions and report matched baselines."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VARIANTS = [('Soft-DTW CE gamma=0.1', 'runexp_softdtw_fuse_phoneme'),
            ('CTC', 'runexp_ctc_fuse_phoneme'), ('OTTC CE', 'runexp_ottc_fuse_phoneme'),
            ('OTTC upstream jsd', 'runexp_ottc_fuse_phoneme_jsd'),
            ('OTTC standard JS', 'runexp_ottc_fuse_phoneme_jsd_standard')]


def main():
    evidence, rows = {}, []
    for label, suffix in VARIANTS:
        output = ROOT/'reproduction/results'/suffix
        assert (output/'TRAIN_COMPLETE.json').exists() and (output/'EVAL_COMPLETE.json').exists()
        data = json.loads((output/'metrics.json').read_text())
        assert data['epoch'] == 40 and data['global_step'] == 2920
        checkpoint = Path(data['checkpoint'])
        config = json.loads((checkpoint.parent/'run_config.json').read_text())
        assert config['dataset'] == 'timit_fused_phoneme' and config['batch_size'] == 64
        assert config['num_train_epochs'] == 40 and config['oracle_weights_model'] is None
        assert not any(data['checkpoint_loading_info'].get(k) for k in
                       ['missing_keys', 'unexpected_keys', 'mismatched_keys', 'error_msgs'])
        if suffix == 'runexp_softdtw_fuse_phoneme':
            assert config['loss_type'] == 'softdtw' and config['learning_rate'] == 5e-5
            assert data['softdtw'] == dict(gamma=.1, cost='ce', normalize=False)
            assert sum(m['utterances'] for m in data['splits'].values()) == 3172
        for split, result in data['splits'].items():
            records = [json.loads(line) for line in (output/(split+'.jsonl')).read_text().splitlines()]
            assert len(records) == result['utterances']
            assert len({r['wav'] for r in records}) == len(records)
            assert sum(r['raw_errors'] for r in records) == result['counts']['raw_errors']
            assert sum(r['normalized_errors'] for r in records) == result['counts']['nosil_errors']
            b = result['boundaries']['20']
            rows.append(f"| {label} | {split} | {100*result['raw_PER']:.2f} | "
                        f"{100*result['normalized_nosil_PER']:.2f} | {100*b['f1']:.2f} | "
                        f"{100*b['r_value']:.2f} | {100*result['matched_interval_overlap_ratio']:.2f} |")
        history = json.loads((checkpoint.parent/'trainer_state.json').read_text())['log_history']
        training = next(h for h in reversed(history) if 'train_runtime' in h)
        evidence[label] = dict(checkpoint=str(checkpoint), minutes=training['train_runtime']/60,
                               trainer_eval_wer=data['trainer_eval']['eval_wer'])
        if suffix == 'runexp_softdtw_fuse_phoneme':
            digest = hashlib.sha256()
            with (checkpoint/'model.safetensors').open('rb') as handle:
                for block in iter(lambda: handle.read(8*1024*1024), b''):
                    digest.update(block)
            (ROOT/'reproduction/provenance'/f'{suffix}_checkpoint.json').write_text(json.dumps(
                dict(checkpoint=str(checkpoint), model_sha256=digest.hexdigest(), epoch=40, global_step=2920), indent=2))
    report = """# TIMIT fused-phoneme：原仓库 Soft-DTW 复现

使用原仓库 `loss_type=softdtw`：CE 帧-音素代价、gamma=0.1、normalize=False，
保留原 CPU/Numba 动态规划与逐语音 loss 求和、batch 平均方式。
官方 Wav2Vec2-large-LV60 底座、seed 42、TIMIT train 4620 / kaldi_test 240
（含 SA）、官方 fused 映射 40 音素 / 42 输出、batch 64、40 epochs、
AdamW lr 5e-5 / wd .005、warmup 500 / linear scheduler、CNN 冻结。
学习率沿用已复现的 README CTC 基线；README 未列出独立的 TIMIT Soft-DTW
完整命令，因此它是原实现基线的复现，不是论文某个数值表格的逐项复现。
Soft-DTW 使用单个 linear 音素 head，不使用 OTTC 时间权重 head 或 regfrozen。
最终 epoch 40 / step 2920 用于测试，未按测试指标挑选 checkpoint。

必要兼容修复：将 Soft-DTW 接入已有 CTC 识别指标；克隆 labels 避免原地
改写标注；按每条实际帧数截断，避免 batch padding 掩码长度不一致。
Soft-DTW 动态规划和 CE 代价源码未改；已用独立 PyTorch DP 验证 loss 和梯度。

下表均为百分比。所有模型都只输入音频；分段为 argmax 类别变化（含 blank），
20ms 时间步，无拟合偏移，20ms 容忍的一对一边界匹配。
Overlap 为正确识别音素区间交集 / 全部参考时长。
这些分段指标与论文 starting-frame F1/IDR 不等价；结果来自单 seed，未做调参。
Buckeye 和 L2-ARCTIC 只用于补充测试。

| 模型 | 测试集 | raw PER ↓ | no-sil PER ↓ | boundary F1@20ms ↑ | R-value ↑ | interval overlap ↑ |
|---|---|---:|---:|---:|---:|---:|
""" + '\n'.join(rows) + '\n\n'
    for label, d in evidence.items():
        report += f"- {label}：训练 {d['minutes']:.2f} 分钟，Trainer eval_wer {100*d['trainer_eval_wer']:.2f}%；checkpoint `{d['checkpoint']}`。\n"
    soft = ROOT/'reproduction/results/runexp_softdtw_fuse_phoneme'
    detail = json.loads((soft/'metrics.json').read_text())['splits']['readme_kaldi_test_240']
    counts = detail['counts']
    report += f"""
## 结果解读与运行核对

该配置出现严重音素漏识别和长区间合并。README TIMIT 240 条参考共
{counts['raw_phones']} 个音素，编辑对齐得到删除 {counts['raw_deletions']}、
替换 {counts['raw_substitutions']}、插入 {counts['raw_insertions']}。
后验与分段对照图也显示部分长语音区间被分配给单一音素。
因此本次既没有达到 CTC/OTTC 的识别水平，也没有得到有用的分段效果。
这是原实现 gamma=0.1、lr=5e-5、seed 42 的结果；尚未搜索 gamma、学习率
或训练时长，不能据此判断 Soft-DTW 调参后的最佳表现。

TIMIT train 原始 PHN 核对为 4620 条、约 3.94 小时、平均 3.07 秒。
batch 64 对应每轮 73 步、40 轮 2920 步；GH200 上训练实测 17.35 分钟。
训练是官方预训练底座上的微调。最终 checkpoint 的 CNN 权重保持不变，
Transformer 第 1 层注意力和第 24 层前馈权重均已更新，核对记录见
`results/runexp_softdtw_fuse_phoneme/weight_update_check.json`。
音素输出分布见同目录 `phone_output_distribution.json`。
运行于 interactive 作业 3506088.opbs / mg0033，完成后已释放 GPU。
"""
    report += "\nTrainer 的 eval_wer 是批量 padding 下的音素编辑率，与单音频 raw PER 分开保存。\n"
    report += "完整逐语音预测、metrics 和后验图在 `results/runexp_softdtw_fuse_phoneme/`；统一 benchmark 图、收敛曲线和分段对照在 `results/softdtw_comparison/`。\n"
    (ROOT/'reproduction/RESULTS_SOFTDTW_ZH.md').write_text(report)
    (ROOT/'reproduction/results/SOFTDTW_COMPLETE.json').write_text(json.dumps(evidence, indent=2))
    print(report)


if __name__ == '__main__':
    main()
