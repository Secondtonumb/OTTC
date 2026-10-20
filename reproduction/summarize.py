"""Write the final reproduction report only after both 40-epoch runs finish."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    result_root = ROOT/'reproduction/results'
    rows = []
    evidence = {}
    for loss in ['ottc', 'ctc']:
        suffix = f'runexp_{loss}_fuse_phoneme'
        target = result_root/suffix
        assert (target/'TRAIN_COMPLETE.json').exists()
        assert (target/'EVAL_COMPLETE.json').exists()
        metrics = json.loads((target/'metrics.json').read_text())
        assert metrics['epoch'] == 40
        model_dir = Path(metrics['checkpoint']).parent
        config = json.loads((model_dir/'run_config.json').read_text())
        assert config['batch_size'] == 64 and config['num_train_epochs'] == 40
        assert config['learning_rate'] == (2e-4 if loss == 'ottc' else 5e-5)
        assert config['dataset'] == 'timit_fused_phoneme'
        assert config['oracle_weights_model'] is None
        evidence[loss] = metrics
        for split, m in metrics['splits'].items():
            b = m['boundaries']['20']
            rows.append(f"| {loss.upper()} | {split} | {m['utterances']} | "
                        f"{100*m['raw_PER']:.2f} | {100*m['normalized_nosil_PER']:.2f} | "
                        f"{100*b['precision']:.2f} | {100*b['recall']:.2f} | "
                        f"{100*b['f1']:.2f} | {100*b['r_value']:.2f} | "
                        f"{100*m['matched_interval_overlap_ratio']:.2f} |")
    text = """# 官方 README：TIMIT fused-phoneme 复现结果

CTC 和 OTTC 都完成 40 epochs，使用官方 Wav2Vec2-large-LV60 底座和
README 超参数，未经自有 checkpoint 初始化。训练数据包含 SA，train 4,620，
kaldi_test 240；音素映射保持官方代码（含 q 的 40 类，加 pad/unk 为 42）。
最终 epoch-40 checkpoint 用于全部结果，没有按测试指标挑选 checkpoint。

下表均为百分比；PER 越低越好，边界指标与 overlap 越高越好。
识别和分段推理只提供音频。Boundary 指标采用 20ms 容忍、无拟合时间偏移、
一对一匹配；预测边界为 argmax 类别变化（包括 blank），参考为标注区间
全部端点并排除语音两端 20ms。Overlap 分子来自编辑对齐中正确识别音素
的区间交集，分母包括全部参考音素的时长。这些是明确协议下的补充分段
评估，不能冒称复现论文未完整公开的 starting-frame F1/IDR 数字。

| 模型 | 测试集 | 语音数 | raw PER | normalized no-sil PER | boundary P | boundary R | boundary F1 | R-value | overlap |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
""" + '\n'.join(rows) + '\n\n'
    for loss, d in evidence.items():
        state = json.loads((Path(d['checkpoint']).parent/'trainer_state.json').read_text())
        text += f"- {loss.upper()} checkpoint：`{d['checkpoint']}`，global_step={d['global_step']}。\n"
        text += f"  官方 Trainer eval_wer={100*d['trainer_eval']['eval_wer']:.2f}%；"
        text += "单音频 raw PER 单独列出，避免 padding 解码差异。\n"
        history = state['log_history']
        train = next((h for h in reversed(history) if 'train_runtime' in h), {})
        if train:
            text += f"  训练实测 {train['train_runtime']/60:.2f} 分钟（Trainer runtime，含逐 epoch 评估/保存）。\n"
    text += "\n环境、必要兼容修复、启动命令与 metric contract 见 `reproduction/README.md`；"
    text += "底座来源见 `reproduction/provenance/upstream.json`；"
    text += "每组 `results/runexp_*_fuse_phoneme/` 内有 runtime/source hashes、"
    text += "完整逐语音 JSONL、10/20/30ms 指标、编辑计数和完成标记。\n"
    (ROOT/'reproduction/RESULTS_ZH.md').write_text(text)
    (result_root/'COMPLETE.json').write_text(json.dumps(dict(models={
        k: v['checkpoint'] for k, v in evidence.items()}), indent=2))
    print(text)


if __name__ == '__main__':
    main()
