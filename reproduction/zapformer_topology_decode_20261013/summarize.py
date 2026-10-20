"""Render the completed, fixed-checkpoint decoder comparison."""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main():
    assert (HERE / 'COMPLETE.json').exists(), 'Evaluation is not complete'
    results = json.loads((HERE / 'results.json').read_text())
    assert len(results) == 16
    lines = ['四组 Zapformer / Libri100：固定 epoch50 checkpoint 的 S3T2** 解码复评', '',
             '复用各组 step5300 的 model 与 model_avg；未训练、未改原 checkpoint 或训练历史。',
             '音频推理不输入参考音素或时间戳。图直接使用冻结 ifmdd_losses.py 的原始音素 topology，构建自由音素循环。',
             'k2.intersect_dense 使用精确前后向 output pruning 后 shortest_path，不使用近似 search beam；无语言模型、无音素插入惩罚。',
             '每条最佳路径核对原始图的 src/dst 连续性、每帧覆盖、入口 S0、末尾返回音素边界以及 blank 仅在边界。',
             '同一 BF16 声学推理的 greedy-completion 预测逐句重放历史记录，全部一致才接受对比。',
             '评分保持同一 speech-only 参考、IPA→English-collapsed ARPAbet/TIMIT39 映射及 jiwer S/D/I，PER=100×Σ(S+D+I)/Σ参考音素数。',
             '这是 TIMIT dev400/core-test192 监控评测，不能当作 LibriSpeech WER。test 不用于选择 checkpoint 或解码超参数。', '',
             '实际解码图与参考对照：', '',
             '- [b 分支 FSA](fsa_b_exact.svg)：图中红色入口输出音素，自环与其他边输出 eps。',
             '- [完整104音素 FSA](fsa_full_exact.svg)，[实际图文本](phone_loop.fst.txt)。',
             '- [开头120ms实际路径](fsa_actual_b_path.svg)。',
             '- [逐句参考/解码/编辑对齐](EXAMPLES.md)，[Bricks句时间对照](si1279_reference_decode.png)，[Fat句时间对照](si1909_reference_decode.png)。',
             '- IFMDD_GRAPH_EQUIVALENCE_PASS.json 记录104分支与当前IF-MDD原始 _add_phone_arcs 的逐边一致性；SELF_LOOP_COLLAPSE_AUDIT.json 记录全部4736句独立折叠与FSA输出一致。',
             '- b0 b0 b1 b1 b2 b2 只输出一个 b；b0 b2 b0 b2 输出两个 b。这些结果中的重复是完整音素路径再次进入，不能通过把所有同音素合并来冒称原始解码结果改善。', '']
    for variant in ['final', 'cumulative_average']:
        lines.extend([variant, '',
            '| arm | split | greedy PER% | topology PER% | insertion% | deletion% | substitution% | argmax改动帧% |',
            '| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |'])
        for r in results:
            if r['checkpoint_variant'] != variant:
                continue
            m, g = r['mapped_arpabet'], r['greedy_mapped_arpabet']
            lines.append(f"| {r['arm']} | {r['split']} | {g['PER_percent']:.2f} | {m['PER_percent']:.2f} | {m['insertions_percent']:.2f} | {m['deletions_percent']:.2f} | {m['substitutions_percent']:.2f} | {r['raw_argmax_changed_frame_percent']:.2f} |")
        lines.extend([''])
    lines.extend(['Text-free topology 路径边界指标（与 supplied-phone FA 分开）', '',
        '沿用冻结评分代码的 10ms grid、包含 blank 的音素变化边界、排除同音素子状态变化及原边界匹配；R-value 使用冻结 IF-MDD 公式。',
        '这里不复评 FA；旧 FA 图约束指标不与本表混用。边界均为全套 dev/test，未计算 equal-phone-count TSE 子集。', '',
        '| arm | checkpoint | split | P@20ms% | Recall@20ms% | F1@20ms% | R-value@20ms% |',
        '| --- | --- | --- | ---: | ---: | ---: | ---: |'])
    for r in results:
        m = r['text_free_topology']
        lines.append(f"| {r['arm']} | {r['checkpoint_variant']} | {r['split']} | {m['boundary_precision_20ms_percent']:.2f} | {m['boundary_recall_20ms_percent']:.2f} | {m['boundary_F1_20ms_percent']:.2f} | {m['boundary_R_value_20ms_percent']:.2f} |")
    lines.extend(['', '每句 raw argmax、合法路径与两种预测在各 arm 的 *_predictions.jsonl；原始计数在 results.json。',
                  '图构造、声学推理、映射、评分代码和 dev/test 数据 SHA256 在 protocol.json；历史IF-MDD raw-PHN与R-value源码hash另见scorer_provenance.json；checkpoint SHA256 在各 arm/COMPLETE.json。',
                  '新结果属于 checkpoint 固定的 evaluation，保留 epoch50/step5300；没有把重评时间当训练时间，也没有插值。', ''])
    (HERE / 'REPORT.md').write_text('\n'.join(lines))
    for r in results:
        print(r['arm'], r['checkpoint_variant'], r['split'],
              'greedy', round(r['greedy_mapped_arpabet']['PER_percent'], 2),
              'topology', round(r['mapped_arpabet']['PER_percent'], 2))


if __name__ == '__main__':
    main()
