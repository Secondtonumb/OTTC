"""Rescore original CTC/OTTC saved core192 outputs under the current protocol."""
import json
from pathlib import Path

from reproduction.gamma_pool_timit_20261022.evaluate import HERE, ROOT, score, report, sha, write
from reproduction.timit_ssl_native10_20261019.common import normalize


def main():
    target = HERE/'original_20ms_baselines'
    target.mkdir(exist_ok=True)
    common = [json.loads(l) for l in (HERE/'results_aggregation_comparison/jsd_upstream_test.predictions.jsonl').open()]
    refs = {r['utt']:r['reference'] for r in common}
    manifest = [json.loads(l) for l in (ROOT/'reproduction/multihead_timit/data/test.jsonl').open()]
    manifest = {r['utt']:r for r in manifest}
    results, provenance = {}, {}
    for label, suffix in [('CTC','runexp_ctc_fuse_phoneme'),
                          ('OTTC_CE','runexp_ottc_fuse_phoneme'),
                          ('OTTC_upstream_JSD','runexp_ottc_fuse_phoneme_jsd')]:
        source = ROOT/'reproduction/results'/suffix
        assert (source/'TRAIN_COMPLETE.json').exists() and (source/'EVAL_COMPLETE.json').exists()
        path = source/'timit_core_test_192.jsonl'
        saved = [json.loads(l) for l in path.open()]
        assert len(saved) == len(refs) == 192
        rows = []
        for row in saved:
            uid = Path(row['wav']).stem
            reference = normalize([p for p,_,_ in row['reference']])
            assert reference == refs[uid]
            assert Path(row['wav']).resolve() == Path(manifest[uid]['wav']).resolve()
            rows.append(dict(utt=uid, reference=reference,
                             hypotheses={'greedy':normalize(row['hypothesis'])}))
        assert {r['utt'] for r in rows} == set(refs)
        totals = score([r['reference'] for r in rows], [r['hypotheses']['greedy'] for r in rows])
        assert totals['reference_phones'] == 5754
        results[label] = totals
        report(target/f'{label}_test_greedy.wer.txt', rows, 'greedy', totals)
        with (target/f'{label}.predictions.jsonl').open('w') as stream:
            for row in rows: stream.write(json.dumps(row)+'\n')
        metrics = json.loads((source/'metrics.json').read_text())
        provenance[label] = dict(predictions=str(path), predictions_sha256=sha(path),
            metrics_sha256=sha(source/'metrics.json'), checkpoint=metrics['checkpoint'],
            epoch=metrics['epoch'], step=metrics['global_step'],
            checkpoint_provenance=str(ROOT/'reproduction/provenance'/f'{suffix}_checkpoint.json'),
            reference_and_audio_identity_match=True,
            replay='saved historical predictions; no new inference',
            selection='historical final epoch40; native20ms; different from native10ms best-dev models',
            scorer='SpeechBrain ErrorRateStats; same current no-sil normalization')
    write(target/'RESULTS.json', results)
    write(target/'PROVENANCE.json', provenance)
    write(target/'COMPLETE.json', dict(complete=True, utterances_per_model=192,
        original_baselines=3, new_inference=False, same_reference_and_audio=True,
        results_sha256=sha(target/'RESULTS.json')))
    lines = ['# 原始 20ms 单头模型：同协议重评分', '',
        '复用历史 final40epoch / step2920 的实际预测，未重新推理或训练。',
        '已逐句核对与当前 native10ms 试验的 core-test192 参考音素和音频路径一致；统一 SpeechBrain 原生打分。',
        '历史原模型保持 20ms，而新解码试验使用 native10ms / best-dev；两组不可当作只改变 loss 的消融。', '',
        '| 模型 | test PER % | S | D | I |', '|---|---:|---:|---:|---:|']
    for label, result in results.items():
        lines.append(f"| {label} | {result['PER_percent']:.2f} | {result['substitutions']} | {result['deletions']} | {result['insertions']} |")
    lines += ['', 'CTC 没有本模型定义下的 OTTC alpha/gamma 质量头，因此保留其标准 greedy；不向它强行套用 gamma 聚合。', '']
    (target/'RESULTS_ZH.md').write_text('\n'.join(lines))
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
