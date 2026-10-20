"""Audit complete paired TIMIT evaluation and produce Markdown/CSV reports."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch


NAMES = {
    "k2": "K2",
    "k2_refined": "K2 + 局部质量细化",
    "ot_global": "全局 OT 连续插值",
    "ot_global_frame_left": "全局 OT crossing 帧左端",
}


def read(path):
    return json.loads(path.read_text())


def build(root, output):
    records, paired, timing, audits = [], [], [], []
    expected_ids = {}
    for model in ("jsd_upstream", "ce"):
        for split, expected_n in (("dev", 400), ("test", 192)):
            cache = root / model / split
            meta = read(cache / "index.json")
            metrics = read(root / model / f"eval_{split}" / "metrics.json")
            predictions = [
                json.loads(x)
                for x in (root / model / f"eval_{split}" / "predictions.jsonl")
                .read_text()
                .splitlines()
            ]
            manifest_rows = [
                json.loads(x) for x in Path(meta["manifest"]).read_text().splitlines()
            ]
            ids = [p["utt"] for p in predictions]
            assert len(set(ids)) == len(ids) == expected_n == metrics["utterances"]
            assert meta["subset_limit"] == 0 and metrics["subset_limit"] == 0
            assert (
                ids
                == [e["utt"] for e in meta["entries"]]
                == [r["utt"] for r in manifest_rows]
            )
            if split in expected_ids:
                assert ids == expected_ids[split]
            expected_ids[split] = ids
            assert (
                metrics["max_shift_ms"] == 20 and metrics["training_required"] is False
            )
            sums_before, sums_after, ns = [], [], []
            accepted = improved = worsened = unchanged = 0
            for row_index, p in enumerate(predictions):
                before, after, ref, global_times = (
                    np.asarray(p[k], dtype=float)
                    for k in ("coarse", "refined", "reference", "global_crossings")
                )
                gate = np.asarray(p["accepted"], dtype=bool)
                expected_reference = [
                    r[1]
                    for r in manifest_rows[row_index]["references"]["ottc_arpabet"][1:]
                ]
                assert np.allclose(ref, expected_reference, atol=1e-6, rtol=0)
                assert before.shape == after.shape == ref.shape == gate.shape
                assert np.isfinite(after).all() and (np.diff(after) >= 0).all()
                assert (np.abs(after - before) <= 0.020001).all()
                assert np.array_equal(after, np.where(gate, global_times, before))
                eb, ea = 1000 * abs(before - ref), 1000 * abs(after - ref)
                sums_before.append(eb.sum())
                sums_after.append(ea.sum())
                ns.append(len(ref))
                accepted += int(gate.sum())
                improved += int((ea < eb - 0.0001).sum())
                worsened += int((ea > eb + 0.0001).sum())
                unchanged += int((abs(ea - eb) <= 0.0001).sum())
            assert sum(ns) == metrics["internal_boundaries"]
            assert accepted == metrics["accepted_crossings"]
            for name, sums in (("k2", sums_before), ("k2_refined", sums_after)):
                assert abs(sum(sums) / sum(ns) - metrics[name]["mae_ms"]) < 1e-4
            # Paired utterance bootstrap of micro-average boundary MAE change.
            # This is not speaker-cluster uncertainty, and does not tune a model.
            rng = np.random.default_rng(20261013)
            samples = rng.integers(0, len(ns), size=(2000, len(ns)))
            delta_sum = np.asarray(sums_after) - sums_before
            boot = delta_sum[samples].sum(1) / np.asarray(ns)[samples].sum(1)
            lo, hi = np.quantile(boot, [0.025, 0.975])
            paired.append(
                dict(
                    model=model,
                    split=split,
                    utterances=expected_n,
                    boundaries=sum(ns),
                    mae_before_ms=metrics["k2"]["mae_ms"],
                    mae_after_ms=metrics["k2_refined"]["mae_ms"],
                    delta_mae_ms=float(delta_sum.sum() / sum(ns)),
                    bootstrap95_lo_ms=float(lo),
                    bootstrap95_hi_ms=float(hi),
                    accepted=accepted,
                    improved=improved,
                    worsened=worsened,
                    unchanged=unchanged,
                    f1_before=metrics["k2"]["boundaries"]["20"]["f1_pct"],
                    f1_after=metrics["k2_refined"]["boundaries"]["20"]["f1_pct"],
                )
            )
            for name in NAMES:
                for tolerance in ("5", "10", "20"):
                    score = metrics[name]["boundaries"][tolerance]
                    records.append(
                        dict(
                            model=model,
                            split=split,
                            utterances=expected_n,
                            decoder=name,
                            tolerance_ms=int(tolerance),
                            mae_ms=metrics[name]["mae_ms"],
                            p90_ms=metrics[name]["p90_ms"],
                            precision_pct=score["precision_pct"],
                            recall_pct=score["recall_pct"],
                            f1_pct=score["f1_pct"],
                            r_value_pct=score["r_value_pct"],
                        )
                    )
            # Export-time measurements are descriptive, not a fresh fair-speed
            # experiment against an optimized single-encoder K2 implementation.
            forward_ms, k2_ms, durations = [], [], []
            for entry in meta["entries"][
                1:
            ]:  # Exclude first item/startup for each process.
                item = torch.load(
                    cache / entry["file"], map_location="cpu", weights_only=True
                )
                forward_ms.append(item["model_forward_ms"])
                k2_ms.append(item["k2_ms"])
                durations.append(item["duration"])
            timing.append(
                dict(
                    model=model,
                    split=split,
                    device=meta["device_name"],
                    refiner_mean_ms=metrics["head_only_ms_per_utterance"],
                    refiner_median_ms=metrics["head_only_median_ms"],
                    refiner_p90_ms=metrics["head_only_p90_ms"],
                    prediction_plus_mass_forward_mean_ms=float(np.mean(forward_ms)),
                    k2_decoder_mean_ms=float(np.mean(k2_ms)),
                    timing_utterances=len(forward_ms),
                    mean_duration_s=float(np.mean(durations)),
                )
            )
            audits.append(
                dict(
                    model=model,
                    split=split,
                    utterances=expected_n,
                    failures=0,
                    checkpoint=meta["checkpoint"],
                    checkpoint_sha256=meta["checkpoint_sha256"],
                    manifest=meta["manifest"],
                    manifest_sha256=meta["manifest_sha256"],
                    scorer_sha256=metrics["scorer_sha256"],
                    implementation_sha256=meta["implementation_sha256"],
                )
            )
    assert not set(expected_ids["dev"]) & set(expected_ids["test"])
    output.mkdir(parents=True, exist_ok=True)
    summary = dict(
        root=str(root.resolve()),
        paired=paired,
        timing=timing,
        metrics=records,
        audit=audits,
        bootstrap="2000 paired utterance resamples, seed20261013; micro boundary MAE; not speaker clustered",
    )
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    for filename, rows in (
        ("metrics.csv", records),
        ("paired.csv", paired),
        ("timing.csv", timing),
    ):
        with (output / filename).open("w") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    lines = [
        "## 2026-10-13：免训练局部细化，全量 TIMIT 评测",
        "",
        "直接复用原始单头 OTTC 的 upstream JSD 与 CE checkpoint-2920；没有训练、更新权重或根据 test 调参。固定局部最大移动 20 ms，并由相邻粗边界中点进一步约束。",
        "",
        "**完整覆盖：每个模型 dev400 + core test192，共 1,184 次音频/模型推理；无失败或跳过，两个模型使用同一批样本。** 推理提供参考音素序列，不提供参考时间；这是 supplied-phone forced alignment，不是无转录 segmentation/PER。",
        "",
        "### K2 与免训练局部细化",
        "",
        "| 模型 | 集合 | 边界数 | K2 MAE ms | 细化 MAE ms | ΔMAE ms | K2 F1@20ms % | 细化 F1@20ms % | 接受 crossing % |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for p in paired:
        lines.append(
            f"| {p['model']} | {p['split']}{p['utterances']} | {p['boundaries']} | {p['mae_before_ms']:.3f} | {p['mae_after_ms']:.3f} | {p['delta_mae_ms']:+.3f} | {p['f1_before']:.2f} | {p['f1_after']:.2f} | {100*p['accepted']/p['boundaries']:.2f} |"
        )
    lines += [
        "",
        "ΔMAE＝细化−K2，负值为改善。所有边界都纳入评分，未只选择通过窗口检查的边界。",
        "",
        "### 全部读出对照，20 ms 容差",
        "",
        "| 模型 | 集合 | 读出 | MAE ms | P90 ms | P % | R % | F1 % | R-value % |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in records:
        if r["tolerance_ms"] == 20:
            lines.append(
                f"| {r['model']} | {r['split']} | {NAMES[r['decoder']]} | {r['mae_ms']:.3f} | {r['p90_ms']:.3f} | {r['precision_pct']:.2f} | {r['recall_pct']:.2f} | {r['f1_pct']:.2f} | {r['r_value_pct']:.2f} |"
            )
    lines += [
        "",
        "### 配对变化的辅助检查",
        "",
        "| 模型 | 集合 | ΔMAE 的 utterance-bootstrap 95% 区间 ms | 误差减小边界 | 误差增大边界 | 不变边界 |",
        "|---|---|---|---:|---:|---:|",
    ]
    for p in paired:
        lines.append(
            f"| {p['model']} | {p['split']} | [{p['bootstrap95_lo_ms']:+.3f}, {p['bootstrap95_hi_ms']:+.3f}] | {p['improved']} | {p['worsened']} | {p['unchanged']} |"
        )
    lines += [
        "",
        "2000 次按句配对重采样，seed=20261013，计算按边界数加权的 MAE 差。此区间没有按说话人聚类，不作为跨说话人显著性证明。",
        "",
        "### GPU 时间",
        "",
        "| 模型 | 集合 | 双分支前向均值 ms/句 | K2 解码均值 ms/句 | 局部细化均值 ms/句 | 局部细化 P90 ms/句 |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for t in timing:
        lines.append(
            f"| {t['model']} | {t['split']} | {t['prediction_plus_mass_forward_mean_ms']:.3f} | {t['k2_decoder_mean_ms']:.3f} | {t['refiner_mean_ms']:.3f} | {t['refiner_p90_ms']:.3f} |"
        )
    lines += [
        "",
        f"GPU：{timing[0]['device']}。前向包含 prediction encoder 与 regfrozen alignment encoder，导出阶段同步计时并剔除每个进程第一句；K2 单独计时。细化预热后计时，包含输入校验和 CUDA 同步，排除声学编码、K2、磁盘和传输。不同组件来自分阶段计时，不能直接相加冒充端到端实测；也不能忽略读取 alpha 所需额外冻结分支，声称相对单 encoder K2 仅增加细化耗时。",
        "",
        "### 协议与原始记录",
        "",
        "- 所有方法共用 duration / valid_frames 的时间轴；全局 OT 帧左端与连续插值使用同一质量 crossing。保留原40-phone映射（含sil/q/dx）及重复token位置。",
        "- MAE/P90 对内部 token 边界逐位置计算，包含静音相邻边界，排除句首句尾。P/R/F1 使用冻结 IF-MDD 无标签 directional matching，R-value 同样来自冻结评分器。这个协议与此前非静音起终点 MAE 或四头历史指标不同。",
        "- 局部候选若落在 K2 窗口外，严格回退到原 K2 边界；没有将远处 crossing 的小数部分搬到 K2 帧内，也没有用参考时间决定是否接受。分数帧时间来自均匀帧内质量假设，不代表新增亚帧声学观测。",
        "- 5/10/20 ms 的完整 P/R/F1/R-value 在 `summary/metrics.csv`；逐句 K2/细化/全局候选/接受掩码在各 `eval_*/predictions.jsonl`；原始 alpha、K2 segments 与音频时长在 cache。",
        f"- 结果根目录：`{root.resolve()}`。`summary/summary.json` 含完整统计、核验结果及 checkpoint/manifest/scorer/source 哈希；`provenance/` 保留实际执行代码。",
        "",
    ]
    (output / "report.md").write_text("\n".join(lines))
    print(json.dumps(dict(paired=paired, timing=timing), indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    build(args.root, args.output or args.root / "summary")
