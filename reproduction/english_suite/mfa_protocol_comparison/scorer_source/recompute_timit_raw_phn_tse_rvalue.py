#!/usr/bin/env python3
"""Recompute TIMIT phone TSE/ACC and boundary R-value with raw PHN references.

This script intentionally avoids the rounded Kaldi CTMs under
``data/timit_proccessed`` and rebuilds references from the original TIMIT
``.phn`` sample boundaries.  Hypotheses are read from IF-MDD alignment JSONL
files.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from boundary_rvalue import finalize_counts, score_events


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TIMIT_PHN_ROOT = Path("/work/gm64/m64000/dataset/TIMIT/timit")
DEFAULT_OUTPUT_DIR = REPO_ROOT / "fa_research/results/timit_raw_phn_tse_rvalue"
DEFAULT_SOURCES = [
    (
        "context",
        REPO_ROOT / "fa_research/results/cross_domain_json_fa_k2/alignments.jsonl",
    ),
    (
        "topology",
        REPO_ROOT
        / "fa_research/results/librispeech_topology_full_ft_fa_interactive/timit_dev_timit_test/alignments.jsonl",
    ),
    (
        "jtli",
        REPO_ROOT / "fa_research/results/jtli_timestamp_output_20260605_timit_dx_to_t/alignments.jsonl",
    ),
]
DATASET_UTT_FILES = {
    "timit_dev": REPO_ROOT / "data/timit_proccessed/dev_clean/utt2spk",
    "timit_test": REPO_ROOT / "data/timit_proccessed/test_clean/utt2spk",
}
SILENCE = {"", "sil", "<sil>", "!sil", "sp", "<sp>", "spn", "pau", "h#", "epi"}
DROP_FOR_SCORING = {"q"}
CLOSURE_TO_COMPATIBLE_NEXT = {
    "bcl": {"b"},
    "dcl": {"d", "jh"},
    "gcl": {"g"},
    "pcl": {"p"},
    "tcl": {"t", "ch"},
    "kcl": {"k"},
}
TIMIT_TO_SCORING_PHONE = {
    "h#": "sil",
    "pau": "sil",
    "epi": "sil",
    "bcl": "sil",
    "dcl": "sil",
    "gcl": "sil",
    "pcl": "sil",
    "tcl": "sil",
    "kcl": "sil",
    "ax": "ah",
    "ax-h": "ah",
    "axh": "ah",
    "axr": "er",
    "el": "l",
    "em": "m",
    "en": "n",
    "eng": "ng",
    "hv": "hh",
    "ix": "ih",
    "nx": "n",
    "ux": "uw",
    "zh": "sh",
    "j": "jh",
    "ao": "aa",
}
TOLERANCES_MS = (10, 20, 25, 30, 40, 50)


def resolve_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else REPO_ROOT / path


def parse_source(value: str) -> tuple[str, Path]:
    if ":" not in value:
        raise argparse.ArgumentTypeError("source must be FAMILY:PATH")
    family, path = value.split(":", 1)
    return family, resolve_path(path)


def load_source_tsv(path: Path, *, family_field: str, path_field: str) -> list[tuple[str, Path]]:
    path = resolve_path(path)
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        if reader.fieldnames is None:
            raise ValueError(f"empty source TSV: {path}")
        missing = [field for field in (family_field, path_field) if field not in reader.fieldnames]
        if missing:
            raise ValueError(f"{path} missing required column(s): {', '.join(missing)}")
        sources = []
        seen: set[tuple[str, Path]] = set()
        for row in reader:
            family = str(row.get(family_field) or "").strip()
            source_path = str(row.get(path_field) or "").strip()
            if not family or not source_path:
                continue
            item = (family, resolve_path(source_path))
            if item in seen:
                continue
            sources.append(item)
            seen.add(item)
    return sources


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._=-]+", "_", value)


def normalize_phone(phone: str, *, dx_map: str) -> str | None:
    phone = str(phone).strip().lower()
    phone = re.sub(r"_\d+$", "", phone)
    phone = re.sub(r"\d+$", "", phone)
    if phone in DROP_FOR_SCORING:
        return None
    if phone == "dx":
        return dx_map
    return TIMIT_TO_SCORING_PHONE.get(phone, phone)


def raw_phn_rows(
    items: list[tuple[int, int, str]],
    *,
    sample_rate: float,
    dx_map: str,
    closure_mode: str,
) -> list[tuple[str, float, float]]:
    if closure_mode == "silence":
        rows: list[tuple[str, float, float]] = []
        for start_i, end_i, phone in items:
            label = normalize_phone(phone, dx_map=dx_map)
            if label is None:
                continue
            rows.append((label, float(start_i) / sample_rate, float(end_i) / sample_rate))
        return rows

    if closure_mode != "merge_stop":
        raise ValueError(f"Unsupported TIMIT closure mode: {closure_mode}")

    rows = []
    idx = 0
    while idx < len(items):
        start_i, end_i, phone = items[idx]
        phone = phone.strip().lower()
        if phone in DROP_FOR_SCORING:
            idx += 1
            continue
        if phone in CLOSURE_TO_COMPATIBLE_NEXT:
            if idx + 1 < len(items):
                next_start_i, next_end_i, next_phone = items[idx + 1]
                next_label = normalize_phone(next_phone, dx_map=dx_map)
                if next_label in CLOSURE_TO_COMPATIBLE_NEXT[phone]:
                    rows.append(
                        (
                            next_label,
                            float(start_i) / sample_rate,
                            float(next_end_i) / sample_rate,
                        )
                    )
                    idx += 2
                    continue
            rows.append(("sil", float(start_i) / sample_rate, float(end_i) / sample_rate))
            idx += 1
            continue

        label = normalize_phone(phone, dx_map=dx_map)
        if label is not None:
            rows.append((label, float(start_i) / sample_rate, float(end_i) / sample_rate))
        idx += 1
    return rows


def is_silence(phone: str | None) -> bool:
    return phone is None or str(phone).lower() in SILENCE


def round_time(value: float, step_ms: float) -> float:
    if step_ms <= 0.0:
        return float(value)
    step_s = step_ms / 1000.0
    return round(float(value) / step_s) * step_s


def build_phn_index(root: Path) -> dict[tuple[str, str], Path]:
    root = resolve_path(root)
    if not root.exists():
        raise FileNotFoundError(f"TIMIT PHN root does not exist: {root}")
    index: dict[tuple[str, str], Path] = {}
    for path in root.rglob("*.phn"):
        index[(path.parent.name.upper(), path.stem.upper())] = path
    return index


def load_dataset_utts(datasets: Iterable[str]) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for dataset in datasets:
        path = DATASET_UTT_FILES.get(dataset)
        if path is None or not path.is_file():
            out[dataset] = []
            continue
        utts = []
        for line in path.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if parts:
                utts.append(parts[0])
        out[dataset] = utts
    return out


def phn_path_for_utt(index: dict[tuple[str, str], Path], utt: str) -> Path | None:
    if "_" not in utt:
        return None
    speaker, sent = utt.split("_", 1)
    return index.get((speaker.upper(), sent.upper()))


def load_raw_phn_ref(
    index: dict[tuple[str, str], Path],
    utt: str,
    *,
    sample_rate: float,
    dx_map: str,
    closure_mode: str,
) -> list[tuple[str, float, float]] | None:
    path = phn_path_for_utt(index, utt)
    if path is None:
        return None
    items: list[tuple[int, int, str]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) != 3:
            continue
        start_i, end_i, phone = parts
        items.append((int(start_i), int(end_i), phone))
    return raw_phn_rows(
        items,
        sample_rate=sample_rate,
        dx_map=dx_map,
        closure_mode=closure_mode,
    )


def write_ref_ctms(
    output_dir: Path,
    dataset_utts: dict[str, list[str]],
    phn_index: dict[tuple[str, str], Path],
    *,
    sample_rate: float,
    dx_map: str,
    closure_mode: str,
    decimals: int,
) -> dict[str, Path]:
    ref_dir = output_dir / "ctm/ref_raw_phn"
    ref_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for dataset, utts in dataset_utts.items():
        path = ref_dir / f"{dataset}.phone.raw_phn.ctm"
        with path.open("w", encoding="utf-8") as f:
            for utt in utts:
                rows = load_raw_phn_ref(
                    phn_index,
                    utt,
                    sample_rate=sample_rate,
                    dx_map=dx_map,
                    closure_mode=closure_mode,
                )
                if not rows:
                    continue
                for label, start, end in rows:
                    f.write(
                        f"{utt} 1 {start:.{decimals}f} "
                        f"{max(0.0, end - start):.{decimals}f} {label}\n"
                    )
        paths[dataset] = path
    return paths


def json_records(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if line.strip():
                yield line_no, json.loads(line)


def record_model(record: dict[str, Any]) -> str:
    return str(record.get("model") or record.get("package") or "unknown")


def record_utt(record: dict[str, Any]) -> str:
    return str(record.get("utt") or Path(str(record.get("wav", ""))).stem)


def record_head(record: dict[str, Any]) -> str:
    head = record.get("alignment_head") or record.get("head")
    if head:
        return str(head)
    alignment_type = str(record.get("alignment_type") or "")
    for prefix in ("content", "topology", "fused"):
        if alignment_type.startswith(f"{prefix}_"):
            return prefix
    return ""


def record_target(record: dict[str, Any]) -> str:
    return str(record.get("target") or record.get("alignment_type") or "phonemes")


def prediction_rows(
    record: dict[str, Any],
    *,
    dx_map: str,
    hyp_round_step_ms: float,
) -> list[tuple[str, float, float]]:
    labels = record.get("pred_phones") or record.get("phones") or record.get("display_phones") or []
    starts = record.get("pred_starts") or []
    ends = record.get("pred_ends") or []
    rows: list[tuple[str, float, float]] = []
    for label, start, end in zip(labels, starts, ends):
        mapped = normalize_phone(str(label), dx_map=dx_map)
        if mapped is None:
            continue
        rows.append(
            (
                mapped,
                round_time(float(start), hyp_round_step_ms),
                round_time(float(end), hyp_round_step_ms),
            )
        )
    return rows


def filter_silence(rows: list[tuple[str, float, float]]) -> list[tuple[str, float, float]]:
    return [row for row in rows if not is_silence(row[0])]


def boundary_events(
    rows: list[tuple[str, float, float]],
    *,
    filter_sil: bool,
) -> list[tuple[float, frozenset[str]]]:
    if filter_sil:
        rows = filter_silence(rows)
    events: list[tuple[float, frozenset[str]]] = []
    for idx in range(len(rows) - 1):
        events.append((float(rows[idx][2]), frozenset({rows[idx][0], rows[idx + 1][0]})))
    return events


def greedy_boundary_matches(
    pred: list[tuple[float, frozenset[str]]],
    ref: list[tuple[float, frozenset[str]]],
    *,
    tolerance_s: float,
    label_aware: bool,
) -> int:
    candidates: list[tuple[float, int, int]] = []
    for pred_idx, (pred_time, pred_labels) in enumerate(pred):
        for ref_idx, (ref_time, ref_labels) in enumerate(ref):
            distance = abs(pred_time - ref_time)
            if distance > tolerance_s:
                continue
            if label_aware and not pred_labels.intersection(ref_labels):
                continue
            candidates.append((distance, pred_idx, ref_idx))
    candidates.sort()
    used_pred: set[int] = set()
    used_ref: set[int] = set()
    matches = 0
    for _, pred_idx, ref_idx in candidates:
        if pred_idx in used_pred or ref_idx in used_ref:
            continue
        used_pred.add(pred_idx)
        used_ref.add(ref_idx)
        matches += 1
    return matches


def safe_div(num: float, den: float) -> float:
    return num / den if den else 0.0


def active_label(rows: list[tuple[str, float, float]], midpoint: float) -> str | None:
    for label, start, end in rows:
        if start <= midpoint < end:
            return label
    if rows and abs(midpoint - rows[-1][2]) < 1e-9:
        return rows[-1][0]
    return None


def time_overlap(
    pred_rows: list[tuple[str, float, float]],
    ref_rows: list[tuple[str, float, float]],
    *,
    filter_sil: bool,
) -> tuple[float, float]:
    pred_rows = [row for row in pred_rows if row[2] > row[1]]
    ref_rows = [row for row in ref_rows if row[2] > row[1]]
    if filter_sil:
        pred_rows = filter_silence(pred_rows)
        ref_rows = filter_silence(ref_rows)
    if not pred_rows or not ref_rows:
        return 0.0, 0.0
    cuts = sorted({time for _label, start, end in pred_rows + ref_rows for time in (start, end)})
    same = 0.0
    total = 0.0
    for left, right in zip(cuts, cuts[1:]):
        if right <= left:
            continue
        midpoint = (left + right) / 2.0
        ref_label = active_label(ref_rows, midpoint)
        if ref_label is None:
            continue
        pred_label = active_label(pred_rows, midpoint)
        total += right - left
        if pred_label == ref_label:
            same += right - left
    return same, total


def new_accumulator() -> dict[str, Any]:
    acc: dict[str, Any] = {
        "records": 0,
        "missing_ref": 0,
        "empty_pred": 0,
        "length_mismatch_utts": 0,
        "eval_utts": 0,
        "sum_tse_ms": 0.0,
        "overlap_same_s": 0.0,
        "overlap_total_s": 0.0,
    }
    for tau in TOLERANCES_MS:
        acc[f"sum_acc_{tau}"] = 0.0
    for tau in TOLERANCES_MS:
        for prefix in ("boundary", "label"):
            acc[f"{prefix}_{tau}_matches"] = 0
            acc[f"{prefix}_{tau}_precision_matches"] = 0
            acc[f"{prefix}_{tau}_recall_matches"] = 0
            acc[f"{prefix}_{tau}_pred"] = 0
            acc[f"{prefix}_{tau}_ref"] = 0
    return acc


def evaluate_pair(
    pred_rows_all: list[tuple[str, float, float]],
    ref_rows_all: list[tuple[str, float, float]],
    *,
    exclude_length_mismatch: bool,
    filter_silence_for_metrics: bool,
    ignore_edge_boundaries_for_tse_acc: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    pred_rows = filter_silence(pred_rows_all) if filter_silence_for_metrics else pred_rows_all
    ref_rows = filter_silence(ref_rows_all) if filter_silence_for_metrics else ref_rows_all
    length_match = len(pred_rows) == len(ref_rows)

    out: dict[str, Any] = {
        "ref_len": len(ref_rows),
        "hyp_len": len(pred_rows),
        "length_match": int(length_match),
        "used_for_tse": int(length_match or not exclude_length_mismatch),
    }
    if out["used_for_tse"]:
        n = min(len(pred_rows), len(ref_rows))
        if n:
            tse_sum_s = 0.0
            for idx in range(n):
                include_start = not (ignore_edge_boundaries_for_tse_acc and idx == 0)
                include_end = not (ignore_edge_boundaries_for_tse_acc and idx == n - 1)
                if include_start:
                    tse_sum_s += abs(pred_rows[idx][1] - ref_rows[idx][1])
                if include_end:
                    tse_sum_s += abs(pred_rows[idx][2] - ref_rows[idx][2])
            tse_ms = tse_sum_s / n * 1000.0
        else:
            tse_ms = 0.0
        out["phone_tse_ms"] = tse_ms
        for tau in TOLERANCES_MS:
            if ref_rows:
                correct = 0
                for idx in range(n):
                    include_start = not (ignore_edge_boundaries_for_tse_acc and idx == 0)
                    include_end = not (ignore_edge_boundaries_for_tse_acc and idx == n - 1)
                    start_ok = (
                        True
                        if not include_start
                        else abs(pred_rows[idx][1] - ref_rows[idx][1]) * 1000.0 <= tau
                    )
                    end_ok = (
                        True
                        if not include_end
                        else abs(pred_rows[idx][2] - ref_rows[idx][2]) * 1000.0 <= tau
                    )
                    if start_ok and end_ok:
                        correct += 1
                out[f"phone_acc_{tau}"] = correct / len(ref_rows) * 100.0
            else:
                out[f"phone_acc_{tau}"] = 0.0

    boundary_metrics: dict[str, Any] = {}
    overlap_same, overlap_total = time_overlap(
        pred_rows_all,
        ref_rows_all,
        filter_sil=filter_silence_for_metrics,
    )
    boundary_metrics["overlap_same_s"] = overlap_same
    boundary_metrics["overlap_total_s"] = overlap_total
    boundary_metrics["time_overlap"] = safe_div(overlap_same, overlap_total)
    pred_events = boundary_events(pred_rows_all, filter_sil=filter_silence_for_metrics)
    ref_events = boundary_events(ref_rows_all, filter_sil=filter_silence_for_metrics)
    for tau in TOLERANCES_MS:
        tol_s = tau / 1000.0
        for prefix, label_aware in (("boundary", False), ("label", True)):
            summary = score_events(pred_events, ref_events, tol_s, label_aware=label_aware)
            for key, value in summary.items():
                boundary_metrics[f"{prefix}_{tau}_{key}"] = value
    out.update(boundary_metrics)
    return out, boundary_metrics


def add_to_accumulator(acc: dict[str, Any], row: dict[str, Any]) -> None:
    acc["records"] += 1
    if not row.get("has_ref", 1):
        acc["missing_ref"] += 1
        return
    if not row.get("hyp_len_raw", 0):
        acc["empty_pred"] += 1
    if not row.get("length_match", 0):
        acc["length_mismatch_utts"] += 1
    if row.get("used_for_tse", 0):
        acc["eval_utts"] += 1
        acc["sum_tse_ms"] += float(row.get("phone_tse_ms", 0.0))
        for tau in TOLERANCES_MS:
            acc[f"sum_acc_{tau}"] += float(row.get(f"phone_acc_{tau}", 0.0))
    acc["overlap_same_s"] += float(row.get("overlap_same_s", 0.0))
    acc["overlap_total_s"] += float(row.get("overlap_total_s", 0.0))
    for tau in TOLERANCES_MS:
        for prefix in ("boundary", "label"):
            acc[f"{prefix}_{tau}_matches"] += int(row.get(f"{prefix}_{tau}_matches", 0))
            acc[f"{prefix}_{tau}_precision_matches"] += int(
                row.get(f"{prefix}_{tau}_precision_matches", 0)
            )
            acc[f"{prefix}_{tau}_recall_matches"] += int(
                row.get(f"{prefix}_{tau}_recall_matches", 0)
            )
            acc[f"{prefix}_{tau}_pred"] += int(row.get(f"{prefix}_{tau}_predicted_boundaries", 0))
            acc[f"{prefix}_{tau}_ref"] += int(row.get(f"{prefix}_{tau}_reference_boundaries", 0))


def matrix_row(group: tuple[str, str, str, str, str], acc: dict[str, Any]) -> dict[str, Any]:
    family, dataset, model, head, target = group
    row: dict[str, Any] = {
        "family": family,
        "dataset": dataset,
        "model": model,
        "head": head,
        "target": target,
        "records": acc["records"],
        "missing_ref": acc["missing_ref"],
        "empty_pred": acc["empty_pred"],
        "length_mismatch_utts": acc["length_mismatch_utts"],
        "eval_utts": acc["eval_utts"],
        "phone_tse_ms": safe_div(acc["sum_tse_ms"], acc["eval_utts"]),
        "time_overlap": safe_div(acc["overlap_same_s"], acc["overlap_total_s"]),
        "time_overlap_pct": safe_div(acc["overlap_same_s"], acc["overlap_total_s"]) * 100.0,
    }
    for tau in TOLERANCES_MS:
        row[f"phone_acc_{tau}"] = safe_div(acc[f"sum_acc_{tau}"], acc["eval_utts"])
    for tau in TOLERANCES_MS:
        for prefix in ("boundary", "label"):
            summary = finalize_counts(
                acc[f"{prefix}_{tau}_precision_matches"],
                acc[f"{prefix}_{tau}_recall_matches"],
                acc[f"{prefix}_{tau}_pred"],
                acc[f"{prefix}_{tau}_ref"],
            )
            for key, value in summary.items():
                row[f"{prefix}_{tau}_{key}"] = value
    return row


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if fieldnames:
            writer.writeheader()
            writer.writerows(rows)


def evaluate(args: argparse.Namespace) -> None:
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    sources = []
    if args.source_tsv:
        sources.extend(
            load_source_tsv(
                args.source_tsv,
                family_field=args.source_tsv_family_field,
                path_field=args.source_tsv_path_field,
            )
        )
    if args.source:
        sources.extend(args.source)
    if not sources:
        sources = DEFAULT_SOURCES
    datasets = set(args.datasets)
    phn_index = build_phn_index(args.timit_phn_root)
    dataset_utts = load_dataset_utts(datasets)
    ref_ctms = write_ref_ctms(
        output_dir,
        dataset_utts,
        phn_index,
        sample_rate=args.sample_rate,
        dx_map=args.timit_dx_map,
        closure_mode=args.timit_closure_mode,
        decimals=args.ctm_decimals,
    )

    per_utt_rows: list[dict[str, Any]] = []
    mismatch_rows: list[dict[str, Any]] = []
    matrix_acc: dict[tuple[str, str, str, str, str], dict[str, Any]] = defaultdict(new_accumulator)

    hyp_handles: dict[tuple[str, str, str, str], Any] = {}
    try:
        for family, alignments_path in sources:
            if not alignments_path.is_file():
                print(f"WARNING: missing source {family}: {alignments_path}")
                continue
            for line_no, record in json_records(alignments_path):
                dataset = str(record.get("dataset", ""))
                if dataset not in datasets:
                    continue
                utt = record_utt(record)
                model = record_model(record)
                head = record_head(record)
                target = record_target(record)
                group = (family, dataset, model, head, target)
                pred_rows = prediction_rows(
                    record,
                    dx_map=args.timit_dx_map,
                    hyp_round_step_ms=args.hyp_round_step_ms,
                )
                ref_rows = load_raw_phn_ref(
                    phn_index,
                    utt,
                    sample_rate=args.sample_rate,
                    dx_map=args.timit_dx_map,
                    closure_mode=args.timit_closure_mode,
                )
                base = {
                    "family": family,
                    "dataset": dataset,
                    "model": model,
                    "head": head,
                    "target": target,
                    "utt": utt,
                    "line": line_no,
                    "source": str(alignments_path),
                    "has_ref": int(ref_rows is not None),
                    "hyp_len_raw": len(pred_rows),
                }
                if ref_rows is None:
                    row = {**base, "error": "missing raw TIMIT PHN reference"}
                    per_utt_rows.append(row)
                    add_to_accumulator(matrix_acc[group], row)
                    continue

                row_metrics, _boundary = evaluate_pair(
                    pred_rows,
                    ref_rows,
                    exclude_length_mismatch=not args.include_length_mismatch,
                    filter_silence_for_metrics=not args.keep_silence,
                    ignore_edge_boundaries_for_tse_acc=args.ignore_edge_boundaries_for_tse_acc,
                )
                row = {**base, **row_metrics}
                per_utt_rows.append(row)
                add_to_accumulator(matrix_acc[group], row)

                if not row_metrics["length_match"]:
                    mismatch_rows.append(
                        {
                            **base,
                            "ref_len": row_metrics["ref_len"],
                            "hyp_len": row_metrics["hyp_len"],
                            "length_delta": int(row_metrics["hyp_len"]) - int(row_metrics["ref_len"]),
                            "ref_phones": " ".join(label for label, _s, _e in filter_silence(ref_rows)),
                            "hyp_phones": " ".join(label for label, _s, _e in filter_silence(pred_rows)),
                        }
                    )

                if args.write_hyp_ctm:
                    key = (family, dataset, model, head)
                    if key not in hyp_handles:
                        hyp_dir = output_dir / "ctm/hyp" / safe_name(family) / safe_name(model)
                        if head:
                            hyp_dir = hyp_dir / safe_name(head)
                        hyp_dir.mkdir(parents=True, exist_ok=True)
                        hyp_handles[key] = (hyp_dir / f"{dataset}.phone.ctm").open(
                            "w",
                            encoding="utf-8",
                        )
                    handle = hyp_handles[key]
                    for label, start, end in pred_rows:
                        handle.write(
                            f"{utt} 1 {start:.{args.ctm_decimals}f} "
                            f"{max(0.0, end - start):.{args.ctm_decimals}f} {label}\n"
                        )
    finally:
        for handle in hyp_handles.values():
            handle.close()

    matrix_rows = [matrix_row(group, acc) for group, acc in sorted(matrix_acc.items())]
    prefix = args.output_prefix
    matrix_path = output_dir / f"{prefix}_matrix.csv"
    per_utt_path = output_dir / f"{prefix}_per_utterance.csv"
    mismatch_path = output_dir / f"{prefix}_length_mismatches.csv"
    summary_path = output_dir / f"{prefix}_summary.json"
    write_csv(matrix_path, matrix_rows)
    write_csv(per_utt_path, per_utt_rows)
    write_csv(mismatch_path, mismatch_rows)

    summary = {
        "sources": [{"family": family, "path": str(path)} for family, path in sources],
        "datasets": sorted(datasets),
        "timit_phn_root": str(resolve_path(args.timit_phn_root)),
        "ref_ctms": {dataset: str(path) for dataset, path in ref_ctms.items()},
        "sample_rate": args.sample_rate,
        "timit_dx_map": args.timit_dx_map,
        "timit_closure_mode": args.timit_closure_mode,
        "hyp_round_step_ms": args.hyp_round_step_ms,
        "keep_silence": args.keep_silence,
        "include_length_mismatch": args.include_length_mismatch,
        "ignore_edge_boundaries_for_tse_acc": args.ignore_edge_boundaries_for_tse_acc,
        "matrix": matrix_rows,
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"Saved raw reference CTMs: {output_dir / 'ctm/ref_raw_phn'}")
    print(f"Saved matrix: {matrix_path}")
    print(f"Saved per-utterance: {per_utt_path}")
    print(f"Saved length mismatches: {mismatch_path}")
    print(f"Saved summary: {summary_path}")
    print(f"Rows: matrix={len(matrix_rows)} per_utterance={len(per_utt_rows)} mismatches={len(mismatch_rows)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output-prefix", default="raw_phn_tse_rvalue")
    parser.add_argument("--datasets", nargs="+", default=["timit_dev", "timit_test"])
    parser.add_argument("--source", type=parse_source, action="append", default=None)
    parser.add_argument(
        "--source-tsv",
        type=Path,
        default=None,
        help="TSV with alignment sources. Defaults to columns canonical_family and path.",
    )
    parser.add_argument("--source-tsv-family-field", default="canonical_family")
    parser.add_argument("--source-tsv-path-field", default="path")
    parser.add_argument("--timit-phn-root", type=Path, default=DEFAULT_TIMIT_PHN_ROOT)
    parser.add_argument("--sample-rate", type=float, default=16000.0)
    parser.add_argument("--timit-dx-map", choices=["d", "t", "dx"], default="d")
    parser.add_argument(
        "--timit-closure-mode",
        choices=["silence", "merge_stop"],
        default="silence",
        help=(
            "How to map TIMIT closure labels. 'silence' preserves the previous "
            "39-phone style behavior; 'merge_stop' folds closure intervals into "
            "the following compatible stop/affricate phone."
        ),
    )
    parser.add_argument("--hyp-round-step-ms", type=float, default=0.0)
    parser.add_argument("--ctm-decimals", type=int, default=6)
    parser.add_argument(
        "--include-length-mismatch",
        action="store_true",
        help="Use min(ref_len, hyp_len) for TSE/ACC instead of excluding length-mismatched utterances.",
    )
    parser.add_argument(
        "--keep-silence",
        action="store_true",
        help="Keep silence-like phones in TSE/ACC and boundary metrics.",
    )
    parser.add_argument(
        "--ignore-edge-boundaries-for-tse-acc",
        action="store_true",
        help=(
            "Ignore the first scoring phone start boundary and last scoring phone end "
            "boundary when computing phone_tse_ms and phone_acc_*."
        ),
    )
    parser.add_argument("--write-hyp-ctm", action="store_true")
    return parser.parse_args()


def main() -> None:
    evaluate(parse_args())


if __name__ == "__main__":
    main()
