"""Boundary Precision/Recall/F1/R-value helpers.

This follows the ss-phoneme-seg evaluation style: precision and recall use
separate greedy assignments, then R-value is computed from recall and
over-segmentation.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any


EPS = 1e-8
BoundaryEvent = float | tuple[float, frozenset[str]]


def event_time_labels(event: BoundaryEvent) -> tuple[float, frozenset[str]]:
    if isinstance(event, tuple):
        return float(event[0]), frozenset(str(label) for label in event[1])
    return float(event), frozenset()


def safe_div(num: float, den: float) -> float:
    return num / den if den else 0.0


def r_value(precision: float, recall: float) -> tuple[float, float]:
    os_value = recall / (precision + EPS) - 1.0
    r1 = math.sqrt((1.0 - recall) ** 2 + os_value**2)
    r2 = (-os_value + recall - 1.0) / math.sqrt(2.0)
    return 1.0 - (abs(r1) + abs(r2)) / 2.0, os_value


def get_metrics(
    precision_counter: int,
    recall_counter: int,
    predicted_count: int,
    reference_count: int,
) -> dict[str, float | int]:
    precision = precision_counter / (predicted_count + EPS)
    recall = recall_counter / (reference_count + EPS)
    f1 = 2.0 * (precision * recall) / (precision + recall + EPS)
    rval, os_value = r_value(precision, recall)
    return {
        "matches": precision_counter,
        "precision_matches": precision_counter,
        "recall_matches": recall_counter,
        "predicted_boundaries": predicted_count,
        "reference_boundaries": reference_count,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "r_value": rval,
        "over_segmentation": os_value,
        "precision_pct": precision * 100.0,
        "recall_pct": recall * 100.0,
        "f1_pct": f1 * 100.0,
        "r_value_pct": rval * 100.0,
    }


def _directional_match_count(
    references: Sequence[BoundaryEvent],
    queries: Sequence[BoundaryEvent],
    tolerance_s: float,
    *,
    label_aware: bool,
    lenient: bool = False,
) -> int:
    ref_events = [event_time_labels(event) for event in references]
    query_events = [event_time_labels(event) for event in queries]
    used_refs: set[int] = set()
    matches = 0
    duplicates = 0
    for query_time, query_labels in query_events:
        candidates: list[int] = []
        for ref_idx, (ref_time, ref_labels) in enumerate(ref_events):
            distance = abs(ref_time - query_time)
            if distance > tolerance_s:
                continue
            if label_aware and not query_labels.intersection(ref_labels):
                continue
            candidates.append(ref_idx)
        matched_unused = False
        matched_used = False
        # ss-phoneme-seg collects candidates by distance, then sorts reference
        # indices before assignment. Preserve that behavior exactly here.
        for ref_idx in sorted(candidates):
            if ref_idx in used_refs:
                matched_used = True
                continue
            used_refs.add(ref_idx)
            matches += 1
            matched_unused = True
            break
        if not matched_unused and matched_used:
            duplicates += 1
    return matches + duplicates if lenient else matches


def precision_recall_counts(
    predicted_events: Sequence[BoundaryEvent],
    reference_events: Sequence[BoundaryEvent],
    tolerance_s: float,
    *,
    label_aware: bool = False,
    lenient: bool = False,
) -> dict[str, int]:
    precision_matches = _directional_match_count(
        reference_events,
        predicted_events,
        tolerance_s,
        label_aware=label_aware,
        lenient=lenient,
    )
    recall_matches = _directional_match_count(
        predicted_events,
        reference_events,
        tolerance_s,
        label_aware=label_aware,
        lenient=lenient,
    )
    return {
        "precision_matches": precision_matches,
        "recall_matches": recall_matches,
        "predicted_boundaries": len(predicted_events),
        "reference_boundaries": len(reference_events),
    }


def finalize_counts(
    precision_matches: int,
    recall_matches: int,
    predicted_count: int,
    reference_count: int,
) -> dict[str, float | int]:
    return get_metrics(precision_matches, recall_matches, predicted_count, reference_count)


def score_events(
    predicted_events: Sequence[BoundaryEvent],
    reference_events: Sequence[BoundaryEvent],
    tolerance_s: float,
    *,
    label_aware: bool = False,
    lenient: bool = False,
) -> dict[str, float | int]:
    counts = precision_recall_counts(
        predicted_events,
        reference_events,
        tolerance_s,
        label_aware=label_aware,
        lenient=lenient,
    )
    return finalize_counts(
        counts["precision_matches"],
        counts["recall_matches"],
        counts["predicted_boundaries"],
        counts["reference_boundaries"],
    )


def add_count_metrics(acc: dict[str, Any], prefix: str, metrics: dict[str, Any]) -> None:
    acc[f"{prefix}_precision_matches"] += int(metrics[f"{prefix}_precision_matches"])
    acc[f"{prefix}_recall_matches"] += int(metrics[f"{prefix}_recall_matches"])
    acc[f"{prefix}_predicted_boundaries"] += int(metrics[f"{prefix}_predicted_boundaries"])
    acc[f"{prefix}_reference_boundaries"] += int(metrics[f"{prefix}_reference_boundaries"])
