# SPDX-FileCopyrightText: 2024 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Kaloga Yacouba yacouba.kaloga@idiap.ch
# SPDX-FileContributor: Shashi Kumar shashi.kumar@idiap.ch
#
# SPDX-License-Identifier: MIT

from __future__ import annotations

import json
import pickle
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
from datasets import load_metric

from ottc.config.constants import BORDERING_TOKEN, SPLITTING_DOUBLE_LETTER_TOKEN


RUN_LOG_ID = datetime.now().strftime("%Y%m%d-%H%M%S")


def prepare_eval_logging_dir(base_dir: Union[str, Path], run_id: Optional[str] = None) -> Path:
    target_run_id = run_id or RUN_LOG_ID
    base_path = Path(base_dir)
    path = base_path / "transcripts" / target_run_id
    path.mkdir(parents=True, exist_ok=True)
    return path


def _dump_eval_transcripts(
    logging_dir: Path,
    epoch_index: int,
    predictions: Iterable[str],
    references: Iterable[str],
    extra_info: Optional[Dict[str, Any]] = None,
) -> None:
    output_path = Path(logging_dir) / f"epoch_{epoch_index:02d}.json"
    payload: Dict[str, Any] = {
        "predictions": list(predictions),
        "references": list(references),
    }
    if extra_info:
        payload.update(extra_info)

    try:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=True, indent=2)
    except OSError as exc:
        print(f"Failed to write evaluation transcripts for epoch {epoch_index}: {exc}")


def _dump_eval_alignments(logging_dir: Path, epoch_index: int, payload: Any) -> Optional[Path]:
    output_path = Path(logging_dir) / f"epoch_{epoch_index:02d}_alignments.pkl"
    try:
        with open(output_path, "wb") as f:
            pickle.dump(payload, f)
    except OSError as exc:
        print(f"Failed to write evaluation alignments for epoch {epoch_index}: {exc}")
        return None
    return output_path


def _split_prediction_outputs(predictions: Any) -> Tuple[np.ndarray, List[Any]]:
    if isinstance(predictions, np.ndarray):
        return predictions, []
    if isinstance(predictions, (list, tuple)):
        logits = np.asarray(predictions[0])
        extras = [item for item in predictions[1:]]
        return logits, extras
    return np.asarray(predictions), []


def _to_numpy(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value
    try:
        return np.asarray(value)
    except Exception:
        return value


def _build_alignment_payload(extra_outputs: List[Any]) -> Dict[str, Any]:
    payload: Dict[str, Any] = {}
    if not extra_outputs:
        return payload

    try:
        first = extra_outputs[0] if len(extra_outputs) > 0 else None
        if isinstance(first, (list, tuple)) and len(first) == 3:
            payload["weights_logits"] = _to_numpy(first[0])
            payload["output_mask"] = _to_numpy(first[1])
            payload["weights_labels"] = _to_numpy(first[2])
        elif first is not None:
            payload["extra_0"] = _to_numpy(first)

        second = extra_outputs[1] if len(extra_outputs) > 1 else None
        if isinstance(second, (list, tuple)) and len(second) == 3:
            payload["index_logits"] = _to_numpy(second[0])
            payload["index_labels"] = _to_numpy(second[1])
            payload["transport_plan"] = _to_numpy(second[2])
        elif second is not None:
            payload["extra_1"] = _to_numpy(second)

        if len(extra_outputs) > 2:
            payload["extra_remaining"] = [_to_numpy(item) for item in extra_outputs[2:]]
    except Exception as exc:
        payload = {"raw_outputs": extra_outputs, "parsing_error": str(exc)}
    return payload

def add_ignore_token(processor: Any) -> int:
    ''' Necesary to ignore some tokens where the alpha (OTTC) is null'''

    # Let's define a custom special token
    IGNORE_TOKEN = "<ignored>"

    # Add it to the tokenizer if not already there
    if IGNORE_TOKEN not in processor.tokenizer.get_vocab():
        processor.tokenizer.add_special_tokens({'additional_special_tokens': [IGNORE_TOKEN]})
        special_token_id = processor.tokenizer.convert_tokens_to_ids(IGNORE_TOKEN)
    else:
        special_token_id = processor.tokenizer.convert_tokens_to_ids(IGNORE_TOKEN)

    return special_token_id

def build_compute_metrics_ctc(
    processor: Any,
    logging_dir: Optional[Path] = None,
    capture_alignments: bool = False,
) -> Any:
    wer_metric = load_metric("wer")
    epoch_counter = {"value": 1}

    def compute(pred: Any) -> Dict[str, float]:
        logits, extra_outputs = _split_prediction_outputs(pred.predictions)
        pred_ids = np.argmax(logits, axis=-1)

        label_ids = np.asarray(pred.label_ids).copy()
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

        pred_str = processor.batch_decode(pred_ids)
        label_str = processor.batch_decode(label_ids, group_tokens=False)

        wer = wer_metric.compute(predictions=pred_str, references=label_str)

        if logging_dir is not None:
            epoch_index = epoch_counter["value"]
            extra_info: Dict[str, Any] = {}
            if capture_alignments and extra_outputs:
                alignment_payload = _build_alignment_payload(extra_outputs)
                align_path = _dump_eval_alignments(logging_dir, epoch_index, alignment_payload)
                if align_path is not None:
                    extra_info["alignments_file"] = str(align_path)
            _dump_eval_transcripts(logging_dir, epoch_index, pred_str, label_str, extra_info if extra_info else None)
            epoch_counter["value"] = epoch_index + 1

        return {"wer": wer}

    return compute


def build_compute_metrics_ottc(
    processor: Any,
    logging_dir: Optional[Path] = None,
    capture_alignments: bool = False,
) -> Any:
    """
    Duplicate of the CTC metrics builder ready for OTTC-specific tweaks.
    """
    wer_metric = load_metric("wer")
    epoch_counter = {"value": 1}

    def compute(pred: Any) -> Dict[str, float]:
        ignore_token_id = add_ignore_token(processor)

        logits, extra_outputs = _split_prediction_outputs(pred.predictions)

        pred_ids = np.argmax(logits, axis=-1)
        output_mask = (extra_outputs[0][1] > 0) + 0

        pred_ids[output_mask == 0] = processor.tokenizer.pad_token_id
        pred_ids[pred_ids == processor.tokenizer.unk_token_id] = processor.tokenizer.pad_token_id
        # pred_ids[pred_ids == SPLITTING_DOUBLE_LETTER_TOKEN[1]] = processor.tokenizer.pad_token_id
        # pred_ids[pred_ids == BORDERING_TOKEN[1]] = processor.tokenizer.pad_token_id

        weights_logits = extra_outputs[0]
        pred_ids[weights_logits == 0] = ignore_token_id

        label_ids = pred.label_ids
        # label_ids[label_ids == SPLITTING_DOUBLE_LETTER_TOKEN[1]] = processor.tokenizer.pad_token_id
        # label_ids[label_ids == BORDERING_TOKEN[1]] = processor.tokenizer.pad_token_id
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
        label_ids[label_ids == processor.tokenizer.unk_token_id] = processor.tokenizer.pad_token_id

        pred_str = processor.batch_decode(pred_ids)
        label_str = processor.batch_decode(label_ids, group_tokens=False)

        wer = wer_metric.compute(predictions=pred_str, references=label_str)

        if logging_dir is not None:
            epoch_index = epoch_counter["value"]
            extra_info: Dict[str, Any] = {}
            if capture_alignments and extra_outputs:
                alignment_payload = _build_alignment_payload(extra_outputs)
                align_path = _dump_eval_alignments(logging_dir, epoch_index, alignment_payload)
                if align_path is not None:
                    extra_info["alignments_file"] = str(align_path)
            _dump_eval_transcripts(
                logging_dir,
                epoch_index,
                pred_str,
                label_str,
                extra_info if extra_info else None,
            )
            epoch_counter["value"] = epoch_index + 1

        return {"wer": wer}

    return compute


def build_compute_metrics(
    processor: Any,
    loss_type: str = "ctc",
    logging_dir: Optional[Path] = None,
    capture_alignments: bool = False,
) -> Any:
    """
    Backward-compatible alias that returns the CTC compute-metrics helper.
    """
    if loss_type == "ctc":
        return build_compute_metrics_ctc(processor, logging_dir, capture_alignments)
    elif loss_type == "ottc":
        return build_compute_metrics_ottc(processor, logging_dir, capture_alignments)
    else:
        raise ValueError(f"Unsupported loss_type '{loss_type}'")


def compute_metrics(pred: Any, processor: Any) -> Dict[str, float]:
    """
    Legacy compute_metrics helper kept for compatibility.
    """
    compute_fn = build_compute_metrics_ctc(processor)
    return compute_fn(pred)
