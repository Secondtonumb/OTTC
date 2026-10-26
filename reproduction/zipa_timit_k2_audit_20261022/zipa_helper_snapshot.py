#!/usr/bin/env python3
"""Evaluate ZipA CTC checkpoints against CMU/ARPAbet monophone references.

ZipA emits IPA tokens. This runner keeps the conversion layer explicit:

* PER: posterior best path IPA tokens -> CMU/ARPAbet phones.
* FA: CMU/ARPAbet reference phones -> IPA token targets, then CTC forced align.
* Text-free timestamps: posterior best path IPA token spans -> CMU/ARPAbet spans.

The output schema follows the local IF-MDD alignment convention closely enough
for downstream visualizers/scorers: alignments.jsonl, phone.ctm,
phone.score.ctm, per_utterance.csv, alignment_errors.csv, and summary.json.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import librosa
import numpy as np
import soundfile as sf
import torch
import torch.nn as nn


REPO_ROOT = Path(__file__).resolve().parents[1]
WORK_ROOT = Path("/work/gm64/m64000")
ZIPA_REPO = WORK_ROOT / "zipa"
DEFAULT_MODEL = REPO_ROOT / "zipa_try/zipa-small-crctc-ns-700k/zipa_small_crctc_extended_0.5_scale_700000_avg10.pth"
DEFAULT_BPE = ZIPA_REPO / "ipa_simplified/unigram_127.model"
DEFAULT_TOKENS = ZIPA_REPO / "ipa_simplified/tokens.txt"
DEFAULT_OUTPUT = REPO_ROOT / "fa_research/results/zipa_cmu_monophone_eval_20260624"

BLANK_LABELS = {"<blk>", "<blank>"}
SPECIAL_LABELS = BLANK_LABELS | {"", "<sos/eos>", "<unk>", "▁"}
SILENCE = {"sil", "sp", "spn", "pau", "h#", "<sil>", "!sil", ""}
TOLERANCE_S = 0.020


ARPABET_TO_IPA_BASE: dict[str, tuple[str, ...]] = {
    "aa": ("ɑ",),
    "ae": ("æ",),
    "ao": ("ɔ",),
    "aw": ("a", "ʊ"),
    "ay": ("a", "ɪ"),
    "b": ("b",),
    "ch": ("t", "ʃ"),
    "d": ("d",),
    "dh": ("ð",),
    "eh": ("ɛ",),
    "er": ("ə", "˞"),
    "ey": ("e", "ɪ"),
    "f": ("f",),
    "g": ("g",),
    "hh": ("h",),
    "ih": ("ɪ",),
    "iy": ("i",),
    "jh": ("d", "ʒ"),
    "k": ("k",),
    "l": ("l",),
    "m": ("m",),
    "n": ("n",),
    "ng": ("ŋ",),
    "ow": ("o", "ʊ"),
    "oy": ("ɔ", "ɪ"),
    "p": ("p",),
    "r": ("ɹ",),
    "s": ("s",),
    "sh": ("ʃ",),
    "t": ("t",),
    "th": ("θ",),
    "uh": ("ʊ",),
    "uw": ("u",),
    "v": ("v",),
    "w": ("w",),
    "y": ("j",),
    "z": ("z",),
    "zh": ("ʒ",),
}

# Greedy reverse map. Multi-token phones must precede singleton phones.
IPA_TO_ARPABET_SEQS: list[tuple[tuple[str, ...], str]] = [
    (("t", "ʃ"), "ch"),
    (("d", "ʒ"), "jh"),
    (("a", "ʊ"), "aw"),
    (("a", "ɪ"), "ay"),
    (("e", "ɪ"), "ey"),
    (("o", "ʊ"), "ow"),
    (("ɔ", "ɪ"), "oy"),
    (("ə", "˞"), "er"),
    (("ɜ", "˞"), "er"),
    (("ɑ",), "aa"),
    (("ɒ",), "aa"),
    (("æ",), "ae"),
    (("ɔ",), "ao"),
    (("ʌ",), "ah"),
    (("ə",), "ah"),
    (("ɐ",), "ah"),
    (("ɘ",), "ah"),
    (("ɛ",), "eh"),
    (("ɜ",), "er"),
    (("e",), "eh"),
    (("ɪ",), "ih"),
    (("i",), "iy"),
    (("ʊ",), "uh"),
    (("u",), "uw"),
    (("o",), "ow"),
    (("a",), "aa"),
    (("b",), "b"),
    (("d",), "d"),
    (("ð",), "dh"),
    (("f",), "f"),
    (("g",), "g"),
    (("h",), "hh"),
    (("k",), "k"),
    (("l",), "l"),
    (("m",), "m"),
    (("n",), "n"),
    (("ŋ",), "ng"),
    (("p",), "p"),
    (("ɹ",), "r"),
    (("r",), "r"),
    (("s",), "s"),
    (("ʃ",), "sh"),
    (("t",), "t"),
    (("ɾ",), "t"),
    (("ʔ",), "t"),
    (("θ",), "th"),
    (("v",), "v"),
    (("w",), "w"),
    (("j",), "y"),
    (("z",), "z"),
    (("ʒ",), "zh"),
]

DROP_IPA_TOKENS = {
    "ʼ",
    "ʰ",
    "ʲ",
    "ʷ",
    "ː",
    "ˠ",
    "ˤ",
    "̚",
    "̥",
    "̩",
    "̪",
    "̴",
    "̺",
    "̃",
}


@dataclass(frozen=True)
class Utterance:
    utt_id: str
    wav: Path
    duration: float
    ref_phones: list[str]
    ref_starts: list[float] | None
    ref_ends: list[float] | None
    metadata: dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--model-id", default="")
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL)
    parser.add_argument(
        "--model-size",
        choices=["auto", "small", "large"],
        default="auto",
        help="Zipformer architecture size; auto infers it from --model-path.",
    )
    parser.add_argument("--bpe-model", type=Path, default=DEFAULT_BPE)
    parser.add_argument("--tokens", type=Path, default=DEFAULT_TOKENS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--reference-field", default="canonical_aligned")
    parser.add_argument(
        "--ah-ipa",
        choices=["schwa", "strut", "ə", "ʌ"],
        default="schwa",
        help="IPA target used when an ARPAbet AH reference phone is forced aligned.",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--offset", type=int, default=0, help="Skip this many usable manifest entries before evaluating.")
    parser.add_argument("--max-items", type=int, default=0, help="Maximum entries after --offset; 0 means no additional cap.")
    parser.add_argument("--limit", type=int, default=0, help="0 means full manifest")
    parser.add_argument("--blank-policy", choices=["drop", "previous", "next", "split"], default="previous")
    parser.add_argument(
        "--native-k2-beam-configs",
        default="80:40:200:100000,160:80:500:300000,320:160:1000:1000000,640:320:2000:2000000",
        help="Comma/space separated search_beam:output_beam:min_active:max_active retries.",
    )
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--decimals", type=int, default=6)
    parser.add_argument("--save-log-probs", action="store_true")
    return parser.parse_args()


def safe_id(text: str) -> str:
    out = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(text)).strip("._")
    return out or "zipa"


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except Exception:
        return default
    return out if math.isfinite(out) else default


def resolve_wav_path(text: str, manifest_dir: Path) -> Path:
    raw = Path(text).expanduser()
    candidates: list[Path] = []
    if raw.is_absolute():
        candidates.append(raw)
    else:
        candidates.extend([manifest_dir / raw, REPO_ROOT / raw, Path.cwd() / raw])
    replacements = [
        ("/home/m64000/work/dataset/", "/work/gm64/m64000/dataset/"),
        ("/home/m64000/work/IF-MDD/", "/work/gm64/m64000/IF-MDD/"),
    ]
    for old, new in replacements:
        if text.startswith(old):
            candidates.append(Path(text.replace(old, new, 1)))
    if text.endswith(".WAV"):
        candidates.append(Path(text[:-4] + ".wav"))
    elif text.endswith(".wav"):
        candidates.append(Path(text[:-4] + ".WAV"))
    stem = raw.stem
    candidates.extend(
        [
            WORK_ROOT / "dataset/speechocean762_flat/speechocean762/wav" / f"{stem}.wav",
            WORK_ROOT / "dataset/speechocean762_flat/speechocean762/wav" / f"{stem}.WAV",
        ]
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return candidates[0]


def load_tokens(path: Path) -> dict[int, str]:
    vocab: dict[int, str] = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                vocab[int(parts[1])] = parts[0]
    if not vocab:
        raise ValueError(f"No tokens loaded from {path}")
    return vocab


def phones_from_value(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        raw = value
    else:
        raw = str(value).split()
    phones = []
    for item in raw:
        phone = str(item).strip().lower()
        if phone:
            phones.append(phone)
    return phones


def timed_reference_from_record(record: dict[str, Any], reference_field: str) -> tuple[list[str], list[float] | None, list[float] | None]:
    phonemes = record.get("phonemes")
    if isinstance(phonemes, list) and phonemes:
        phones: list[str] = []
        starts: list[float] = []
        ends: list[float] = []
        for item in phonemes:
            if not isinstance(item, dict):
                continue
            phone = str(item.get("phoneme", item.get("phone", ""))).strip().lower()
            if not phone:
                continue
            phones.append(phone)
            starts.append(safe_float(item.get("start")))
            ends.append(safe_float(item.get("end")))
        if phones:
            return phones, starts, ends

    phones = phones_from_value(record.get(reference_field))
    starts_key = "canonical_starts"
    ends_key = "canonical_ends"
    if reference_field in {"perceived_train_target", "target", "target_aligned"}:
        starts_key = "target_starts"
        ends_key = "target_ends"
    starts = record.get(starts_key)
    ends = record.get(ends_key)
    if isinstance(starts, list) and isinstance(ends, list) and len(starts) == len(ends) == len(phones):
        return phones, [safe_float(x) for x in starts], [safe_float(x) for x in ends]
    return phones, None, None


def drop_silence(
    phones: list[str],
    starts: list[float] | None = None,
    ends: list[float] | None = None,
) -> tuple[list[str], list[float] | None, list[float] | None]:
    keep = [idx for idx, phone in enumerate(phones) if phone.lower() not in SILENCE]
    out_phones = [phones[idx] for idx in keep]
    if starts is None or ends is None:
        return out_phones, None, None
    return out_phones, [starts[idx] for idx in keep], [ends[idx] for idx in keep]


def load_manifest(path: Path, reference_field: str, limit: int, offset: int = 0, max_items: int = 0) -> list[Utterance]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        items = list(data.items())
    elif isinstance(data, list):
        items = list(enumerate(data))
    else:
        raise ValueError(f"Unsupported manifest payload: {type(data).__name__}")

    utterances: list[Utterance] = []
    seen_usable = 0
    cap = max_items if max_items > 0 else limit
    for key, value in items:
        if not isinstance(value, dict):
            continue
        wav_text = value.get("wav") or value.get("wav_path") or value.get("audio")
        if not wav_text:
            continue
        wav = resolve_wav_path(str(wav_text), path.parent)
        utt_id = str(value.get("id") or value.get("utt_id") or value.get("utt") or Path(str(wav_text)).stem or key)
        phones, starts, ends = timed_reference_from_record(value, reference_field)
        phones, starts, ends = drop_silence(phones, starts, ends)
        if seen_usable < offset:
            seen_usable += 1
            continue
        seen_usable += 1
        utterances.append(
            Utterance(
                utt_id=utt_id,
                wav=wav,
                duration=safe_float(value.get("duration")),
                ref_phones=phones,
                ref_starts=starts,
                ref_ends=ends,
                metadata=dict(value),
            )
        )
        if cap and len(utterances) >= cap:
            break
    if not utterances:
        raise ValueError(f"No usable utterances found in {path}")
    return utterances


def arpabet_to_ipa(phone: str, ah_ipa: str) -> tuple[str, ...]:
    phone = phone.lower()
    if phone == "ah":
        return ("ə",) if ah_ipa in {"schwa", "ə"} else ("ʌ",)
    if phone not in ARPABET_TO_IPA_BASE:
        raise KeyError(phone)
    return ARPABET_TO_IPA_BASE[phone]


def build_token_lookup(vocab: dict[int, str]) -> dict[str, int]:
    return {label: idx for idx, label in vocab.items()}


def reference_to_ipa_targets(
    phones: list[str],
    token_lookup: dict[str, int],
    ah_ipa: str,
) -> tuple[list[int], list[int], list[str]]:
    token_ids: list[int] = []
    group_sizes: list[int] = []
    ipa_tokens: list[str] = []
    for phone in phones:
        mapped = arpabet_to_ipa(phone, ah_ipa)
        ids: list[int] = []
        for token in mapped:
            if token not in token_lookup:
                raise KeyError(f"IPA token {token!r} for phone {phone!r} is absent from ZipA tokens")
            ids.append(token_lookup[token])
        token_ids.extend(ids)
        group_sizes.append(len(ids))
        ipa_tokens.extend(mapped)
    return token_ids, group_sizes, ipa_tokens


def normalize_ipa_tokens(tokens: Iterable[str]) -> list[str]:
    return [tok for tok in tokens if tok not in SPECIAL_LABELS and tok not in DROP_IPA_TOKENS]


def ipa_tokens_to_arpabet(tokens: list[str]) -> tuple[list[str], dict[str, int]]:
    phones: list[str] = []
    stats = {"dropped_diacritics": 0, "unmapped_ipa_tokens": 0}
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if tok in SPECIAL_LABELS:
            i += 1
            continue
        if tok in DROP_IPA_TOKENS:
            stats["dropped_diacritics"] += 1
            i += 1
            continue
        matched = False
        for seq, phone in IPA_TO_ARPABET_SEQS:
            if tuple(tokens[i : i + len(seq)]) == seq:
                phones.append(phone)
                i += len(seq)
                matched = True
                break
        if matched:
            continue
        phones.append(f"ipa_{tok}")
        stats["unmapped_ipa_tokens"] += 1
        i += 1
    return phones, stats


def edit_counts(ref: list[str], hyp: list[str]) -> tuple[int, int, int, int]:
    n, m = len(ref), len(hyp)
    dp = [[(0, 0, 0, 0) for _ in range(m + 1)] for _ in range(n + 1)]
    for i in range(1, n + 1):
        score, ins, dele, sub = dp[i - 1][0]
        dp[i][0] = (score + 1, ins, dele + 1, sub)
    for j in range(1, m + 1):
        score, ins, dele, sub = dp[0][j - 1]
        dp[0][j] = (score + 1, ins + 1, dele, sub)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            best = dp[i - 1][j - 1] if ref[i - 1] == hyp[j - 1] else (
                dp[i - 1][j - 1][0] + 1,
                dp[i - 1][j - 1][1],
                dp[i - 1][j - 1][2],
                dp[i - 1][j - 1][3] + 1,
            )
            score, ins, dele, sub = dp[i][j - 1]
            best = min(best, (score + 1, ins + 1, dele, sub))
            score, ins, dele, sub = dp[i - 1][j]
            best = min(best, (score + 1, ins, dele + 1, sub))
            dp[i][j] = best
    return dp[n][m]


def load_audio(path: Path, sample_rate: int = 16000) -> torch.Tensor:
    audio, sr = sf.read(str(path))
    if getattr(audio, "ndim", 1) > 1:
        audio = audio[:, 0]
    if sr != sample_rate:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=sample_rate)
    return torch.from_numpy(np.asarray(audio)).float()


class ZipATorchRunner:
    def __init__(
        self, model_path: Path, bpe_model: Path, device: str, model_size: str = "auto"
    ):
        self.model_path = model_path
        self.bpe_model = bpe_model
        self.device = torch.device(device if device != "cuda" or torch.cuda.is_available() else "cpu")
        self.model_size = model_size
        self.model = None
        self.fbank = None

    def load(self) -> "ZipATorchRunner":
        from lhotse.features.kaldi.extractors import Fbank, FbankConfig

        size = (
            model_size_from_path(self.model_path)
            if self.model_size == "auto"
            else self.model_size
        )
        self.model = build_crctc_model(ZIPA_REPO, size)
        state_dict = torch.load(
            self.model_path, map_location="cpu", weights_only=False
        )
        if isinstance(state_dict, dict) and "model" in state_dict and isinstance(state_dict["model"], dict):
            state_dict = state_dict["model"]
        missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
        allowed_unexpected = {
            "ctc_weight_output.weight",
            "ctc_weight_output.bias",
        }
        disallowed_unexpected = [
            key for key in unexpected if key not in allowed_unexpected
        ]
        if missing or disallowed_unexpected:
            raise RuntimeError(
                "ZipA load failed: "
                f"missing={missing} unexpected={disallowed_unexpected}"
            )
        self.model.to(self.device).eval()
        self.fbank = Fbank(FbankConfig(num_filters=80, dither=0.0, snip_edges=False))
        return self

    @torch.inference_mode()
    def encode_batch(self, wavs: list[torch.Tensor]) -> tuple[torch.Tensor, list[int]]:
        if self.model is None or self.fbank is None:
            raise RuntimeError("ZipA model is not loaded")
        feats = self.fbank.extract_batch([wav.unsqueeze(0) for wav in wavs], sampling_rate=16000)
        feat_lens = torch.tensor([len(feat) for feat in feats], dtype=torch.int64)
        feats = torch.nn.utils.rnn.pad_sequence(feats, batch_first=True)
        encoder_out, encoder_lens = self.model.forward_encoder(
            feats.to(self.device), feat_lens.to(self.device)
        )
        log_probs = self.model.ctc_output(encoder_out).detach().float().cpu()
        return log_probs, [int(x) for x in encoder_lens.cpu().tolist()]


class AttributeDict(dict):
    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value


def install_minimal_zipa_shims() -> None:
    icefall = sys.modules.setdefault("icefall", types.ModuleType("icefall"))
    utils = types.ModuleType("icefall.utils")

    def make_pad_mask(lengths: torch.Tensor, max_len: int | None = None) -> torch.Tensor:
        if max_len is None:
            max_len = int(lengths.max().item())
        seq = torch.arange(max_len, device=lengths.device)
        return seq.unsqueeze(0) >= lengths.unsqueeze(1)

    def add_sos(ragged: Any, sos_id: int) -> Any:
        rows = ragged.tolist()
        return ragged.__class__([[sos_id] + list(row) for row in rows])

    utils.AttributeDict = AttributeDict
    utils.make_pad_mask = make_pad_mask
    utils.add_sos = add_sos
    icefall.utils = utils
    sys.modules["icefall.utils"] = utils

    # ZipA's AsrModel imports k2 for transducer annotations even on the CTC-only
    # path. Prefer the real module; a stub is only for model construction in an
    # environment where k2 cannot import. The FA path below rejects this stub.
    try:
        import k2  # noqa: F401
    except Exception:
        k2_stub = sys.modules.setdefault("k2", types.ModuleType("k2"))
        if not hasattr(k2_stub, "RaggedTensor"):
            k2_stub.RaggedTensor = object


def _to_int_tuple(value: str) -> tuple[int, ...]:
    return tuple(int(item) for item in value.replace(" ", "").split(","))


def model_size_from_path(path: Path) -> str:
    text = str(path).lower()
    if "large" in text:
        return "large"
    if "small" in text:
        return "small"
    raise ValueError(f"Cannot infer ZipA model size from path; include small or large in filename: {path}")


def crctc_params(size: str) -> AttributeDict:
    common: dict[str, Any] = {
        "feature_dim": 80,
        "subsampling_factor": 4,
        "vocab_size": 127,
        "blank_id": 0,
        "context_size": 2,
        "sos_id": 1,
        "eos_id": 1,
        "ignore_id": -1,
        "label_smoothing": 0.1,
        "causal": False,
        "chunk_size": "16,32,64,-1",
        "left_context_frames": "64,128,256,-1",
        "use_transducer": False,
        "use_ctc": True,
        "use_attention_decoder": False,
        "use_cr_ctc": True,
    }
    if size == "small":
        common.update(
            {
                "num_encoder_layers": "2,2,3,4,3,2",
                "downsampling_factor": "1,2,4,8,4,2",
                "feedforward_dim": "512,768,1024,1536,1024,768",
                "num_heads": "4,4,4,8,4,4",
                "encoder_dim": "192,256,384,512,384,256",
                "query_head_dim": "32",
                "value_head_dim": "12",
                "pos_head_dim": "4",
                "pos_dim": 48,
                "encoder_unmasked_dim": "192,192,256,256,256,192",
                "cnn_module_kernel": "31,31,15,15,15,31",
                "decoder_dim": 512,
                "joiner_dim": 512,
                "attention_decoder_dim": 512,
                "attention_decoder_num_layers": 6,
                "attention_decoder_attention_dim": 512,
                "attention_decoder_num_heads": 8,
                "attention_decoder_feedforward_dim": 2048,
            }
        )
    elif size == "large":
        common.update(
            {
                "num_encoder_layers": "4,3,4,5,4,4",
                "downsampling_factor": "1,2,4,8,4,2",
                "feedforward_dim": "768,768,1536,2048,1536,768",
                "num_heads": "6,6,6,8,6,6",
                "encoder_dim": "512,512,768,1024,768,512",
                "query_head_dim": "64",
                "value_head_dim": "48",
                "pos_head_dim": "4",
                "pos_dim": 48,
                "encoder_unmasked_dim": "192,192,256,320,256,192",
                "cnn_module_kernel": "31,31,15,15,15,31",
                "decoder_dim": 1024,
                "joiner_dim": 1024,
                "attention_decoder_dim": 512,
                "attention_decoder_num_layers": 6,
                "attention_decoder_attention_dim": 512,
                "attention_decoder_num_heads": 8,
                "attention_decoder_feedforward_dim": 2048,
            }
        )
    else:
        raise ValueError(f"Unknown ZipA model size: {size}")
    return AttributeDict(common)


def build_crctc_model(zipa_repo: Path, size: str) -> nn.Module:
    install_minimal_zipa_shims()
    recipe_dir = zipa_repo / "zipformer_crctc"
    sys.path.insert(0, str(recipe_dir))
    from model import AsrModel
    from scaling import ScheduledFloat
    from subsampling import Conv2dSubsampling
    from zipformer import Zipformer2

    params = crctc_params(size)
    encoder_dims = _to_int_tuple(params.encoder_dim)
    encoder_embed = Conv2dSubsampling(
        in_channels=params.feature_dim,
        out_channels=encoder_dims[0],
        dropout=ScheduledFloat((0.0, 0.3), (20000.0, 0.1)),
    )
    encoder = Zipformer2(
        output_downsampling_factor=1,
        downsampling_factor=_to_int_tuple(params.downsampling_factor),
        num_encoder_layers=_to_int_tuple(params.num_encoder_layers),
        encoder_dim=encoder_dims,
        encoder_unmasked_dim=_to_int_tuple(params.encoder_unmasked_dim),
        query_head_dim=_to_int_tuple(params.query_head_dim),
        pos_head_dim=_to_int_tuple(params.pos_head_dim),
        value_head_dim=_to_int_tuple(params.value_head_dim),
        pos_dim=params.pos_dim,
        num_heads=_to_int_tuple(params.num_heads),
        feedforward_dim=_to_int_tuple(params.feedforward_dim),
        cnn_module_kernel=_to_int_tuple(params.cnn_module_kernel),
        dropout=ScheduledFloat((0.0, 0.3), (20000.0, 0.1)),
        warmup_batches=4000.0,
        causal=params.causal,
        chunk_size=_to_int_tuple(params.chunk_size),
        left_context_frames=_to_int_tuple(params.left_context_frames),
    )
    return AsrModel(
        encoder_embed=encoder_embed,
        encoder=encoder,
        decoder=None,
        joiner=None,
        attention_decoder=None,
        encoder_dim=max(encoder_dims),
        decoder_dim=params.decoder_dim,
        vocab_size=params.vocab_size,
        use_transducer=False,
        use_ctc=True,
        use_attention_decoder=False,
    )


def frame_times(start_frame: int, end_frame: int, frame_len: int, duration: float) -> tuple[float, float]:
    if duration <= 0:
        return float(start_frame), float(end_frame)
    start = duration * float(start_frame) / float(max(1, frame_len))
    end = duration * float(end_frame) / float(max(1, frame_len))
    return min(max(start, 0.0), duration), min(max(end, start), duration)


def segment_score(log_values: Iterable[float]) -> float:
    values = np.asarray(list(log_values), dtype=np.float64)
    if values.size == 0:
        return 0.0
    return float(np.exp(np.clip(values, -80.0, 0.0)).mean())


def posterior_ipa_segments(
    matrix: torch.Tensor,
    vocab: dict[int, str],
    duration: float,
) -> list[dict[str, Any]]:
    frame_len = int(matrix.shape[0])
    ids = matrix.argmax(dim=-1).tolist()
    frame_scores = matrix[torch.arange(frame_len), torch.tensor(ids)].tolist()
    segments: list[dict[str, Any]] = []
    cur: str | None = None
    start = 0
    scores: list[float] = []

    def flush(end_frame: int) -> None:
        nonlocal cur, start, scores
        if cur is None:
            scores = []
            return
        onset, offset = frame_times(start, end_frame, frame_len, duration)
        if offset > onset:
            segments.append(
                {
                    "ipa": cur,
                    "phone": cur,
                    "onset": onset,
                    "offset": offset,
                    "start_frame": int(start),
                    "end_frame": int(end_frame),
                    "score": segment_score(scores),
                }
            )
        cur = None
        scores = []

    for frame_idx, (token_id, score) in enumerate(zip(ids, frame_scores)):
        label = vocab.get(int(token_id), "")
        if label in SPECIAL_LABELS:
            label = ""
        if label and label == cur:
            scores.append(float(score))
            continue
        flush(frame_idx)
        if not label:
            start = frame_idx + 1
            continue
        cur = label
        start = frame_idx
        scores = [float(score)]
    flush(frame_len)
    return segments


def ipa_segments_to_arpabet_segments(segments: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, int]]:
    out: list[dict[str, Any]] = []
    stats = {"dropped_diacritics": 0, "unmapped_ipa_tokens": 0}
    i = 0
    tokens = [str(seg["ipa"]) for seg in segments]
    while i < len(segments):
        tok = tokens[i]
        if tok in SPECIAL_LABELS:
            i += 1
            continue
        if tok in DROP_IPA_TOKENS:
            stats["dropped_diacritics"] += 1
            i += 1
            continue
        matched = False
        for seq, phone in IPA_TO_ARPABET_SEQS:
            if tuple(tokens[i : i + len(seq)]) != seq:
                continue
            group = segments[i : i + len(seq)]
            out.append(
                {
                    "phone": phone,
                    "ipa_tokens": list(seq),
                    "onset": float(group[0]["onset"]),
                    "offset": float(group[-1]["offset"]),
                    "start_frame": int(group[0]["start_frame"]),
                    "end_frame": int(group[-1]["end_frame"]),
                    "score": float(sum(float(x.get("score", 0.0)) for x in group) / len(group)),
                }
            )
            i += len(seq)
            matched = True
            break
        if matched:
            continue
        seg = segments[i]
        out.append(
            {
                "phone": f"ipa_{tok}",
                "ipa_tokens": [tok],
                "onset": float(seg["onset"]),
                "offset": float(seg["offset"]),
                "start_frame": int(seg["start_frame"]),
                "end_frame": int(seg["end_frame"]),
                "score": float(seg.get("score", 0.0)),
            }
        )
        stats["unmapped_ipa_tokens"] += 1
        i += 1
    return out, stats


def ctc_min_frames(token_ids: list[int]) -> int:
    repeats = sum(1 for left, right in zip(token_ids, token_ids[1:]) if left == right)
    return len(token_ids) + repeats


def parse_native_k2_beam_configs(value: str) -> list[tuple[float, float, int, int]]:
    configs = []
    for raw in value.replace(",", " ").split():
        parts = raw.split(":")
        if len(parts) != 4:
            raise ValueError(f"bad native-k2 beam config: {raw}")
        configs.append((float(parts[0]), float(parts[1]), int(parts[2]), int(parts[3])))
    if not configs:
        raise ValueError("empty native-k2 beam config list")
    return configs


def alignment_chunks(labels: list[int], probs: list[float], blank_id: int) -> list[tuple[int, int, int, float]]:
    chunks: list[tuple[int, int, int, float]] = []
    active_token = None
    active_start = None
    for frame_idx, token_id in enumerate(labels):
        if int(token_id) == int(blank_id):
            if active_token is not None:
                chunks.append((int(active_token), int(active_start), int(frame_idx), float(np.mean(probs[active_start:frame_idx]))))
                active_token = None
                active_start = None
            continue
        if active_token is None:
            active_token = int(token_id)
            active_start = int(frame_idx)
        elif int(token_id) != int(active_token):
            chunks.append((int(active_token), int(active_start), int(frame_idx), float(np.mean(probs[active_start:frame_idx]))))
            active_token = int(token_id)
            active_start = int(frame_idx)
    if active_token is not None:
        chunks.append((int(active_token), int(active_start), len(labels), float(np.mean(probs[active_start:len(labels)]))))
    return chunks


def target_prefix_coverage(chunks: list[tuple[int, int, int, float]], target_ids: list[int]) -> tuple[int, int]:
    cursor = 0
    for target_index, target_id in enumerate(target_ids):
        while cursor < len(chunks) and int(chunks[cursor][0]) != int(target_id):
            cursor += 1
        if cursor >= len(chunks):
            return target_index, cursor
        cursor += 1
    return len(target_ids), cursor


def native_k2_forced_align_ids(
    log_probs: torch.Tensor,
    target_ids: list[int],
    blank_id: int,
    beam_configs: list[tuple[float, float, int, int]],
) -> tuple[torch.Tensor, torch.Tensor]:
    try:
        import k2
    except Exception as exc:
        raise RuntimeError("k2 could not be imported; run through zipa_train_env.sh / sb_k2_cuda") from exc

    k2_log_probs = log_probs.detach().to("cpu").float().contiguous()
    num_frames = int(k2_log_probs.shape[-2])
    target_ids = [int(token_id) for token_id in target_ids]
    k2_device = torch.device("cpu")
    supervision_segments = torch.tensor([[0, 0, num_frames]], dtype=torch.int32, device=k2_device)
    dense_fsa_vec = k2.DenseFsaVec(k2_log_probs, supervision_segments)
    graph = k2.arc_sort(k2.ctc_graph([target_ids], modified=False, device=k2_device))

    last_error = ""
    for search_beam, output_beam, min_active, max_active in beam_configs:
        for allow_partial in (False, True):
            try:
                lattice = k2.intersect_dense_pruned(
                    graph,
                    dense_fsa_vec,
                    search_beam=search_beam,
                    output_beam=output_beam,
                    min_active_states=min_active,
                    max_active_states=max_active,
                    frame_idx_name="frame_idx",
                    allow_partial=allow_partial,
                )
                best_path = k2.shortest_path(lattice, use_double_scores=True)
                labels = best_path.labels.detach().long()
                frame_idx = best_path.frame_idx.detach().long()
                scores = best_path.scores.detach().float()
                valid = (labels != -1) & (frame_idx >= 0) & (frame_idx < num_frames)
                if not bool(valid.any()):
                    last_error = (
                        f"empty best path beam={search_beam}:{output_beam}:{min_active}:{max_active} "
                        f"allow_partial={allow_partial}"
                    )
                    continue
                dense_labels = torch.full((num_frames,), int(blank_id), dtype=torch.long)
                dense_scores = torch.ones((num_frames,), dtype=torch.float32)
                dense_labels[frame_idx[valid]] = labels[valid]
                dense_scores[frame_idx[valid]] = scores[valid].exp()
                chunks = alignment_chunks(dense_labels.tolist(), dense_scores.tolist(), blank_id)
                matched, used = target_prefix_coverage(chunks, target_ids)
                if matched != len(target_ids):
                    last_error = (
                        f"partial best path matched_prefix={matched}/{len(target_ids)} "
                        f"chunks={len(chunks)} used_chunks={used} "
                        f"beam={search_beam}:{output_beam}:{min_active}:{max_active} "
                        f"allow_partial={allow_partial}"
                    )
                    continue
                return dense_labels, dense_scores
            except Exception as exc:
                last_error = (
                    f"{type(exc).__name__}: {exc} "
                    f"beam={search_beam}:{output_beam}:{min_active}:{max_active} "
                    f"allow_partial={allow_partial}"
                )
    raise RuntimeError(f"native-k2 alignment failed after beam retries: {last_error}")


def dense_path_to_target_segments(
    dense_labels: torch.Tensor,
    dense_scores: torch.Tensor,
    target_ids: list[int],
    blank_id: int,
    duration: float,
    blank_policy: str,
) -> list[dict[str, Any]]:
    labels = dense_labels.detach().cpu().tolist()
    probs = dense_scores.detach().cpu().tolist()
    chunks = alignment_chunks(labels, probs, blank_id)
    segments = []
    cursor = 0
    for target_index, target_id in enumerate(target_ids):
        while cursor < len(chunks) and int(chunks[cursor][0]) != int(target_id):
            cursor += 1
        if cursor >= len(chunks):
            raise ValueError(
                f"could not recover target {target_index}/{len(target_ids)} "
                f"target_id={target_id} from {len(chunks)} chunks"
            )
        token_id, start_frame, end_frame, mean_score = chunks[cursor]
        segments.append(
            {
                "token_id": int(token_id),
                "start_frame": int(start_frame),
                "end_frame": int(end_frame),
                "score": float(mean_score),
            }
        )
        cursor += 1
    apply_blank_policy(segments, len(labels), blank_policy)
    for segment in segments:
        segment["onset"], segment["offset"] = frame_times(
            int(segment["start_frame"]),
            int(segment["end_frame"]),
            len(labels),
            duration,
        )
    return segments


def apply_blank_policy(segments: list[dict[str, Any]], frame_len: int, policy: str) -> None:
    if policy == "drop" or not segments:
        return
    starts = [int(seg["start_frame"]) for seg in segments]
    ends = [int(seg["end_frame"]) for seg in segments]
    segments[0]["start_frame"] = 0
    if policy == "previous":
        for idx in range(len(segments) - 1):
            segments[idx]["end_frame"] = starts[idx + 1]
        segments[-1]["end_frame"] = frame_len
    elif policy == "next":
        for idx in range(1, len(segments)):
            segments[idx]["start_frame"] = ends[idx - 1]
        segments[-1]["end_frame"] = frame_len
    elif policy == "split":
        for idx in range(len(segments) - 1):
            midpoint = int(round((ends[idx] + starts[idx + 1]) / 2.0))
            segments[idx]["end_frame"] = midpoint
            segments[idx + 1]["start_frame"] = midpoint
        segments[-1]["end_frame"] = frame_len


def forced_align_arpabet(
    matrix: torch.Tensor,
    ref_phones: list[str],
    vocab: dict[int, str],
    duration: float,
    ah_ipa: str,
    blank_policy: str,
    beam_configs: list[tuple[float, float, int, int]],
) -> tuple[list[dict[str, Any]], list[str]]:
    frame_len = int(matrix.shape[0])
    token_ids, group_sizes, ipa_tokens = reference_to_ipa_targets(ref_phones, build_token_lookup(vocab), ah_ipa)
    if not token_ids:
        return [], []
    if frame_len < ctc_min_frames(token_ids):
        raise ValueError(f"too few frames for CTC FA: frames={frame_len} target_min={ctc_min_frames(token_ids)}")
    dense_labels, dense_scores = native_k2_forced_align_ids(
        matrix.unsqueeze(0).cpu().float(),
        token_ids,
        blank_id=0,
        beam_configs=beam_configs,
    )
    token_segments = dense_path_to_target_segments(
        dense_labels,
        dense_scores,
        token_ids,
        blank_id=0,
        duration=duration,
        blank_policy=blank_policy,
    )
    if len(token_segments) != len(token_ids):
        raise ValueError(f"FA recovered {len(token_segments)} spans for {len(token_ids)} IPA targets")
    for segment in token_segments:
        segment["ipa"] = vocab.get(int(segment["token_id"]), "")

    out: list[dict[str, Any]] = []
    cursor = 0
    for phone, size in zip(ref_phones, group_sizes):
        group = token_segments[cursor : cursor + size]
        cursor += size
        out.append(
            {
                "phone": phone,
                "ipa_tokens": [str(item["ipa"]) for item in group],
                "onset": float(group[0]["onset"]),
                "offset": float(group[-1]["offset"]),
                "start_frame": int(group[0]["start_frame"]),
                "end_frame": int(group[-1]["end_frame"]),
                "score": float(sum(float(item["score"]) for item in group) / len(group)),
            }
        )
    return out, ipa_tokens


def ctm_line(utt: str, seg: dict[str, Any], decimals: int, score: bool) -> str:
    onset = float(seg["onset"])
    dur = max(0.0, float(seg["offset"]) - onset)
    line = f"{utt} 1 {onset:.{decimals}f} {dur:.{decimals}f} {seg['phone']}"
    if score:
        line += f" {float(seg.get('score', 0.0)):.{decimals}f}"
    return line


def write_alignment_record(
    handle: Any,
    *,
    dataset: str,
    model_id: str,
    alignment_type: str,
    utt: Utterance,
    duration: float,
    segments: list[dict[str, Any]],
    extra: dict[str, Any] | None = None,
) -> None:
    record = {
        "dataset": dataset,
        "model": model_id,
        "alignment_type": alignment_type,
        "utt": utt.utt_id,
        "wav": str(utt.wav),
        "duration": duration,
        "phones": [seg["phone"] for seg in segments],
        "pred_starts": [float(seg["onset"]) for seg in segments],
        "pred_ends": [float(seg["offset"]) for seg in segments],
        "mean_scores": [float(seg.get("score", 0.0)) for seg in segments],
        "start_frames": [int(seg["start_frame"]) for seg in segments],
        "end_frames": [int(seg["end_frame"]) for seg in segments],
        "segments": segments,
    }
    if utt.ref_starts is not None and utt.ref_ends is not None:
        record["gt_phones"] = utt.ref_phones
        record["gt_starts"] = utt.ref_starts
        record["gt_ends"] = utt.ref_ends
    if extra:
        record.update(extra)
    handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_segments_tsv(path: Path, rows: list[tuple[str, Utterance, dict[str, Any]]], decimals: int) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["kind", "utt", "wav", "phone", "onset", "offset", "duration", "score", "start_frame", "end_frame", "ipa_tokens"])
        for kind, utt, seg in rows:
            onset = float(seg["onset"])
            offset = float(seg["offset"])
            writer.writerow(
                [
                    kind,
                    utt.utt_id,
                    str(utt.wav),
                    seg["phone"],
                    f"{onset:.{decimals}f}",
                    f"{offset:.{decimals}f}",
                    f"{max(0.0, offset - onset):.{decimals}f}",
                    f"{float(seg.get('score', 0.0)):.{decimals}f}",
                    int(seg["start_frame"]),
                    int(seg["end_frame"]),
                    " ".join(map(str, seg.get("ipa_tokens", []))),
                ]
            )


def safe_div(num: float, den: float) -> float:
    return num / den if den else 0.0


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def boundary_events(starts: list[float], ends: list[float]) -> list[float]:
    if len(starts) <= 1 or len(ends) <= 1:
        return []
    return [float(ends[idx]) for idx in range(len(ends) - 1)]


def match_count(queries: list[float], refs: list[float], tol: float) -> int:
    if not queries or not refs:
        return 0
    queries_sorted = sorted(float(x) for x in queries)
    refs_sorted = sorted(float(x) for x in refs)
    count = 0
    ref_idx = 0
    for query in queries_sorted:
        while ref_idx < len(refs_sorted) and refs_sorted[ref_idx] < query - tol:
            ref_idx += 1
        if ref_idx < len(refs_sorted) and abs(refs_sorted[ref_idx] - query) <= tol:
            count += 1
            ref_idx += 1
    return count


def r_value(precision: float, recall: float) -> tuple[float, float]:
    eps = 1.0e-8
    os_value = recall / (precision + eps) - 1.0
    r1 = math.sqrt((1.0 - recall) ** 2 + os_value**2)
    r2 = (-os_value + recall - 1.0) / math.sqrt(2.0)
    return 1.0 - (abs(r1) + abs(r2)) / 2.0, os_value


def boundary_prf(pred: list[float], ref: list[float]) -> dict[str, float | int]:
    pm = match_count(pred, ref, TOLERANCE_S)
    rm = match_count(ref, pred, TOLERANCE_S)
    precision = safe_div(pm, len(pred))
    recall = safe_div(rm, len(ref))
    f1 = safe_div(2.0 * precision * recall, precision + recall)
    rval, os_value = r_value(precision, recall) if pred or ref else (0.0, 0.0)
    return {
        "precision_matches": pm,
        "recall_matches": rm,
        "predicted_boundaries": len(pred),
        "reference_boundaries": len(ref),
        "precision_pct": precision * 100.0,
        "recall_pct": recall * 100.0,
        "f1_pct": f1 * 100.0,
        "r_value_pct": rval * 100.0,
        "over_segmentation": os_value,
    }


def aggregate_boundary_prf(pairs: list[tuple[list[float], list[float]]]) -> dict[str, float | int]:
    pm = 0
    rm = 0
    pred_total = 0
    ref_total = 0
    for pred, ref in pairs:
        pred_total += len(pred)
        ref_total += len(ref)
        pm += match_count(pred, ref, TOLERANCE_S)
        rm += match_count(ref, pred, TOLERANCE_S)
    precision = safe_div(pm, pred_total)
    recall = safe_div(rm, ref_total)
    f1 = safe_div(2.0 * precision * recall, precision + recall)
    rval, os_value = r_value(precision, recall) if pred_total or ref_total else (0.0, 0.0)
    return {
        "precision_matches": pm,
        "recall_matches": rm,
        "predicted_boundaries": pred_total,
        "reference_boundaries": ref_total,
        "precision_pct": precision * 100.0,
        "recall_pct": recall * 100.0,
        "f1_pct": f1 * 100.0,
        "r_value_pct": rval * 100.0,
        "over_segmentation": os_value,
    }


def score_forced(rows: list[dict[str, Any]]) -> dict[str, Any]:
    utterances = 0
    phones = 0
    start_abs_sum = 0.0
    end_abs_sum = 0.0
    segment_hits = 0
    boundary_hits = 0
    utt_tse: list[float] = []
    boundary_pairs: list[tuple[list[float], list[float]]] = []
    failures = 0
    for rec in rows:
        pred_starts = rec["pred_starts"]
        pred_ends = rec["pred_ends"]
        gt_starts = rec.get("gt_starts")
        gt_ends = rec.get("gt_ends")
        if gt_starts is None or gt_ends is None:
            continue
        if not (len(pred_starts) == len(pred_ends) == len(gt_starts) == len(gt_ends)):
            failures += 1
            continue
        if not pred_starts:
            continue
        utterances += 1
        utt_abs = 0.0
        for ps, pe, gs, ge in zip(pred_starts, pred_ends, gt_starts, gt_ends):
            start_abs = abs(float(ps) - float(gs))
            end_abs = abs(float(pe) - float(ge))
            start_abs_sum += start_abs
            end_abs_sum += end_abs
            utt_abs += start_abs + end_abs
            phones += 1
            boundary_hits += int(start_abs <= TOLERANCE_S) + int(end_abs <= TOLERANCE_S)
            segment_hits += int(max(start_abs, end_abs) <= TOLERANCE_S)
        utt_tse.append(utt_abs / max(1, len(pred_starts)) * 1000.0)
        boundary_pairs.append((boundary_events(pred_starts, pred_ends), boundary_events(gt_starts, gt_ends)))
    prf = aggregate_boundary_prf(boundary_pairs)
    return {
        "utterances_with_gt": utterances,
        "length_mismatch_failures": failures,
        "nonsil_phones": phones,
        "TSE_no_sil_ms": safe_div(sum(utt_tse), len(utt_tse)),
        "TSE_no_sil_global_ms": safe_div(start_abs_sum + end_abs_sum, phones) * 1000.0,
        "TSE_median_ms": percentile(utt_tse, 50),
        "TSE_p90_ms": percentile(utt_tse, 90),
        "boundary_mae_no_sil_ms": safe_div(start_abs_sum + end_abs_sum, 2 * phones) * 1000.0,
        "boundary_ACC20_no_sil_pct": safe_div(boundary_hits, 2 * phones) * 100.0,
        "segment_ACC20_no_sil_pct": safe_div(segment_hits, phones) * 100.0,
        "boundary_P_20ms_no_sil_pct": prf["precision_pct"],
        "boundary_R_20ms_no_sil_pct": prf["recall_pct"],
        "boundary_F1_20ms_no_sil_pct": prf["f1_pct"],
        "boundary_R_value_20ms_no_sil_pct": prf["r_value_pct"],
    }


def score_text_free_boundary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    boundary_pairs: list[tuple[list[float], list[float]]] = []
    utts = 0
    for rec in rows:
        gt_starts = rec.get("gt_starts")
        gt_ends = rec.get("gt_ends")
        if gt_starts is None or gt_ends is None:
            continue
        utts += 1
        boundary_pairs.append((boundary_events(rec["pred_starts"], rec["pred_ends"]), boundary_events(gt_starts, gt_ends)))
    prf = aggregate_boundary_prf(boundary_pairs)
    return {
        "utterances_with_gt": utts,
        "boundary_P_20ms_no_sil_pct": prf["precision_pct"],
        "boundary_R_20ms_no_sil_pct": prf["recall_pct"],
        "boundary_F1_20ms_no_sil_pct": prf["f1_pct"],
        "boundary_R_value_20ms_no_sil_pct": prf["r_value_pct"],
        "predicted_boundaries": prf["predicted_boundaries"],
        "reference_boundaries": prf["reference_boundaries"],
    }


def json_record_from_segments(dataset: str, model_id: str, alignment_type: str, utt: Utterance, duration: float, segments: list[dict[str, Any]], extra: dict[str, Any] | None = None) -> dict[str, Any]:
    rec = {
        "dataset": dataset,
        "model": model_id,
        "alignment_type": alignment_type,
        "utt": utt.utt_id,
        "wav": str(utt.wav),
        "duration": duration,
        "phones": [seg["phone"] for seg in segments],
        "pred_starts": [float(seg["onset"]) for seg in segments],
        "pred_ends": [float(seg["offset"]) for seg in segments],
        "mean_scores": [float(seg.get("score", 0.0)) for seg in segments],
    }
    if utt.ref_starts is not None and utt.ref_ends is not None:
        rec["gt_phones"] = utt.ref_phones
        rec["gt_starts"] = utt.ref_starts
        rec["gt_ends"] = utt.ref_ends
    if extra:
        rec.update(extra)
    return rec


def maybe_open_h5(path: Path, vocab: dict[int, str], utterances: list[Utterance], args: argparse.Namespace):
    if not args.save_log_probs:
        return None, None
    import h5py

    h5 = h5py.File(path, "w")
    h5.attrs["model_path"] = str(args.model_path)
    h5.attrs["manifest"] = str(args.manifest)
    h5.attrs["dataset_name"] = args.dataset_name
    h5.create_dataset("ids", data=[u.utt_id for u in utterances], dtype=h5py.string_dtype("utf-8"))
    h5.create_dataset("labels", data=[vocab[idx] for idx in sorted(vocab)], dtype=h5py.string_dtype("utf-8"))
    h5.create_dataset("durations", shape=(len(utterances),), dtype=np.float32)
    h5.create_dataset("frame_lengths", shape=(len(utterances),), dtype=np.int32)
    return h5, None


def main() -> None:
    args = parse_args()
    ah_ipa = {"schwa": "ə", "strut": "ʌ"}.get(args.ah_ipa, args.ah_ipa)
    beam_configs = parse_native_k2_beam_configs(args.native_k2_beam_configs)
    model_id = safe_id(args.model_id or args.model_path.parent.name or args.model_path.stem)
    out_dir = args.output_dir / model_id / safe_id(args.dataset_name)
    if out_dir.exists() and not args.overwrite:
        raise FileExistsError(f"{out_dir} exists; pass --overwrite")
    out_dir.mkdir(parents=True, exist_ok=True)
    tf_dir = out_dir / "text_free"
    fa_dir = out_dir / "forced_alignment"
    tf_dir.mkdir(parents=True, exist_ok=True)
    fa_dir.mkdir(parents=True, exist_ok=True)

    vocab = load_tokens(args.tokens)
    utterances = load_manifest(args.manifest, args.reference_field, args.limit, args.offset, args.max_items)
    runner = ZipATorchRunner(
        args.model_path, args.bpe_model, args.device, args.model_size
    ).load()

    per_rows: list[dict[str, Any]] = []
    fa_errors: list[dict[str, str]] = []
    text_records: list[dict[str, Any]] = []
    forced_records: list[dict[str, Any]] = []
    text_segment_rows: list[tuple[str, Utterance, dict[str, Any]]] = []
    fa_segment_rows: list[tuple[str, Utterance, dict[str, Any]]] = []
    mapping_stats = {"dropped_diacritics": 0, "unmapped_ipa_tokens": 0}

    text_align_f = (tf_dir / "alignments.jsonl").open("w", encoding="utf-8")
    text_ctm_f = (tf_dir / "phone.ctm").open("w", encoding="utf-8")
    text_score_ctm_f = (tf_dir / "phone.score.ctm").open("w", encoding="utf-8")
    fa_align_f = (fa_dir / "alignments.jsonl").open("w", encoding="utf-8")
    fa_ctm_f = (fa_dir / "phone.ctm").open("w", encoding="utf-8")
    fa_score_ctm_f = (fa_dir / "phone.score.ctm").open("w", encoding="utf-8")
    h5, log_dset = maybe_open_h5(out_dir / "log_probs.h5", vocab, utterances, args)

    total_err = total_ins = total_del = total_sub = total_ref = 0
    max_frames = 0
    try:
        for start in range(0, len(utterances), args.batch_size):
            batch = utterances[start : start + args.batch_size]
            wavs = [load_audio(utt.wav) for utt in batch]
            log_probs, frame_lengths = runner.encode_batch(wavs)
            batch_frames = int(log_probs.shape[1])
            classes = int(log_probs.shape[2])
            if h5 is not None:
                if log_dset is None:
                    import h5py  # noqa: F401

                    log_dset = h5.create_dataset(
                        "log_probs",
                        shape=(len(utterances), 0, classes),
                        maxshape=(len(utterances), None, classes),
                        chunks=(1, min(256, max(1, batch_frames)), classes),
                        dtype=np.float32,
                        fillvalue=-1.0e30,
                    )
                if batch_frames > max_frames:
                    log_dset.resize((len(utterances), batch_frames, classes))
                    max_frames = batch_frames

            for local_idx, utt in enumerate(batch):
                row_idx = start + local_idx
                frame_len = int(frame_lengths[local_idx])
                duration = utt.duration if utt.duration > 0 else float(wavs[local_idx].numel()) / 16000.0
                matrix = log_probs[local_idx, :frame_len, :].contiguous()
                if h5 is not None and log_dset is not None:
                    log_dset[row_idx, :frame_len, :] = matrix.numpy()
                    h5["durations"][row_idx] = duration
                    h5["frame_lengths"][row_idx] = frame_len

                ipa_segments = posterior_ipa_segments(matrix, vocab, duration)
                arpabet_segments, stats = ipa_segments_to_arpabet_segments(ipa_segments)
                for key, value in stats.items():
                    mapping_stats[key] += int(value)
                hyp_phones = [seg["phone"] for seg in arpabet_segments]
                err, ins, dele, sub = edit_counts(utt.ref_phones, hyp_phones)
                total_err += err
                total_ins += ins
                total_del += dele
                total_sub += sub
                total_ref += len(utt.ref_phones)
                per_rows.append(
                    {
                        "utt": utt.utt_id,
                        "wav": str(utt.wav),
                        "duration": duration,
                        "ref_len": len(utt.ref_phones),
                        "hyp_len": len(hyp_phones),
                        "errors": err,
                        "insertions": ins,
                        "deletions": dele,
                        "substitutions": sub,
                        "per": 100.0 * safe_div(err, len(utt.ref_phones)),
                        "ref_arpabet": " ".join(utt.ref_phones),
                        "hyp_arpabet": " ".join(hyp_phones),
                        "hyp_ipa": " ".join(seg["ipa"] for seg in ipa_segments),
                        "dropped_diacritics": stats["dropped_diacritics"],
                        "unmapped_ipa_tokens": stats["unmapped_ipa_tokens"],
                    }
                )

                text_rec = json_record_from_segments(
                    args.dataset_name,
                    model_id,
                    "posterior_best_path_ipa_to_arpabet",
                    utt,
                    duration,
                    arpabet_segments,
                    extra={"raw_ipa_segments": ipa_segments},
                )
                text_records.append(text_rec)
                text_align_f.write(json.dumps(text_rec, ensure_ascii=False) + "\n")
                for seg in arpabet_segments:
                    text_ctm_f.write(ctm_line(utt.utt_id, seg, args.decimals, False) + "\n")
                    text_score_ctm_f.write(ctm_line(utt.utt_id, seg, args.decimals, True) + "\n")
                    text_segment_rows.append(("text_free", utt, seg))

                if not utt.ref_phones:
                    fa_errors.append({"utt": utt.utt_id, "stage": "reference", "error": "empty reference after silence filtering"})
                else:
                    try:
                        fa_segments, target_ipa = forced_align_arpabet(
                            matrix,
                            utt.ref_phones,
                            vocab,
                            duration,
                            ah_ipa,
                            args.blank_policy,
                            beam_configs,
                        )
                        fa_rec = json_record_from_segments(
                            args.dataset_name,
                            model_id,
                            f"{args.reference_field}_fa_arpabet_to_ipa",
                            utt,
                            duration,
                            fa_segments,
                            extra={
                                "reference_field": args.reference_field,
                                "reference_arpabet": " ".join(utt.ref_phones),
                                "reference_ipa_tokens": target_ipa,
                                "fa_backend": "native-k2",
                                "native_k2_beam_configs": args.native_k2_beam_configs,
                                "blank_policy": args.blank_policy,
                                "silence_policy": "drop_silence_before_alignment",
                                "ah_ipa": ah_ipa,
                            },
                        )
                        forced_records.append(fa_rec)
                        fa_align_f.write(json.dumps(fa_rec, ensure_ascii=False) + "\n")
                        for seg in fa_segments:
                            fa_ctm_f.write(ctm_line(utt.utt_id, seg, args.decimals, False) + "\n")
                            fa_score_ctm_f.write(ctm_line(utt.utt_id, seg, args.decimals, True) + "\n")
                            fa_segment_rows.append(("forced_alignment", utt, seg))
                    except Exception as exc:
                        fa_errors.append({"utt": utt.utt_id, "stage": "forced_alignment", "error": f"{type(exc).__name__}: {exc}"})

            done = min(start + len(batch), len(utterances))
            if args.progress_every and (done == len(utterances) or done % args.progress_every == 0 or start == 0):
                print(f"{model_id}/{args.dataset_name}: {done}/{len(utterances)}", flush=True)
    finally:
        text_align_f.close()
        text_ctm_f.close()
        text_score_ctm_f.close()
        fa_align_f.close()
        fa_ctm_f.close()
        fa_score_ctm_f.close()
        if h5 is not None:
            h5.attrs["num_utterances"] = len(utterances)
            h5.attrs["max_frames"] = int(max_frames)
            h5.close()

    write_segments_tsv(tf_dir / "segments.tsv", text_segment_rows, args.decimals)
    write_segments_tsv(fa_dir / "segments.tsv", fa_segment_rows, args.decimals)

    with (out_dir / "per_utterance.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(per_rows[0].keys()) if per_rows else ["utt"])
        writer.writeheader()
        writer.writerows(per_rows)
    with (fa_dir / "alignment_errors.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["utt", "stage", "error"])
        writer.writeheader()
        writer.writerows(fa_errors)

    per_summary = {
        "ref_phones": total_ref,
        "errors": total_err,
        "insertions": total_ins,
        "deletions": total_del,
        "substitutions": total_sub,
        "per": 100.0 * safe_div(total_err, total_ref),
    }
    summary = {
        "model_id": model_id,
        "model_path": str(args.model_path),
        "tokens": str(args.tokens),
        "bpe_model": str(args.bpe_model),
        "manifest": str(args.manifest),
        "dataset_name": args.dataset_name,
        "reference_field": args.reference_field,
        "ah_ipa": ah_ipa,
        "utterances": len(utterances),
        "offset": args.offset,
        "max_items": args.max_items,
        "limit": args.limit,
        "silence_policy": "drop_silence_before_per_and_alignment",
        "mapping_stats": mapping_stats,
        "per": per_summary,
        "text_free": {
            "alignments_jsonl": str(tf_dir / "alignments.jsonl"),
            "phone_ctm": str(tf_dir / "phone.ctm"),
            "phone_score_ctm": str(tf_dir / "phone.score.ctm"),
            "segments_tsv": str(tf_dir / "segments.tsv"),
            "boundary_metrics": score_text_free_boundary(text_records),
        },
        "forced_alignment": {
            "alignments_jsonl": str(fa_dir / "alignments.jsonl"),
            "phone_ctm": str(fa_dir / "phone.ctm"),
            "phone_score_ctm": str(fa_dir / "phone.score.ctm"),
            "segments_tsv": str(fa_dir / "segments.tsv"),
            "alignment_errors_csv": str(fa_dir / "alignment_errors.csv"),
            "failures": len(fa_errors),
            "backend": "native-k2",
            "native_k2_beam_configs": args.native_k2_beam_configs,
            "metrics": score_forced(forced_records),
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (out_dir / "MAPPING.md").write_text(mapping_markdown(args, summary), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def mapping_markdown(args: argparse.Namespace, summary: dict[str, Any]) -> str:
    lines = [
        f"# ZipA IPA to CMU/ARPAbet Mapping: {summary['dataset_name']}",
        "",
        f"- Model: `{summary['model_path']}`",
        f"- Manifest: `{summary['manifest']}`",
        f"- Reference field: `{summary['reference_field']}`",
        "- Silence phones are dropped before PER and FA because ZipA's IPA inventory has no CMU silence phone.",
        f"- AH reference phones are mapped to IPA `{summary['ah_ipa']}` for reference-conditioned FA.",
        "",
        "## ARPAbet to IPA for FA",
        "",
    ]
    ah_ipa = summary["ah_ipa"]
    for phone in sorted(ARPABET_TO_IPA_BASE):
        mapped = ("ə",) if phone == "ah" and ah_ipa == "ə" else ARPABET_TO_IPA_BASE[phone]
        lines.append(f"- `{phone}` -> `{' '.join(mapped)}`")
    lines.extend(["", "## IPA to ARPAbet for PER/Text-Free", ""])
    for seq, phone in IPA_TO_ARPABET_SEQS:
        lines.append(f"- `{' '.join(seq)}` -> `{phone}`")
    lines.extend(
        [
            "",
            "## Mapping Stats",
            "",
            f"- Dropped diacritics: `{summary['mapping_stats']['dropped_diacritics']}`",
            f"- Unmapped IPA tokens retained as `ipa_<token>`: `{summary['mapping_stats']['unmapped_ipa_tokens']}`",
            "",
        ]
    )
    return "\n".join(lines)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        raise SystemExit(f"{type(exc).__name__}: {exc}") from exc
