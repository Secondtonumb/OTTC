<!--
SPDX-FileCopyrightText: 2024 Idiap Research Institute <contact@idiap.ch>

SPDX-FileContributor: Kaloga Yacouba yacouba.kaloga@idiap.ch
SPDX-FileContributor: Shashi Kumar shashi.kumar@idiap.ch

SPDX-License-Identifier: MIT
-->

# OTTC: Optimal Temporal Transport Classification

本 fork 的近期修改与复现入口见 [修改整理（2026-10-21）](reproduction/CHANGES_20261021_ZH.md)。

[全部当前实验结果、dev/test 覆盖与自动续训](reproduction/english_suite/RESULTS.md)。

[已跑实验状态标记](reproduction/english_suite/EXPERIMENT_STATUS.md) · [声学前端差异](reproduction/english_suite/FRONTENDS.md)。

CIF 音素分段基线：[TIMIT40，CE + quantity，及 CTC / upstream-JSD OTTC 辅助对照](reproduction/cif_timit_20261016/README.md)。

本地英语音素实验：[最新四头训练与 K2-FSA / text-free 评估报告](reproduction/multihead_timit/notion_update/README.md)。
TIMIT 的 CTC、OTTC CE、OTTC 上游 jsd 均已完成 40 轮；报告包含训练 loss 曲线。
后续全套训练：[LibriSpeech 100h / 960h 与 MFA English 调度记录](reproduction/english_suite/README.md)。

Implementation of the **Optimal Temporal Transport Classification (OTTC)** loss introduced in
*A Differentiable Alignment Framework for Sequence-to-Sequence Modeling via Optimal Transport*
(Kaloga et al., 2025). OTTC replaces the non-differentiable alignment surrogate used by CTC with a
1-D optimal transport solver that stays fully differentiable, produces sharper alignments, and
maintains practical training stability on modern GPUs.

Paper: https://arxiv.org/abs/2502.01588

## Contact
Questions are welcome at:

- Yacouba Kaloga (`yacouba.kaloga@idiap.ch`)
- Shashi Kumar (`shashi.kumar@idiap.ch`)

## Contents
- [Overview](#overview)
- [Highlights](#highlights)
- [Getting Started](#getting-started)
- [Configuring Paths](#configuring-paths)
- [Datasets and Pretrained Encoders](#datasets-and-pretrained-encoders)
- [Running Experiments](#running-experiments)
- [Monitoring and Outputs](#monitoring-and-outputs)
- [Reproducing the Paper](#reproducing-the-paper)
- [Repository Layout](#repository-layout)
- [Citation](#citation)
- [License](#license)

## Overview
OTTC formulates alignment between acoustic features and target symbols as a differentiable optimal
transport problem. The project provides:

- Reference PyTorch implementations of OTTC, soft-DTW, and CTC heads on top of Wav2Vec2-style
  encoders.
- Training and evaluation pipelines for speech recognition benchmarks (LibriSpeech, Multilingual
  LibriSpeech, AMI, and TIMIT) using the Hugging Face ecosystem.
- Utilities to build vocabularies, collate speech batches, and compute WER for evaluation.

## Highlights
- **Differentiable alignments:** Fast batched 1-D optimal transport with cross-entropy, Euclidean,
  soft-Euclidean, or Jensen–Shannon costs (see `ottc/ops_loss/ot_loss.py`).
- **Baseline losses:** Drop-in CTC and soft-DTW heads to benchmark against OTTC.
- **Encoder flexibility:** Works with Hugging Face checkpoints such as Wav2Vec2-Large-LV60,
  XLS-R, Whisper-Small, etc. (`--encoder_name`).
- **Dataset tooling:** Ready-to-use loaders, collators, and vocab builders for LibriSpeech,
  Multilingual LibriSpeech, AMI, and TIMIT (letters, phonemes, fused phonemes).
- **Resumable training:** Automatic checkpoint discovery when `--continue_if_exist` is enabled,
  plus evaluation-only runs for frozen models.

## Getting Started
### 1. Clone the repository
```bash
git clone https://github.com/idiap/OTTC.git
cd OTTC
```

### 2. Create a Python environment
The recommended setup is via Conda (CUDA 12.1, PyTorch 2.3):
```bash
conda env create -f env.yaml
conda activate ottcenv
```

Alternatively, create a virtualenv and install dependencies from `requirements.txt` (ensure that a
CUDA-enabled PyTorch build >= 2.3.0 is already installed):
```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Make the package importable
Running the scripts from the repository root automatically exposes the local `ottc` package.
If you need to launch Python from elsewhere, export `PYTHONPATH=$(pwd)` (or the absolute path to
this repository) before invoking the scripts.

## Configuring Paths
All data locations live in `ottc/config/path_config.py`. A ready-to-fill template with placeholder
paths is available at `ottc/config/path_config_to_complete.py`—copy it (or edit the existing
`path_config.py`) and update the values before running experiments:

```python
from pathlib import Path

DATASETS_ROOT_PATH = Path('/your/datasets/root')
HUGGINGFACE_CACHE = Path('/your/hf_cache')
LARGE_MODELS_PATH = Path('/your/pretrained/models')
TIMIT_DATASET_PATH = Path('/your/timit/root')  # optional override when TIMIT lives elsewhere
```

These directories host:
- extracted datasets (LibriSpeech, AMI, etc.)
- the Hugging Face cache for dataset + model downloads
- pretrained encoders and experiment outputs (`large_models_results/`). Common Hugging Face
  checkpoints (e.g., `facebook/wav2vec2-large-lv60`, `facebook/wav2vec2-large-xlsr-53`,
  `openai/whisper-small`) are downloaded automatically the first time they are requested via
  `--encoder_name` and cached under `${LARGE_MODELS_PATH}` for reuse.

Make sure the parent folders exist and are writable. If you work on a shared cluster, you can point
these variables to scratch volumes and export environment variables such as
`export LARGE_MODELS_PATH=/scratch/$USER/large_models` before launching jobs.

## Datasets and Pretrained Encoders
- **LibriSpeech / Multilingual LibriSpeech:** downloaded via 🤗 Datasets. The loader expects
  `librispeech_asr` and `facebook/multilingual_librispeech` under `DATASETS_ROOT_PATH` and builds
  letter vocabularies automatically.
- **AMI:** requires manual download of the AMI IHM files into
  `${DATASETS_ROOT_PATH}/edinburghcstr/ami` (see `ottc/tools/dataset_loader/ami.py`).
- **TIMIT:** requires a licensed copy. Download it from https://catalog.ldc.upenn.edu/LDC93S1,
  extract the archive into `${DATASETS_ROOT_PATH}/timit` (or point `TIMIT_DATASET_PATH` to the
  extraction root), and ensure the directory is readable by your jobs.
- **Pretrained encoders:** place checkpoints such as `wav2vec2-large-lv60` inside
  `${LARGE_MODELS_PATH}`. During training OTTC writes checkpoints and TensorBoard logs under
  `${LARGE_MODELS_PATH}/large_models_results/<encoder>/<suffix>/`.

Vocabulary files for each dataset live in `vocab/` and are created the first time you run an
experiment.

## Running Experiments
`run_exp.py` is the main entry point for training and evaluation. The minimum command looks like:

```bash
python run_exp.py \
  --dataset librispeech_asr \
  --train_name train.100 \
  --test_name test \
  --loss_type ottc \
  --encoder_name wav2vec2-large-lv60 \
  --suffix libri_ottc_demo
```

Key arguments:
- `--dataset` selects the loader from `DATASET_LOADERS`. Available options include
  `librispeech_asr`, `ami`, `timit_letter`, `timit_phoneme`, `timit_fused_phoneme`, and
  `multilingual_librispeech_<lang>` with `<lang> ∈ {en, fr, it, de, es}`.
- `--train_name` / `--test_name` choose the split names used inside 🤗 datasets (for LibriSpeech, e.g.
  `train.100`, `train.360`, `train.960`, `test`, `validation`).
- `--loss_type {ctc,ottc,softdtw,softdtweuc,...}` toggles the output head; OTTC-specific options are
  controlled by `--bins_metric {crossentropy,euclidian,ag,jsd,jsd_standard,kl,kl_reverse,jdiv}` and
  `--forward_type {regular,alternate_scheme,alternate_scheme2,regfrozen}`.
- `--prepare_dataset` points to a function in `ottc/utils/prepare_dataset.py`; common choices are
  `prepare_dataset`, `prepare_dataset_border`, or `prepare_dataset_breakdoubleletter` for breaking
  double letters when using OTTC.
- `--frozen_epochs_for_regfrozen` sets the number of epochs the alignment stays frozen when using the
  `regfrozen` forward pass.
- `--continue_if_exist true` resumes from the most recent checkpoint; combine with
  `--only_evaluate true` to compute WER without further training.
- `--batch_size`, `--learning_rate`, `--num_train_epochs`, `--weight_decay`, and
  `--dataloader_num_workers` behave as in Hugging Face `TrainingArguments`.

OTTC divergence costs use natural logarithms and sum class contributions before
weighting by the transport plan. With `p=softmax(logits)`:

| `--bins_metric` | Cost |
| --- | --- |
| `ag` or legacy `jsd` | Upstream arithmetic-geometric divergence: `.5 KL(m||p) + .5 KL(m||q_old)`, `m=(p+q_old)/2` |
| `jsd_standard` | Standard Jensen-Shannon: `.5 KL(p||m) + .5 KL(q||m)`, `m=(p+q)/2` |
| `kl` | `KL(p||q_eps)` (prediction to target) |
| `kl_reverse` | `KL(q_eps||p)` (target to prediction) |
| `jdiv` | Jeffreys J-divergence: `KL(p||q_eps) + KL(q_eps||p)`, with no factor of one half |

For KL/J, `q_eps = normalize(clamp(q, min=1e-12))`; smoothing is necessary
because prediction-to-one-hot-target KL is otherwise infinite. New KL/J costs
use log-softmax and accumulate in float32 for fp16/bfloat16 inputs (float64 is
preserved). The epsilon is currently fixed at `1e-12` in the training entrypoint;
standalone helpers accept `eps=`. `kl_reverse` is approximately cross-entropy
minus target entropy; for unsmoothed one-hot targets it would equal CE.

`ag` deliberately retains the exact upstream perturbation
`q_old = q + (1-q)*1e-12 - q*1e-12` without renormalization. It is an explicit
alias for the existing `jsd` behavior, including its numerical limitations;
old checkpoints/configs remain interpretable. Existing runs named `jsd` must
not be relabeled as standard JS experiments. These definitions follow
[Taneja, equations 1.7–1.14](https://rgmia.org/papers/v7n4/Gsdmi_RGMIA.pdf).

Select the new costs with `--bins_metric kl`, `--bins_metric kl_reverse`, or
`--bins_metric jdiv`. At the loss API, use
`batched_ottc_loss_bucketized(..., divergence='kl')` (also supported by the
concatenated implementation). Explicit `divergence` cannot be combined with
`jsd` or `euclidian`; defaults and historical `jsd=True` calls are unchanged.

Example workflows:

```bash
# CTC baseline on LibriSpeech 100h


# OTTC with reg-frozen alignment on TIMIT fused phonemes
python run_exp.py \
  --dataset timit_fused_phoneme \
  --train_name train \
  --test_name kaldi_test \
  --loss_type ottc \
  --prepare_dataset prepare_dataset_for_timit_phoneme \
  --forward_type regfrozen \
  --frozen_epochs_for_regfrozen 30 \
  --suffix timit_ottc_fused

# Evaluate the latest checkpoint only
python run_exp.py --only_evaluate true --continue_if_exist true --suffix timit_ottc_fused
```

Inspect all options with `python run_exp.py --help`.

## Monitoring and Outputs
- **Checkpoints:** stored in `${LARGE_MODELS_PATH}/large_models_results/<encoder>/<suffix>/`.
- **TensorBoard:** logs are written alongside checkpoints; launch `tensorboard --logdir` on the same
  directory to monitor training and alignment plots.
- **WER computation:** handled by `ottc/utils/eval_metrics.py` using the 🤗 `wer` metric during
  evaluation. To reproduce the LibriSpeech WER experiment reported in the paper once the OTTC model
  is trained, run:

```bash
python run_exp.py \
  --dataset librispeech_asr \
  --train_name train.100 \
  --test_name test \
  --loss_type ottc \
  --suffix libri_ottc_100h \
  --only_evaluate true \
  --continue_if_exist true
```

## Reproducing the Paper
To reproduce experiments reported in the OTTC paper, adapt the following templates to your local
paths:

```bash
# LibriSpeech (100h ctc)
python run_exp.py \
  --dataset librispeech_asr \
  --train_name train.100 \
  --test_name test \
  --loss_type ctc \
  --prepare_dataset prepare_dataset \
  --batch_size 64 \
  --learning_rate 5e-5 \
  --num_train_epochs 40 \
  --force_lowercase False \
  --suffix runexp_ctc_libri_100h

# LibriSpeech (100h ottc)
python run_exp.py \
  --dataset librispeech_asr \
  --train_name train.100 \
  --test_name test \
  --loss_type ottc \
  --forward_type regfrozen \
  --frozen_epochs_for_regfrozen 30 \
  --prepare_dataset prepare_dataset_breakdoubleletter \
  --batch_size 64 \
  --learning_rate 2e-4 \
  --num_train_epochs 40 \
  --suffix runexp_ottc_libri_100h_rf

# TIMIT (fused phonemes ctc)
python run_exp.py \
  --dataset timit_fused_phoneme \
  --train_name train \
  --test_name kaldi_test \
  --loss_type ctc \
  --prepare_dataset prepare_dataset_for_timit_phoneme \
  --batch_size 64 \
  --learning_rate 5e-5 \
  --num_train_epochs 40 \
  --force_lowercase True \
  --suffix runexp_ctc_fuse_phoneme

# TIMIT baseline (fused phonemes ottc)
python run_exp.py \
  --dataset timit_fused_phoneme \
  --train_name train \
  --test_name kaldi_test \
  --loss_type ottc \
  --forward_type regfrozen \
  --frozen_epochs_for_regfrozen 30 \
  --prepare_dataset prepare_dataset_for_timit_phoneme \
  --batch_size 64 \
  --learning_rate 2e-4 \
  --num_train_epochs 40 \
  --suffix runexp_ottc_fuse_phoneme
```

AMI reproduction currently depends on an internal tokenizer variant and is not bundled in this
release. Support will be added in a future update—coming soon alongside alignment-metric recipes.

## Repository Layout
```text
OTTC/
├── run_exp.py                    # Main training / evaluation script
├── run_ottc.py                   # Legacy research playground (older API)
├── ottc/
│   ├── config/path_config.py     # Global paths for datasets, cache, pretrained models
│   ├── models/sequence/          # Wav2Vec2 heads for CTC and OTTC
│   ├── ops_loss/                 # Optimal transport + soft-DTW losses
│   ├── tools/                    # Dataset loaders, collators, vocab builder
│   └── utils/                    # Metrics, dataset preprocessing helpers
├── vocab/                        # Auto-generated vocabularies per dataset
├── requirements.txt              # Python dependency list
└── xfs.yaml                      # Conda environment specification (CUDA 12.1, PyTorch 2.2)
```

## Citation
If you use OTTC in your work, please cite:

```
@inproceedings{kaloga2025ottc,
  title     = {A Differentiable Alignment Framework for Sequence-to-Sequence Modeling via Optimal Transport},
  author    = {Kaloga, Yacouba and 
               Shashi, Kumar and
               Kodrasi, Ina and 
               Moltieck, Petr},
  year      = {2025},
  eprint    = {2502.01588},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CL}
}
```

## License
This project is released under the MIT License. See the `LICENSE` file for details. Third-party
libraries retain their respective licenses (Transformers: Apache 2.0, torchaudio: BSD, POT: MIT).
