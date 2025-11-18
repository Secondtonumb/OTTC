# SPDX-FileCopyrightText: 2024 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Kaloga Yacouba yacouba.kaloga@idiap.ch
# SPDX-FileContributor: Shashi Kumar shashi.kumar@idiap.ch
#
# SPDX-License-Identifier: MIT

from pathlib import Path

"""
Template for repository paths.

Copy this file to `ottc/config/path_config.py` (or edit the existing one) and update the values
to point to your local directories before launching any experiment.
"""

# Absolute path where all speech datasets live (LibriSpeech, AMI, TIMIT, ...).
DATASETS_ROOT_PATH = Path("/absolute/path/to/your/datasets")

# Hugging Face cache directory for datasets and pretrained checkpoints.
HUGGINGFACE_CACHE = Path("/absolute/path/to/your/huggingface/cache")

# Location that stores pretrained encoders and OTTC experiment outputs.
LARGE_MODELS_PATH = Path("/absolute/path/to/your/large_models_root")
