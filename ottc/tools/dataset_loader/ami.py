# SPDX-FileCopyrightText: 2024 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Kaloga Yacouba yacouba.kaloga@idiap.ch
# SPDX-FileContributor: Shashi Kumar shashi.kumar@idiap.ch
#
# SPDX-License-Identifier: MIT

import re
import json
import os
from typing import Dict, Any
from datasets import load_dataset, load_metric
from ottc.config.path_config import DATASETS_ROOT_PATH, HUGGINGFACE_CACHE, LARGE_MODELS_PATH
from transformers import Wav2Vec2FeatureExtractor
from transformers import Wav2Vec2CTCTokenizer
from transformers import Wav2Vec2Processor
from transformers import Wav2Vec2CTCTokenizer
import datasets
from ottc.utils.utils import get_vocab_from_text
from ottc.utils.prepare_dataset import prepare_dataset
from ottc.utils.prepare_dataset import prepare_dataset_breakdoubleletter
from ottc.utils.utils import remove_special_characters

from pathlib import Path


def load_ami(
    train_name: str,
    test_name: str,
    prepare_dataset: str = 'prepare_dataset',
    force_lowercase: bool = False,
    num_proc: int = 15,
):
    dataset_ = load_dataset("edinburghcstr/ami", "ihm",  data_dir=DATASETS_ROOT_PATH/'edinburghcstr', cache_dir=HUGGINGFACE_CACHE)
    dataset_ = dataset_.filter(lambda x: x['end_time'] - x['begin_time'] > 0.5)
    print('Modifying dataset')
    dataset_ = dataset_.map(
    remove_special_characters,
    fn_kwargs={"transcript_key": "text", "force_lowercase": force_lowercase},
    )

    suffix = 'forcelowercase' if force_lowercase else ''

    os.makedirs(Path(f'./vocab'), exist_ok=True)
    vocab_path = Path(f'./vocab/vocab_edin_{suffix}.json')
    if not os.path.exists(vocab_path):
        vocab = get_vocab_from_text(dataset_, splits_name = ['train', 'test'], columns_to_remove = ["audio"])
        with open(vocab_path, 'w') as fp:
            json.dump(vocab, fp)
    else:
        with open(vocab_path, 'r') as fp:
            vocab = json.load(fp)

    tokenizer = Wav2Vec2CTCTokenizer(vocab_path)
    feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0, do_normalize=True, return_attention_mask=True, padding_side= "right")
    processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)

    prepare_dataset_fn = globals()[prepare_dataset]

    # prepare dataset
    dataset_ = dataset_.map(
        prepare_dataset_fn,
        remove_columns=dataset_.column_names["test"],
        fn_kwargs={"processor": processor, "transcript": 'text'},
        num_proc=num_proc,
    )
    return dataset_, vocab, processor
