# SPDX-FileCopyrightText: 2024 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Kaloga Yacouba yacouba.kaloga@idiap.ch
# SPDX-FileContributor: Shashi Kumar shashi.kumar@idiap.ch
#
# SPDX-License-Identifier: MIT

import re
import os
import json
from typing import Dict, Any, Tuple
from datasets import load_dataset, load_metric
from pathlib import Path
from ottc.config.path_config import DATASETS_ROOT_PATH, HUGGINGFACE_CACHE, LARGE_MODELS_PATH
from transformers import Wav2Vec2FeatureExtractor
from transformers import Wav2Vec2CTCTokenizer, Wav2Vec2PhonemeCTCTokenizer, PreTrainedTokenizerFast
from transformers import Wav2Vec2Processor, WhisperProcessor
from transformers import Wav2Vec2CTCTokenizer, WhisperTokenizer, Wav2Vec2PhonemeCTCTokenizer, PreTrainedTokenizerFast
import datasets
from ottc.utils.utils import get_vocab_from_text
from ottc.utils import prepare_dataset as prepare_dataset_utils
from ottc.utils.utils import remove_special_characters

def load_librispeech(
    train_name: str,
    test_name: str,
    prepare_dataset: str = 'prepare_dataset',
    force_lowercase: bool = False,
    num_proc: int = 15,
):
        dataset_ = load_dataset("librispeech_asr", "clean",  data_dir=DATASETS_ROOT_PATH/'librispeech_asr', cache_dir=HUGGINGFACE_CACHE)
        if train_name == 'train.960':
            dataset_temp = load_dataset("librispeech_asr", "other",  data_dir=DATASETS_ROOT_PATH/'librispeech_asr', cache_dir=HUGGINGFACE_CACHE)
            dataset_ = datasets.DatasetDict({"train.100":dataset_['train.100'],
                                             "train.360":dataset_['train.360'],
                                             "train.960":datasets.concatenate_datasets([dataset_['train.100'], dataset_['train.360'], dataset_temp['train.500']]), 
                                             "validation":dataset_['validation'], 
                                             "test":dataset_['test'], 
                                             "validation.other":dataset_temp['validation'], 
                                             "test.other":dataset_temp['test']})
        dataset_ = dataset_.map(
        remove_special_characters,
        fn_kwargs={"transcript_key": "text", "force_lowercase": force_lowercase},
        )

        suffix = 'forcelowercase' if force_lowercase else ''

        os.makedirs(Path(f'./vocab'), exist_ok=True)
        vocab_path = Path(f'./vocab/vocab_libri_{suffix}.json')
        if not os.path.exists(vocab_path):
            print(f'Create vocab_libri_{suffix}.json')
            vocab =  get_vocab_from_text(dataset_, splits_name = ['train.100', 'train.360', 'test'], columns_to_remove = ["file", "audio"], transcript_key ='text')
            with open(vocab_path, 'w') as fp:
                json.dump(vocab, fp)
        else:
            with open(vocab_path, 'r') as fp:
                vocab = json.load(fp)

        # build processor
        tokenizer = Wav2Vec2CTCTokenizer(vocab_path)
        feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0, do_normalize=True, return_attention_mask=True, padding_side= "right")
        processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)

        prepare_dataset_fn = getattr(prepare_dataset_utils, prepare_dataset, None)
        if prepare_dataset_fn is None:
            raise KeyError(f"Unknown prepare_dataset function '{prepare_dataset}'")
        dataset_ = dataset_.map(
            prepare_dataset_fn,
            remove_columns=dataset_.column_names["test"],
            fn_kwargs={"processor": processor, "transcript": 'text'},
            num_proc=num_proc,
        )
        return dataset_, vocab, processor


def load_librispeech_multilingual(
    language: str,
    train_name: str,
    test_name: str,
    prepare_dataset: str = 'prepare_dataset',
    force_lowercase: bool = False,
    num_proc: int = 15,
):
        dataset_ = load_dataset("facebook/multilingual_librispeech", language,  cache_dir=HUGGINGFACE_CACHE)
        dataset_ = dataset_.map(
        remove_special_characters,
        fn_kwargs={"transcript_key": "text", "force_lowercase": force_lowercase},
        )

        suffix = f'{language}_forcelowercase' if force_lowercase else f'{language}'

        os.makedirs('./vocab', exist_ok=True)
        vocab_path = Path(f'./vocab/vocab_multilibri_{suffix}.json')
        if not os.path.exists(vocab_path):
            print(f'Create vocab_multilibri_{suffix}.json')
            vocab = get_vocab_from_text(
                dataset_,
                splits_name=['train', 'test'],
                columns_to_remove=["audio"],
                transcript_key='text',
                pipe_number=SPLITTING_TOKEN,
            )
            with open(vocab_path, 'w') as fp:
                json.dump(vocab, fp)
        else:
            with open(vocab_path, 'r') as fp:
                vocab = json.load(fp)

        # build processor
        tokenizer = Wav2Vec2CTCTokenizer(vocab_path)
        feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0, do_normalize=True, return_attention_mask=True, padding_side= "right")
        processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)

        prepare_dataset_fn = getattr(prepare_dataset_utils, prepare_dataset, None)
        if prepare_dataset_fn is None:
            raise KeyError(f"Unknown prepare_dataset function '{prepare_dataset}'")
        dataset_ = dataset_.map(
            prepare_dataset_fn,
            remove_columns=dataset_.column_names["test"],
            fn_kwargs={"processor": processor, "transcript": 'text'},
            num_proc=num_proc,
        )
        return dataset_, vocab, processor
