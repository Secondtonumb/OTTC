# SPDX-FileCopyrightText: 2024 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Kaloga Yacouba yacouba.kaloga@idiap.ch
# SPDX-FileContributor: Shashi Kumar shashi.kumar@idiap.ch
#
# SPDX-License-Identifier: MIT

from typing import List, Dict, Tuple, Any
import os
import json
from typing import Dict, Any
from datasets import load_dataset, load_metric
from ottc.config.path_config import (
    DATASETS_ROOT_PATH,
    HUGGINGFACE_CACHE,
)
from ottc.config import path_config
TIMIT_DATASET_PATH = getattr(path_config, 'TIMIT_DATASET_PATH', None)
from transformers import Wav2Vec2CTCTokenizer, Wav2Vec2PhonemeCTCTokenizer, PreTrainedTokenizerFast
import datasets
from ottc.utils.utils import get_vocab_from_text, get_vocab_from_phoneme
from ottc.utils.utils import remove_special_characters
from ottc.utils import prepare_dataset as prepare_dataset_utils
from transformers import Wav2Vec2FeatureExtractor
from transformers import Wav2Vec2Processor
from transformers import Wav2Vec2CTCTokenizer, Wav2Vec2PhonemeCTCTokenizer, PreTrainedTokenizerFast
from pathlib import Path

timit_phoneme_fusion_map = { # source: https://www.intechopen.com/chapters/15948
    "aa": "aa", "ao": "aa", "ah": "ah", "ax": "ah", "ax-h": "ah",
    "er": "er", "axr": "er", "hh": "hh", "hv": "hh", "ih": "ih",
    "ix": "ih", "l": "l", "el": "l", "m": "m", "em": "m",
    "n": "n", "en": "n", "nx": "n", "ng": "ng", "eng": "ng",
    "sh": "sh", "zh": "sh", "uw": "uw", "ux": "uw",
    "pcl": "sil", "tcl": "sil", "kcl": "sil", "bcl": "sil",
    "dcl": "sil", "gcl": "sil", "h#": "sil", "pau": "sil", "epi": "sil"
}


def phoneme_process(batch):
    batch['phonemes'] = batch['phonetic_detail']['utterance']
    batch['phonemes_unique'] = list(set(batch['phonetic_detail']['utterance']))
    return batch


def merge_intervals(input_vec, starting_time_vec, ending_time_vec, phoneme_map):
    # Step 1: Apply the mapping
    output_vec = [phoneme_map[ph] if ph in phoneme_map else ph for ph in input_vec]
    
    # Step 2: Merge consecutive duplicates
    merged_output = []
    merged_start_times = []
    merged_end_times = []
    
    prev_phoneme = None
    for i, ph in enumerate(output_vec):
        if ph != prev_phoneme:  # New phoneme detected (not duplicate)
            merged_output.append(ph)
            merged_start_times.append(starting_time_vec[i])  # Start of new phoneme
            if prev_phoneme is not None:  # Close previous interval
                merged_end_times.append(ending_time_vec[i-1])
        prev_phoneme = ph
    
    # Close last interval
    if merged_output:
        merged_end_times.append(ending_time_vec[-1])
    
    return merged_output, merged_start_times, merged_end_times


from typing import List, Dict, Tuple

def merge_intervals(
    input_vec: List[str], 
    starting_time_vec: List[float], 
    ending_time_vec: List[float], 
    phoneme_map: Dict[str, str]
) -> Tuple[List[str], List[float], List[float]]:
    """
    Merges consecutive identical phonemes in a sequence while preserving time intervals.

    This function first applies a phoneme mapping to the input sequence and then merges 
    consecutive duplicate phonemes, updating their start and end times accordingly.

    Args:
        input_vec (List[str]): List of phonemes or symbols representing a sequence.
        starting_time_vec (List[float]): List of start times corresponding to `input_vec`.
        ending_time_vec (List[float]): List of end times corresponding to `input_vec`.
        phoneme_map (Dict[str, str]): Mapping of phonemes to their corresponding replacements.

    Returns:
        Tuple[List[str], List[float], List[float]]: 
            - Merged phoneme sequence with consecutive duplicates removed.
            - Updated list of start times.
            - Updated list of end times.
    """

    # Step 1: Apply phoneme mapping
    mapped_vec: List[str] = [phoneme_map.get(ph, ph) for ph in input_vec]

    # Step 2: Merge consecutive duplicates
    merged_output: List[str] = []
    merged_start_times: List[float] = []
    merged_end_times: List[float] = []

    prev_phoneme: str = None

    for i, phoneme in enumerate(mapped_vec):
        if phoneme != prev_phoneme:  
            # A new phoneme detected (not a duplicate of the previous one)
            merged_output.append(phoneme)
            merged_start_times.append(starting_time_vec[i])  # Start time of the new phoneme
            if prev_phoneme is not None:  
                # Close previous phoneme's interval
                merged_end_times.append(ending_time_vec[i - 1])
        prev_phoneme = phoneme

    # Step 3: Close the last phoneme interval
    if merged_output:
        merged_end_times.append(ending_time_vec[-1])

    return merged_output, merged_start_times, merged_end_times

def fusephoneme_process(batch: Dict[str, Any]) -> Dict[str, Any]:
    """
    Processes phoneme details by merging consecutive identical phonemes.

    This function extracts phonetic details from the input batch, applies phoneme merging 
    using `merge_intervals`, and updates the batch with the processed phoneme sequence, 
    start times, and stop times.

    Args:
        batch (Dict[str, Any]): A dictionary containing phonetic details, expected to have:
            - 'phonetic_detail': A nested dictionary with:
                - 'utterance': List of phonemes.
                - 'start': List of start times.
                - 'stop': List of stop times.

    Returns:
        Dict[str, Any]: Updated batch with:
            - 'phonetic_detail' containing merged phonemes, start times, and stop times.
            - 'phonemes': Merged phoneme sequence.
            - 'phonemes_unique': Unique phonemes in the utterance.
    """

    # Step 1: Extract phonetic details from batch
    utterance_list = list(batch['phonetic_detail']['utterance'])
    start_times = batch['phonetic_detail']['start']
    stop_times = batch['phonetic_detail']['stop']

    # Step 2: Merge consecutive phonemes
    merged_output, merged_start_times, merged_end_times = merge_intervals(
        utterance_list, start_times, stop_times, timit_phoneme_fusion_map
    )

    # Step 3: Update batch with merged phoneme details
    batch['phonetic_detail']['utterance'] = merged_output
    batch['phonetic_detail']['start'] = merged_start_times
    batch['phonetic_detail']['stop'] = merged_end_times

    # Step 4: Store phonemes and unique phonemes
    batch['phonemes'] = merged_output
    batch['phonemes_unique'] = list(set(merged_output))

    return batch

def load_timit(
    train_name: str,
    test_name: str,
    dataset_name: str,
    prepare_dataset: str = None,
    force_lowercase: bool = False,
    num_proc: int = 15,
) -> Tuple[datasets.Dataset, Wav2Vec2Processor]:  
        timit_root = TIMIT_DATASET_PATH or (DATASETS_ROOT_PATH / 'timit')
        dataset_ = datasets.load_dataset(
            './ottc/tools/dataset_loader/custom_huggingface_loader/timit_custom_loader.py',
            data_dir=str(timit_root),
            cache_dir=HUGGINGFACE_CACHE,
            trust_remote_code=True,
        )
        if test_name == 'kaldi_test':
            # if kaldi test is used"
            test_speaker_id = ["mdab0","mwbt0","felc0","mtas1","mwew0","fpas0","mjmp0","mlnt0","fpkt0","mlll0","mtls0","fjlm0","mbpm0","mklt0","fnlp0","mcmj0","mjdh0","fmgd0","mgrt0","mnjm0","fdhc0","mjln0","mpam0","fmld0"]#https://github.com/kaldi-asr/kaldi/blob/master/egs/timit/s5/conf/test_spk.list
            dataset_['kaldi_test'] = dataset_['test'].filter(lambda x: x['speaker_id'] in test_speaker_id)
        dataset_ = dataset_.map(
            remove_special_characters,
            fn_kwargs={"transcript_key": "text", "force_lowercase": force_lowercase},
        )

        os.makedirs('./vocab', exist_ok=True)
        if 'timit_letter' in dataset_name:
            vocab_path = Path(f'./vocab/vocab_timit_letter.json')
            if not os.path.exists(vocab_path):
                vocab = get_vocab_from_text(
                    dataset_,
                    splits_name = ['train', 'test'],
                    columns_to_remove = ["audio"],
                    transcript_key='text',
                )
                with open(vocab_path, 'w') as fp:
                    json.dump(vocab, fp)
            else:
                with open(vocab_path, 'r') as fp:
                    vocab = json.load(fp)
            tokenizer = Wav2Vec2CTCTokenizer(vocab_path)

        elif 'timit_phoneme' in dataset_name:
            dataset_ = dataset_.map(phoneme_process)
            vocab_path = Path(f'./vocab/vocab_timit_phoneme.json')
            if not os.path.exists(vocab_path):
                vocab = get_vocab_from_phoneme(dataset_,  splits_name = ['train', 'test'], columns_to_remove = ["audio"])
                with open(vocab_path, 'w') as fp:
                    json.dump(vocab, fp)
            else:
                with open(vocab_path, 'r') as fp:
                    vocab = json.load(fp)
            tokenizer = Wav2Vec2PhonemeCTCTokenizer(vocab_path, do_phonemize=False)

        elif 'timit_fused_phoneme' in dataset_name: 
            dataset_ = dataset_.map(fusephoneme_process)
            vocab_path = Path(f'./vocab/vocab_timitfuse_phoneme.json')
            if not os.path.exists(vocab_path):
                vocab = get_vocab_from_phoneme(dataset_,  splits_name = ['train', 'test'], columns_to_remove = ["audio"])
                with open(vocab_path, 'w') as fp:
                    json.dump(vocab, fp)
            else:
                with open(vocab_path, 'r') as fp:
                    vocab = json.load(fp)
            tokenizer = Wav2Vec2PhonemeCTCTokenizer(vocab_path, do_phonemize=False)

        else:
            raise ValueError('The dataset must be timit_letter or timit_phoneme')

        feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0, do_normalize=True, return_attention_mask=True, padding_side= "right")
        processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)
        prepare_fn_name = prepare_dataset
        fn_kwargs = {"processor": processor}
        if 'timit_letter' in dataset_name:
            prepare_fn_name = prepare_fn_name or 'prepare_dataset'
            fn_kwargs["transcript"] = 'text'
        else:
            prepare_fn_name = prepare_fn_name or 'prepare_dataset_for_timit_phoneme'

        prepare_dataset_fn = getattr(prepare_dataset_utils, prepare_fn_name, None)
        if prepare_dataset_fn is None:
            raise KeyError(f"Unknown prepare_dataset function '{prepare_fn_name}' for TIMIT.")

        dataset_ = dataset_.map(
            prepare_dataset_fn,
            remove_columns=dataset_.column_names["test"],
            fn_kwargs=fn_kwargs,
            num_proc=num_proc,
        )
        return dataset_, vocab, processor
