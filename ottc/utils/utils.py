# SPDX-FileCopyrightText: 2024 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Kaloga Yacouba yacouba.kaloga@idiap.ch
# SPDX-FileContributor: Shashi Kumar shashi.kumar@idiap.ch
#
# SPDX-License-Identifier: MIT

from datasets import Dataset
from datasets import DatasetDict

import random
import re
import pandas as pd
from typing import Any, Dict, List, Set, Tuple

from ottc.config.constants import BORDERING_TOKEN, SPLITTING_DOUBLE_LETTER_TOKEN, PIPE_TOKEN

def add_token_to_vocab(vocab: Dict[str, int], token: Tuple[str, int]) -> Dict[str, int]:
    """
    Insert or reposition a character/index entry while keeping indices unique.

    The function mutates `vocab` in-place:
      * If the character already exists, it keeps the current mapping unless the requested
        index is already taken, in which case it swaps indices with the occupant.
      * If the character is new but the index is already assigned, the previous occupant
        is moved to the next free index and the character takes the requested slot.

    Args:
        vocab (Dict[str, int]): Mapping from characters to indices.
        token (Tuple[str, int]): Pair containing the character and desired index.

    Returns:
        Dict[str, int]: The updated vocabulary reference.
    """
    character, index = token
    if character in vocab:
        if index in vocab.values():
            previous_character = next((k for k, v in vocab.items() if v == index))
            previous_index = vocab[character]
            vocab[character] = index
            vocab[previous_character] = previous_index
        return vocab
        

    if index in vocab.values():
        previous_character = next((k for k, v in vocab.items() if v == index))
        vocab[previous_character] = len(vocab)

    vocab[character] = index
    return vocab

    
def remove_special_characters(batch: Dict[str, Any], transcript_key: str, force_lowercase: bool = False) -> Dict[str, Any]:
    """
    Removes special characters from the transcript and optionally converts it to lowercase.

    Args:
        batch (Dict[str, Any]): A dictionary containing the 'transcript' key.
        transcript_key (str): The key for the transcript text in the batch.
        force_lowercase (bool, optional): If True, converts the transcript to lowercase. Defaults to False.

    Returns:
        Dict[str, Any]: The batch with the cleaned transcript.
    """
    
    # Define regex pattern to match special characters
    chars_to_ignore_regex = '[,?.!-;:"*]'
    
    # Remove special characters and optionally convert to lowercase
    if force_lowercase:
        batch[transcript_key] = re.sub(chars_to_ignore_regex, '', batch[transcript_key]).lower()
    else:
        batch[transcript_key] = re.sub(chars_to_ignore_regex, '', batch[transcript_key])
    
    return batch

def show_random_elements(dataset: Dataset, num_examples: int = 10) -> None:
    """
    Displays a random subset of elements from the given dataset as an HTML table.

    Args:
        dataset (List[Dict[str, Any]]): The dataset from which random elements are to be selected. 
            It is assumed to be a list of dictionaries, where each dictionary represents a data entry.
        num_examples (int, optional): The number of random examples to show. Defaults to 10.

    Raises:
        AssertionError: If `num_examples` is greater than the number of elements in `dataset`.
    """
    
    from IPython.display import display, HTML
    # Check if num_examples is valid
    assert num_examples <= len(dataset), "Can't pick more elements than there are in the dataset."

    # Randomly select num_examples unique entries
    picks = random.sample(range(len(dataset)), num_examples)

    # Convert selected data into a DataFrame and display it
    df = pd.DataFrame(dataset[i] for i in picks)
    display(HTML(df.to_html()))

def get_vocab_from_text(
    data_hf: DatasetDict,
    splits_name: List[str] = ["train", "test"],
    columns_to_remove: List[str] = ["file", "audio"],
    transcript_key: str = "transcript",
    add_split_tokens: bool = True,
) -> Dict[str, int]:
    """
    Extracts a vocabulary from the text transcripts of a Hugging Face Dataset.

    Args:
        data_hf (DatasetDict): The Hugging Face Dataset containing the data.
        splits_name (List[str], optional): Dataset splits to extract data from.
        columns_to_remove (List[str], optional): Columns to remove before processing.
        transcript_key (str, optional): Column containing text transcripts.
        pipe_number (int, optional): Target index for the pipe token "|".

    Returns:
        Dict[str, int]: A dictionary mapping characters to their corresponding indices.
    """
    # Extract all unique characters across the specified dataset splits
    all_chars: Set[str] = set()
    for split_name in splits_name:
        transcripts = [
            temp[transcript_key]
            for temp in data_hf[split_name].remove_columns(columns_to_remove)
        ]
        all_chars.update(set(" ".join(transcripts)))

    # Sort and assign indices
    sorted_chars = sorted(all_chars)
    vocab_dict: Dict[str, int] = {char: idx for idx, char in enumerate(sorted_chars)}

    if " " in vocab_dict:
        vocab_dict["|"] = vocab_dict.pop(" ")

    # Add toher special tokens
    # vocab_dict = add_token_to_vocab(vocab_dict, SPLITTING_DOUBLE_LETTER_TOKEN)
    # vocab_dict = add_token_to_vocab(vocab_dict, BORDERING_TOKEN)

    vocab_dict = add_token_to_vocab(vocab_dict, ("<pad>", 0))
    vocab_dict = add_token_to_vocab(vocab_dict, ("<unk>", len(vocab_dict)))

    #huggingface tokenizer recognizes "|" as space -at both encoding and decoding time -
    vocab_dict = add_token_to_vocab(vocab_dict, PIPE_TOKEN) # to change the index of the pipe token

    if " " in vocab_dict:
        del vocab_dict[" "]
    return vocab_dict 

def get_vocab_from_phoneme(
    data_hf: DatasetDict,
    splits_name: List[str] = ["train", "test"],
    columns_to_remove: List[str] = ["file", "audio"],
    phoneme_column: str = "phonemes_unique"
) -> Dict[str, int]:
    """
    Extracts a vocabulary from phoneme sequences in a Hugging Face Dataset.

    Args:
        data_hf (DatasetDict): The Hugging Face Dataset containing the phoneme data.
        splits_name (List[str], optional): Dataset splits to extract phonemes from.
        columns_to_remove (List[str], optional): Columns to remove before processing.
        phoneme_column (str, optional): Column containing phoneme sequences.

    Returns:
        Dict[str, int]: A dictionary mapping phonemes to their corresponding indices.
    """
    print("Extracting vocabulary from phoneme data...")

    # Extract all unique phonemes across the specified dataset splits
    all_phonemes: Set[str] = set()
    for split_name in splits_name:
        phoneme_lists = data_hf[split_name].remove_columns(columns_to_remove)[phoneme_column]
        all_phonemes.update(phoneme for phonemes in phoneme_lists for phoneme in phonemes)

    # Sort and assign indices
    sorted_phonemes = sorted(list(set(all_phonemes)))
    vocab_dict: Dict[str, int] = {phoneme: idx for idx, phoneme in enumerate(sorted_phonemes)}


    # vocab_dict = add_token_to_vocab(vocab_dict, BORDERING_TOKEN)
    vocab_dict = add_token_to_vocab(vocab_dict, ("<pad>", 0))
    vocab_dict = add_token_to_vocab(vocab_dict, ("<unk>", len(vocab_dict)))
    
    return vocab_dict
