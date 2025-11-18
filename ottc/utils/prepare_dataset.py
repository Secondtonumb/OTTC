# SPDX-FileCopyrightText: 2024 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Kaloga Yacouba yacouba.kaloga@idiap.ch
# SPDX-FileContributor: Shashi Kumar shashi.kumar@idiap.ch
#
# SPDX-License-Identifier: MIT

from typing import Any, Dict, List

from ottc.config.constants import BORDERING_TOKEN, SPLITTING_DOUBLE_LETTER_TOKEN

def insert_blank(lst: List[int], token_id: int = SPLITTING_DOUBLE_LETTER_TOKEN[1]) -> List[int]:
    """
    Inserts a blank (token_id) between each element of the list.

    Args:
        lst (List[int]): The list of integers where blanks will be inserted.
        token_id (int, optional): The token to insert between elements. Defaults to 0.

    Returns:
        List[int]: A new list with blanks inserted.
    """
    new_lst = [token_id]
    for i in lst:
        new_lst.append(i)
        new_lst.append(token_id)
    return new_lst


def insert_blank_inbetween(lst: List[int], token_id: int = SPLITTING_DOUBLE_LETTER_TOKEN[1]) -> List[int]:
    """
    Inserts a token between consecutive duplicate elements in the list.

    Args:
        lst (List[int]): The input list of integers.
        token_id (int, optional): The token to insert between duplicates. Defaults to 0.

    Returns:
        List[int]: The modified list with inserted token(s).
    """
    i = 1  # Start at index 1 since we compare the current element with the previous one
    while i < len(lst):  # Use while loop as the list length changes during iteration
        if lst[i] == lst[i-1]:  # If consecutive elements are the same
            lst.insert(i, token_id)  # Insert the token between them
            i += 1  # Skip the token we just inserted
        i += 1  # Move to the next element
    return lst

def prepare_dataset(batch: Dict[str, Any], processor: Any, transcript: str = "text") -> Dict[str, Any]:
    """
    Prepares the dataset for model input by processing the audio and transcript.

    Args:
        batch (Dict[str, Any]): The input batch containing audio and transcript data.

    Returns:
        Dict[str, Any]: The batch with processed input values and labels.
    """
    audio = batch["audio"]
    batch["input_values"] = processor(audio["array"], sampling_rate=audio["sampling_rate"]).input_values[0]
    batch['length'] = len(batch["input_values"])
    with processor.as_target_processor():
        batch["labels"] = processor(batch[transcript]).input_ids
    return batch

def prepare_dataset_border(batch: Dict[str, Any], processor: Any, transcript: str = "text", border_token: int = BORDERING_TOKEN[1]) -> Dict[str, Any]:
    """
    Prepares the dataset with borders, adding token boundaries to the labels.

    Args:
        batch (Dict[str, Any]): The input batch containing audio and transcript data.

    Returns:
        Dict[str, Any]: The batch with processed input values and bordered labels.
    """
    audio = batch["audio"]
    batch["input_values"] = processor(audio["array"], sampling_rate=audio["sampling_rate"]).input_values[0]
    batch['length'] = len(batch["input_values"])
    with processor.as_target_processor():
        batch["labels"] = [border_token] + processor(batch[transcript]).input_ids + [border_token]
    return batch

def prepare_dataset_for_timit_phoneme(batch: Dict[str, Any], processor: Any) -> Dict[str, Any]:
    """
    Prepares the dataset for TIMIT phoneme processing, flattening phoneme IDs.

    Args:
        batch (Dict[str, Any]): The input batch containing audio and phoneme data.

    Returns:
        Dict[str, Any]: The batch with processed input values and phoneme labels.
    """
    audio = batch["audio"]
    batch["input_values"] = processor(audio["array"], sampling_rate=audio["sampling_rate"]).input_values[0]
    batch['length'] = len(batch["input_values"])
    with processor.as_target_processor():
        batch['labels'] = [item for sublist in processor(batch['phonemes']).input_ids for item in sublist]
    return batch

def prepare_dataset_insert_blank(batch: Dict[str, Any], processor: Any, transcript: str = "text") -> Dict[str, Any]:
    """
    Prepares the dataset with borders and inserts blanks into the labels.

    Args:
        batch (Dict[str, Any]): The input batch containing audio and transcript data.

    Returns:
        Dict[str, Any]: The batch with processed input values, bordered labels, and inserted blanks.
    """
    audio = batch["audio"]
    batch["input_values"] = processor(audio["array"], sampling_rate=audio["sampling_rate"]).input_values[0]
    batch['length'] = len(batch["input_values"])
    with processor.as_target_processor():
        batch["labels"] = processor(batch[transcript]).input_ids
        # batch["labels"] = insert_blank(batch["labels"], token_id=SPLITTING_DOUBLE_LETTER_TOKEN[1])
        token_id = processor.tokenizer.unk_token_id
        batch["labels"] = insert_blank(batch["labels"], token_id=token_id)
    return batch

def prepare_dataset_breakdoubleletter(batch: Dict[str, Any], processor: Any, transcript: str = "text") -> Dict[str, Any]:
    """
    Prepares the dataset by processing audio and transcript, then breaks double letters by inserting a blank token.

    Args:
        batch (Dict[str, Any]): The input batch containing audio and transcript data.

    Returns:
        Dict[str, Any]: The batch with processed input values and modified labels.
    """
    audio = batch["audio"]
    batch["input_values"] = processor(audio["array"], sampling_rate=audio["sampling_rate"]).input_values[0]
    batch['length'] = len(batch["input_values"])
    with processor.as_target_processor():
        batch["labels"] = processor(batch[transcript]).input_ids
        # batch["labels"] = insert_blank_inbetween(batch["labels"], token_id=SPLITTING_DOUBLE_LETTER_TOKEN[1])
        token_id = processor.tokenizer.unk_token_id
        batch["labels"] = insert_blank_inbetween(batch["labels"], token_id=token_id)
    return batch
