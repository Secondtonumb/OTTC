# SPDX-FileCopyrightText: 2024 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Kaloga Yacouba yacouba.kaloga@idiap.ch
# SPDX-FileContributor: Shashi Kumar shashi.kumar@idiap.ch
#
# SPDX-License-Identifier: MIT

from datasets import DatasetDict
from transformers import Wav2Vec2Processor
from transformers import TrainingArguments, Trainer
import argparse
from ottc.config.path_config import DATASETS_ROOT_PATH, HUGGINGFACE_CACHE, LARGE_MODELS_PATH
from ottc.tools.data_collator.data_collator import DataCollatorCTCWithPadding
from ottc.models.sequence.wav2vec2_ctc import wvForCTC, wvCTCConfig
from ottc.models.sequence.wav2vec2_ottc import wvForOTTC, wvOTTCConfig
from ottc.utils.eval_metrics import build_compute_metrics, prepare_eval_logging_dir

from typing import Dict, Tuple, Any, Union, List
from dataclasses import dataclass, field
import os
from typing import Callable
from ottc.tools.dataset_loader.librispeech import load_librispeech
from ottc.tools.dataset_loader.ami import load_ami
from ottc.tools.dataset_loader.timit import load_timit
from ottc.tools.dataset_loader.librispeech import load_librispeech_multilingual
from functools import partial
from pathlib import Path

@dataclass
class DatasetConfig:
    name: str
    loader: Callable
    transcript_column: str
    kwargs: Dict[str, Any] = field(default_factory=dict)

DATASET_LOADERS: Dict[str, DatasetConfig] = {
    'librispeech_asr': DatasetConfig(
        name='librispeech_asr',
        loader=load_librispeech,
        transcript_column='text',

    ),
    'ami': DatasetConfig(
        name='ami',
        loader=load_ami,
        transcript_column='text'
    )} | {
    f'timit_{k}': DatasetConfig(
        name=f'timit_{k}',
        loader=partial(load_timit, **{"dataset_name": f"timit_{k}"}),
        transcript_column='text',
    ) for k in ['letter', 'phoneme', 'fused_phoneme']
    } | {
        f'multilingual_librispeech_{lang}': DatasetConfig(
            name='multilingual_librispeech',
            loader=partial(load_librispeech_multilingual, **{"language": lang}),
            transcript_column='transcript'
        ) for lang in ['en', 'fr', 'it', 'de', 'es']
    }
def load_config_from_args(args: argparse.Namespace) -> Dict[str, Any]:
    """Loads config from the parsed arguments and performs data validation."""
    
    if args.dataset not in DATASET_LOADERS:
            raise ValueError(f"Invalid dataset: {args.dataset}. Must be one of {list(DATASET_LOADERS.keys())}")

    config = {
        'suffix': args.suffix,
        'freeze_cnn': args.freeze_cnn,
        'continue_if_exist': args.continue_if_exist,
        'only_evaluate': args.only_evaluate,

        'dataset': args.dataset,
        'train_name': args.train_name,
        'test_name': args.test_name,
        'force_lowercase': args.force_lowercase,
        'prepare_dataset': args.prepare_dataset,

        'dataloader_num_workers': args.dataloader_num_workers,
        'group_by_length': args.group_by_length,
        'batch_size': args.batch_size,
        'num_train_epochs': args.num_train_epochs,
        'learning_rate': args.learning_rate,
        'weight_decay' : args.weight_decay,

        'encoder_name': args.encoder_name,
        'forward_type': args.forward_type, #type of forward
        'frozen_epochs_for_regfrozen': args.frozen_epochs_for_regfrozen, # nb of epochs to freeze the wav2vec2 part if forward_type is regfrozen
        'loss_type': args.loss_type, # ctc or ottc
        'oracle_weights_model': args.oracle_weights_model, #ctc pretrained model for ottc
        'bins_metric': args.bins_metric, #cost function for ottc
        'final_layers': args.final_layers,
        'dump_eval_artifacts': args.dump_eval_artifacts,

    }
    return config

def load_dataset_and_processor(config: Dict[str, Any], break_douuble_letter : bool) -> Tuple[DatasetDict, List[str], Wav2Vec2Processor]:
    """Loads the dataset based on config and returns it with the vocabulary and processor."""
    dataset_config = DATASET_LOADERS[config['dataset']]
    dataset, vocab, processor = dataset_config.loader(
        config['train_name'],
        config['test_name'],
        prepare_dataset=config['prepare_dataset'],
        force_lowercase=config['force_lowercase'],
        num_proc=config['dataloader_num_workers'],
    )
    return dataset, vocab, processor

def load_model(config: Dict[str, Any], vocab_size: int) -> Union[wvForCTC, wvForOTTC]:
    """Loads or creates a model based on the configuration."""
    wvModel = wvForOTTC if config['loss_type'] == 'ottc' else wvForCTC 
    wvConfig = wvOTTCConfig if config['loss_type'] == 'ottc' else wvCTCConfig
    model_output_dir = LARGE_MODELS_PATH / 'large_models_results' / config['encoder_name'] /  config['suffix']
    prefix = "checkpoint-"

    model_config_kwargs = {
        "wav2vec2_path": str(LARGE_MODELS_PATH / config['encoder_name']),
        "vocab_size": vocab_size,
        "wav2vec2_output_dropout": 0.1,
        "final_dropout": 0.1,
        "loss_type": config['loss_type'],
    }

    if config['loss_type'] == 'ottc':
        model_config_kwargs.update({
            "bins_metric": config['bins_metric'],
            "final_layers": config['final_layers'],
            "oracle_weights_model_name": config['oracle_weights_model'],
            "forward_type": config['forward_type'],
        })
        if 'frozen_steps_for_regfrozen' in config:
            model_config_kwargs["frozen_steps_for_regfrozen"] = config['frozen_steps_for_regfrozen']

    model_config = wvConfig(**model_config_kwargs)

    if config['continue_if_exist'] and os.path.exists(model_output_dir) and len(set(os.listdir(model_output_dir)) - {'transcripts'}) > 1:
            print(f"... existing model in {model_output_dir}")
            highest_folder = max((f for f in os.listdir(model_output_dir) if f.startswith(prefix)), key=lambda x: int(x[len(prefix):]), default=None)
            checkpoint_path = str(LARGE_MODELS_PATH / 'large_models_results' / config['encoder_name'] / config['suffix'] / highest_folder)
            print(f"Get last checkpoint in {checkpoint_path}")
            model = wvModel.from_pretrained(checkpoint_path)
            model.step = int(highest_folder[len(prefix):])
    else:
        print(f"... new model in {model_output_dir}")
        model = wvModel(model_config)
    return model


def create_trainer(
    model: Union[wvForCTC, wvForOTTC],
    dataset: DatasetDict,
    processor: Wav2Vec2Processor,
    config: Dict[str, Any],
    metrics_fn: Callable[[Any], Dict[str, float]],
    model_output_dir: Path,
) -> Trainer:
    """Creates the Trainer object."""
    data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)
    training_args = TrainingArguments(
        output_dir=model_output_dir,
        dataloader_num_workers = config['dataloader_num_workers'],
        group_by_length= config['group_by_length'],
        per_device_train_batch_size=config['batch_size'],
        num_train_epochs=config['num_train_epochs'],
        fp16=True,
        gradient_checkpointing=False,
        evaluation_strategy = "epoch",
        save_strategy = "epoch",
        logging_steps=200,
        learning_rate=config['learning_rate'],
        weight_decay=config['weight_decay'],
        warmup_steps=500,
        save_total_limit=5,
        report_to="tensorboard",
        push_to_hub=False,
        disable_tqdm=False,
        # per_device_eval_batch_size=config['batch_size'],
    )

    trainer = Trainer(
        model=model,
        data_collator=data_collator,
        args=training_args,
        compute_metrics=metrics_fn,
        train_dataset=dataset[config['train_name']],
        eval_dataset=dataset[config['test_name']],
        tokenizer=processor.feature_extractor,
    )

    return trainer

def train_model(trainer: Trainer, checkpoint_path: str = None) -> None:
    """Trains the model."""
    if checkpoint_path is not None:
        print("Continuing training from checkpoint.")
        trainer.train(checkpoint_path)
    else:
        print("Starting new training.")
        trainer.train()
    print("Training finished.")


def evaluate_model(trainer: Trainer, dataset: DatasetDict, config: Dict[str, Any]) -> None:
    """Evaluates the model."""
    print("Evaluating model.")
    trainer.evaluate(dataset[config['test_name']])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--suffix', type=str, default='temst', help='Suffix for model output directory.')
    parser.add_argument('--freeze_cnn', type=bool, default=True, help='Whether to freeze CNN layers.')
    parser.add_argument('--continue_if_exist', type=bool, default=True, help='Continue training from existing checkpoint if present.')
    parser.add_argument('--only_evaluate', type=bool, default=False, help='If only evaluation should happen.')

    parser.add_argument('--dataset', type=str, default='timit_letter', help='Name of the dataset to use (librispeech_asr, multilingual_librispeech, ami, timit).')
    parser.add_argument('--train_name', type=str, default='train', help='Name of the training split.')
    parser.add_argument('--test_name', type=str, default='kaldi_test', help='Name of the test split.')
    parser.add_argument('--force_lowercase', type=bool, default=False, help='Whether to force the dataset transcript to lowercase.')
    parser.add_argument('--prepare_dataset', type=str, default='prepare_dataset_breakdoubleletter', help='Dataset preparation method.')

    parser.add_argument('--dataloader_num_workers', type=int, default=15, help='Number of workers for dataloader.')
    parser.add_argument('--group_by_length', type=bool, default = True)
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training.')
    parser.add_argument('--num_train_epochs', type=int, default=40, help='Number of training epochs.')
    parser.add_argument('--learning_rate', type=float, default=2e-4, help='Learning rate for training.')
    parser.add_argument('--weight_decay', type=float, default = 0.005)

    parser.add_argument('--encoder_name', type=str, default='wav2vec2-large-lv60', help='Name of the encoder to use.')
    parser.add_argument('--forward_type', type=str, default ='regfrozen')
    parser.add_argument('--frozen_epochs_for_regfrozen', type=int, default = 30) # only for regfrozen forward type
    parser.add_argument('--loss_type', type=str, default='ottc', help='Type of loss to use (ctc).')
    parser.add_argument('--oracle_weights_model', type=str, default =None)
    parser.add_argument('--bins_metric', type=str, default = 'crossentropy')
    parser.add_argument('--final_layers', type=str, default = 'linear')
    parser.add_argument('--dump_eval_artifacts', action='store_true', help='Dump eval transcripts (and alignments for OTTC) after each evaluation epoch.')

    args = parser.parse_args()
    config = load_config_from_args(args)
    config['dump_eval_artifacts'] = True

    model_output_dir = LARGE_MODELS_PATH / 'large_models_results' / config['encoder_name'] /  config['suffix']

    logging_dir = prepare_eval_logging_dir(model_output_dir) if config['dump_eval_artifacts'] else None

    dataset, vocab, processor = load_dataset_and_processor(config, break_douuble_letter = (config['loss_type'] == 'ottc' ))

    if (
        config['loss_type'] == 'ottc'
        and config['forward_type'] == 'regfrozen'
        and config['frozen_epochs_for_regfrozen'] >= 0
    ):
        train_size = len(dataset[config['train_name']])
        steps_until_freeze = int(config['frozen_epochs_for_regfrozen'] * train_size / config['batch_size'])
        config['frozen_steps_for_regfrozen'] = steps_until_freeze

    model = load_model(config, len(vocab))

    if config['freeze_cnn']:
        model.freeze_feature_extractor()

    metrics_fn = build_compute_metrics(
        processor=processor,
        loss_type=config['loss_type'],
        logging_dir=logging_dir,
        capture_alignments=config['dump_eval_artifacts'] and config['loss_type'] == 'ottc',
    )

    trainer = create_trainer(
        model,
        dataset,
        processor,
        config,
        metrics_fn,
        model_output_dir,
    )

    prefix = "checkpoint-"

    if config['continue_if_exist'] and os.path.exists(model_output_dir) and len(set(os.listdir(model_output_dir)) - {'transcripts'}) > 1:
        highest_folder = max((f for f in os.listdir(model_output_dir) if f.startswith(prefix)), key=lambda x: int(x[len(prefix):]), default=None)
        checkpoint_path = str(LARGE_MODELS_PATH / 'large_models_results' / config['encoder_name'] /  config['suffix'] / highest_folder)
    else:
        checkpoint_path = None
    
    if not config['only_evaluate']:
        train_model(trainer, checkpoint_path)
    else:
        evaluate_model(trainer, dataset, config)


if __name__ == "__main__":
    main()
