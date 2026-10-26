"""Independent configurable native10ms OTTC recipe; upstream AG objective unchanged."""
from reproduction.timit_ssl_native10_20261019.common import *
from reproduction.timit_ssl_native10_20261019 import common as base

HERE = Path(__file__).resolve().parent

def source_hashes():
    paths = [HERE / n for n in ('common.py', 'train.py', 'profile_recipe.py', 'evaluate.py')]
    return dict(base.source_hashes(), **{str(p): sha(p) for p in paths})

def recipe(args):
    config = base.recipe(f'wav2vec2__timit__10ms__{args.head}__{args.cost}')
    config.update(epochs=args.epochs, global_batch=args.batch_size,
        learning_rate=args.learning_rate, warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay, seed=args.seed, dropout=args.dropout,
        max_grad_norm=args.max_grad_norm, precision=args.precision,
        requested_microbatch=args.microbatch, logging_steps=args.logging_steps,
        num_workers=args.num_workers, evaluation_uploads=False,
        experiment_family=HERE.name, source_sha256=source_hashes())
    config['name'] += f'__seed{args.seed}__interactive'
    return config

def build_model(config, vocab, steps_per_epoch):
    model = ScratchOTTC(wvOTTCConfig(wav2vec2_path=str(SSL), vocab_size=len(vocab),
        bins_metric=config['bins_metric'], forward_type='regular',
        wav2vec2_output_dropout=config['dropout']))
    model, topo = base.configure_model(model, config, vocab)
    assert all(p.dtype == torch.float32 for p in model.parameters())
    return model, topo

def atomic_weights(model, path, metadata):
    assert all(v.dtype == torch.float32 for v in model.state_dict().values() if v.is_floating_point())
    base.atomic_weights(model, path, metadata)
