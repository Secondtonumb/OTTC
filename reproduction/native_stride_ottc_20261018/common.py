"""Independent native-stride OTTC recipes; all losses reuse the upstream core."""
import ast
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT/'reproduction/vendor'), str(ROOT/'reproduction'), str(ROOT)]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name+'.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n'); tmp.replace(path)


def atomic_weights(model, path, metadata):
    import torch
    temporary = path.with_name(path.name+'.tmp')
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(model={k:v.detach().cpu() for k,v in model.state_dict().items()}, **metadata), temporary)
    temporary.replace(path)


def source_hashes():
    files = [HERE/name for name in ['common.py','train.py','evaluate.py','graphs.py','validate.py']] + [ROOT/'ottc/ops_loss/ot_loss.py',
        ROOT/'ottc/models/sequence/wav2vec2_ottc.py', ROOT/'reproduction/wavlm_timit_20261013/model.py',
        ROOT/'reproduction/zapformer_ifmdd_topology_ottc_20261013/vendor/utils/subphonetic_topology.py',
        ROOT/'reproduction/libri100_singlehead_20261014/train.py',ROOT/'reproduction/libri960_continue5_20261017/evaluate.py',
        ROOT/'reproduction/english_suite/mfa_protocol_comparison/scorer_source/boundary_rvalue.py']
    return {str(p):sha(p) for p in sorted(files)}


def recipe(name):
    backbone, corpus, frame, head, cost = name.split('__')
    assert backbone in ('wav2vec2', 'wavlm') and corpus in ('timit', 'buckeye', 'libri100')
    assert frame in ('10ms', '20ms') and head in ('mono', 's3t2ss')
    assert cost in ('ce', 'jsd_upstream', 'jsd_standard')
    transfer = backbone == 'wav2vec2'
    assert not transfer or (corpus == 'timit' and frame == '10ms')
    return dict(name=name, backbone=backbone, corpus=corpus, frame_ms=int(frame[:-2]), head=head, cost=cost,
        epochs=10 if transfer else 40, global_batch=64, learning_rate=2e-5 if transfer else 2e-4,
        warmup_steps=50 if transfer else 500, seed=42, weight_decay=.005,
        initialization='Libri960 upstream-JSD best-dev supervised checkpoint' if transfer else 'official microsoft/wavlm-large self-supervised checkpoint',
        upsample=False, native_conv_stride=[5,2,2,2,2,2,1 if frame == '10ms' else 2],
        bins_metric={'ce':'crossentropy','jsd_upstream':'jsd','jsd_standard':'jsd_standard'}[cost],
        objective='existing upstream OTTC core; explicitly selected CE/upstream-JSD/standard-JSD branch',
        topology='s3_t2_star_star' if head == 's3t2ss' else None,
        target_mass='state_uniform' if head == 's3t2ss' else 'phone_uniform',
        freeze_cnn=True, checkpoint_policy='atomic independent every-epoch weights; retain all epochs, best-dev and full latest Trainer state',
        source_sha256=source_hashes())


def topology(vocab):
    vendor = ROOT/'reproduction/zapformer_ifmdd_topology_ottc_20261013/vendor'
    sys.path.insert(0, str(vendor))
    from utils.subphonetic_topology import make_subphone_labels, token_fst_text, build_utterance_topology_arcs, decode_subphone_ids
    phones = sorted(p for p in vocab if p not in ('<pad>', '<unk>'))
    labels = make_subphone_labels(phones, 's3_t2_star_star', split_silence=False)
    index = {label:i for i,label in enumerate(labels)}
    mapping = {vocab[p]:[index[f'{p}_{s}'] for s in range(1 if p == 'sil' else 3)] for p in phones}
    return dict(phones=phones, labels=labels, mapping=mapping, token_fst_text=token_fst_text,
                build_arcs=build_utterance_topology_arcs, decode=decode_subphone_ids)


def load_data(config):
    import datasets
    from transformers import Wav2Vec2Processor
    if config['corpus'] == 'timit':
        base = ROOT/'reproduction/multihead_timit/data'
        vocab_path = ROOT/'reproduction/models/large_models_results/wav2vec2-large-lv60/runexp_ottc_fuse_phoneme/processor/vocab.json'
        vocab = json.loads(vocab_path.read_text()); processor = Wav2Vec2Processor.from_pretrained(vocab_path.parent)
        rows = {}; data = {}; hashes = {}
        for split, filename, expected in [('train','train-train',4620), ('dev','train-dev',400), ('test','test',192)]:
            path = base/(filename+'.jsonl'); original = [json.loads(l) for l in path.open()]
            assert len(original) == len({r['utt'] for r in original}) == expected
            rows[split] = [dict(utt=r['utt'], wav=r['wav'], duration=r['duration'],
                reference=r['references']['ifmdd_arpabet'], training_target_ids=r['targets']['ottc_arpabet']) for r in original]
            data[split] = datasets.Dataset.from_list([dict(wav=r['wav'], length=round(r['duration']*16000),
                labels=r['targets']['ottc_arpabet']) for r in original])
            hashes[split] = sha(path)
    else:
        base = ROOT/('reproduction/buckeye_singlehead_20261014/data' if config['corpus']=='buckeye' else 'reproduction/libri100_singlehead_20261014/data')
        vocab_path = base/'vocab.json'; vocab = json.loads(vocab_path.read_text())
        processor = Wav2Vec2Processor.from_pretrained(base/'processor')
        rows = {s:[json.loads(l) for l in (base/(s+'.jsonl')).open()] for s in ['train','dev','test']}
        data = {s:datasets.load_from_disk(base/('cache_'+s)) for s in ['train','dev']}
        hashes = {s:sha(base/(s+'.jsonl')) for s in rows}
    assert set(vocab.values()) == set(range(42)) and vocab['<pad>'] == 0 and vocab['<unk>'] == 41
    return data, rows, vocab, processor, dict(manifests_sha256=hashes, vocab_sha256=sha(vocab_path),
        counts={s:len(r) for s,r in rows.items()}, references='human PHN for TIMIT; original human partial19 Buckeye; existing automatic Libri boundaries, source model unverified')


class Collator:
    def __init__(self, processor, mapping=None):
        self.processor, self.mapping = processor, mapping

    def __call__(self, examples):
        import numpy as np
        import soundfile as sf
        import torch
        features = []; labels = []
        for row in examples:
            if 'input_values' in row:
                values = row['input_values']
            else:
                audio, sr = sf.read(row['wav'], dtype='float32'); assert sr == 16000 and audio.ndim == 1
                values = self.processor(audio, sampling_rate=sr).input_values[0]
            features.append(dict(input_values=values))
            target = row['labels']
            if self.mapping is not None:
                target = [s for token in target for s in self.mapping[int(token)]]
            labels.append(torch.tensor(target, dtype=torch.long))
        batch = self.processor.pad(features, return_tensors='pt', return_attention_mask=True)
        batch['labels'] = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)
        return batch


def build_model(config, vocab, steps_per_epoch):
    import torch
    import transformers
    transformers.utils.import_utils._peft_available = False
    transformers.utils.import_utils._bitsandbytes_available = False
    from ottc.models.sequence.wav2vec2_ottc import wvForOTTC
    transfer = config['backbone'] == 'wav2vec2'
    if transfer:
        parent = json.loads((ROOT/'reproduction/libri960_continue5_20261017/models/jsd_upstream/BEST_DEV.json').read_text())
        model, info = wvForOTTC.from_pretrained(parent['checkpoint'], output_loading_info=True)
        assert not any(info.get(k) for k in ['missing_keys','unexpected_keys','mismatched_keys','error_msgs'])
        config['parent_checkpoint'] = parent['checkpoint']
        config['parent_sha256'] = sha(Path(parent['checkpoint'])/'model.safetensors')
        config['step_offset'] = 17840+parent['stage_step']
    else:
        # Reuse the already validated official WavLM loader and upstream OTTC wrapper.
        module_dir = ROOT/'reproduction/wavlm_timit_20261013'
        sys.path.insert(0, str(module_dir))
        from model import WavLMForOTTC, WavLMOTTCConfig
        model = WavLMForOTTC(WavLMOTTCConfig(wav2vec2_path=str(ROOT/'reproduction/models/wavlm-large'),
            vocab_size=42, bins_metric=config['bins_metric'], forward_type='regfrozen',
            frozen_steps_for_regfrozen=30*steps_per_epoch))
        config['step_offset'] = 0
    model.config.bins_metric = config['bins_metric']
    model.jsd = {'ce':False, 'jsd_upstream':True, 'jsd_standard':'standard'}[config['cost']]
    model.euclidian = False
    topo = topology(vocab) if config['head'] == 's3t2ss' else None
    if topo:
        old = model.lm_head
        head = torch.nn.Linear(old.in_features, len(topo['labels']))
        with torch.no_grad():
            head.weight[0].copy_(old.weight[0]); head.bias[0].copy_(old.bias[0])
            for phone, states in topo['mapping'].items():
                for state in states:
                    head.weight[state].copy_(old.weight[phone])
                    head.bias[state].copy_(old.bias[phone]-math.log(len(states)))
        model.lm_head = head; model.config.vocab_size = len(topo['labels'])
        config['topology_labels'] = topo['labels']; config['phone_to_substates'] = {str(p):states for p,states in topo['mapping'].items()}
    for encoder in [model.wav2vec2, model.wav2vec2_freeze]:
        assert list(encoder.config.conv_stride) == [5,2,2,2,2,2,2]
        encoder.config.conv_stride = list(config['native_conv_stride'])
        encoder.feature_extractor.conv_layers[-1].conv.stride = (config['native_conv_stride'][-1],)
        assert math.prod(encoder.config.conv_stride) == config['frame_ms']*16
    model.wave2vec2_config = model.wav2vec2.config
    model.freeze_feature_encoder()
    model.wav2vec2_freeze.requires_grad_(False)
    if transfer:
        model.lm_weight.requires_grad_(False)
    model.config.native_frame_ms = config['frame_ms']; model.config.upsample = False
    model.config.native_conv_stride = config['native_conv_stride']; model.config.topology = config['topology']
    model.step = config['step_offset']-1
    return model, topo


def scorer():
    import numpy as np
    from rapidfuzz.distance import Levenshtein
    from segmentation_metrics import r_value_percent
    namespace = dict(np=np, Counter=Counter, Levenshtein=Levenshtein, r_value_percent=r_value_percent)
    for path, names in [(ROOT/'reproduction/libri100_singlehead_20261014/train.py', ['phone_per','boundary_matches']),
                        (ROOT/'reproduction/libri960_continue5_20261017/evaluate.py', ['edges','new_counts','score','summary'])]:
        nodes = [n for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef) and n.name in names]
        exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),namespace)
    return namespace


def phones_from_ids(ids, vocab, topo=None, collapse=True):
    import itertools
    ids = [int(i) for i in ids if i >= 0]
    if collapse:
        ids = [key for key,_ in itertools.groupby(ids)]
    if topo:
        return topo['decode'](ids, topo['labels'], mode='phone').split()
    inverse = {v:k for k,v in vocab.items()}
    return [inverse[i] for i in ids if i not in (0,41)]


def normalize(phones):
    return ' '.join('d' if p == 'dx' else p for p in phones if p not in ('sil','q'))
