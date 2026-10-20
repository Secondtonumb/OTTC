"""Cached 10ms inputs and resumable plans produced by native Lhotse sampling."""
from pathlib import Path
import hashlib
import gzip
import json
import numpy as np
import torch
from lhotse import CutSet, MonoCut
from lhotse.dataset import DynamicBucketingSampler

CE_ROOT = Path('/work/gm64/m64000/OTTC/reproduction/spottc_jsd_s3t2_framece_libri100_20261012')
JSD_MANIFEST = Path('/work/gm64/m64000/OTTC/reproduction/zapformer_libri100_data_20261012/training.json')


def read_rows(path):
    return [json.loads(line) for line in Path(path).open()]


def jsd_rows(limit=0, manifest=JSD_MANIFEST):
    manifest=Path(manifest)
    meta=json.loads(manifest.read_text())
    entry=meta['split_manifests'][meta['train_split']]
    source=manifest.parent/entry['path']
    assert hashlib.sha256(source.read_bytes()).hexdigest()==entry['sha256']
    rows=[]
    with gzip.open(source,'rt') as stream:
        for line in stream:
            rows.append(json.loads(line))
            if limit and len(rows)>=limit:break
    assert len(rows)==(min(limit,entry['count']) if limit else entry['count'])
    for row in rows:
        path=Path(row.get('feature_file') or meta['feature_file'])
        if not path.is_absolute():path=manifest.parent/path
        row['feature_file']=str(path)
    return rows


def plan(rows, duration, rank, world, epoch, seed=42):
    cuts = CutSet.from_cuts(MonoCut(id=r['id'], start=0, duration=r['frames']/100., channel=0) for r in rows)
    sampler = DynamicBucketingSampler(cuts, max_duration=duration, quadratic_duration=30.,
        num_buckets=30, buffer_size=20000, shuffle=True, drop_last=True,
        world_size=world, rank=rank, seed=seed+500*(epoch+1), sync_buckets=True)
    # Native recipe recreates the sampler with an epoch-specific seed and
    # explicitly does not call set_epoch(epoch) a second time.
    return [[cut.id for cut in batch] for batch in sampler]


class Inputs:
    def __init__(self, rows):
        self.rows = {r['id']: r for r in rows}
        self.caches = {}

    def batch(self, ids):
        rows = [self.rows[key] for key in ids]
        features = []
        for row in rows:
            path = row['feature_file']
            if path not in self.caches:
                self.caches[path] = np.memmap(path, mode='r', dtype=np.float32)
            offset = row['feature_offset']
            features.append(torch.from_numpy(np.array(self.caches[path][offset:offset+row['frames']*80]).reshape(-1,80)))
        lengths = torch.tensor([len(x) for x in features])
        # Reuse the previously verified historical timeline -> frame target function.
        from historical_targets import targets
        gold = targets(rows, lengths, torch.device('cpu')) if 'intervals' in rows[0] else None
        tokens = [r['tokens'] if 'tokens' in r else [label for label, _, _ in r['intervals']] for r in rows]
        return dict(inputs=torch.nn.utils.rnn.pad_sequence(features, batch_first=True), rows=rows,
                    tokens=torch.tensor([p for seq in tokens for p in seq]), token_lengths=torch.tensor([len(t) for t in tokens]),
                    frame_targets=gold, supervisions=dict(num_frames=lengths, text=[' '.join(map(str,t)) for t in tokens]))


class Loader:
    def __init__(self, inputs, batches, epoch, cursor=0, before_next=None, stop=None):
        self.inputs, self.batches, self.epoch = inputs, batches, epoch
        self.cursor, self.batch_offset = cursor, cursor
        self.before_next, self.stop = before_next, stop
        self.sampler = self

    def __iter__(self):
        return self

    def __next__(self):
        if self.before_next:
            self.before_next()
        if self.cursor >= len(self.batches) or self.stop and self.stop():
            raise StopIteration
        batch = self.inputs.batch(self.batches[self.cursor])
        self.cursor += 1
        return batch

    def state_dict(self):
        serialized = json.dumps(self.batches, separators=(',', ':')).encode()
        return dict(epoch=self.epoch, cursor=self.cursor, plan_sha256=hashlib.sha256(serialized).hexdigest())

    def load_state_dict(self, state):
        assert self.epoch == state['epoch'] and self.state_dict()['plan_sha256'] == state['plan_sha256']
        self.cursor = self.batch_offset = state['cursor']
