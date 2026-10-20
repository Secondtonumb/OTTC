# Libri100 input candidate for native Zapformer SP-OTTC JSD

This preparation is independent of the pending decision about deterministic k2
accumulation. It submits no GPU training and does not unlock migration gates.

`training.json` and `train.jsonl.gz` contain the exact historical accepted
Libri100 utterance subset: **28,538 utterances, 100.586496 hours, 251 speakers**.
Every utterance has one matching native MFA transcript row; there are no missing
or duplicated matches. Original internal split labels are all `train`.

The acoustic inputs reuse **11 existing immutable Fbank shards**. Every shard's
complete SHA256 and byte length was verified against the native source manifest;
frame-offset bounds were also checked. No acoustic features were copied,
recomputed or downloaded. Rows use absolute shard paths so they can be consumed
from a different experiment directory without changing the cache.

Targets are the existing **native MFA word-transcript IPA phone sequences**,
with the current 105-ID inventory. Control IDs 0..2 do not occur in transcripts.
The historical Libri100 framewise-CE manifest is used only to select utterance
IDs: its ARPAbet timeline labels mapped to IPA representatives are a different
target definition and are not substituted here. This distinction must remain
in experiment configuration and in any CE comparison.

Preparation records source manifest and compressed-row hashes, the historical
selection-manifest hash, the token-inventory hash, preparation-source hash,
feature configuration, phone usage and every referenced feature shard hash.
Compressed output uses a fixed gzip timestamp for reproducible bytes.

Rebuild from existing data:

```bash
cd /work/gm64/m64000/OTTC
python reproduction/zapformer_libri100_data_20261012/prepare.py
```

The verified compressed-row SHA256 on 2026-10-12 is
`fbe509829659633312c43c790a335619537a044b28a7d989af313e028b6e7b20`.
The native training adapter still has the old MFA mixed-set input hardcoded.
Before Libri100 formal training, it needs explicit dataset selection plus
successful JSD/CE integration and DDP validation, followed by throughput tests
on the actual Libri100 batch plans. Dataset preparation alone is not evidence
of a completed migration or a trained Libri100 model.
