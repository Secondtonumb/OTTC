# LibriSpeech SentencePiece BPE500

`bpe500/bpe.model` is the locally trained **BPE** tokenizer. It uses all
281,191 transcripts in the existing LibriSpeech 960h training manifest:
train-clean-100, train-clean-360 and train-other-500. No dev/test text is used.
It does not modify any existing model, data manifest or active experiment.

The total vocabulary has exactly 500 entries: 496 ordinary pieces and four
reserved IDs: blank/pad=0, unk=1, bos=2, eos=3. `tokens.json` is indexed by
native SentencePiece ID. `tokens.txt` contains the same IDs, not graph
disambiguation symbols. CTC must use blank=0; AED can use pad=0, bos=2 and
eos=3. Special IDs are not inserted by ordinary SentencePiece encoding.
This is a tokenizer asset, not a change to either trainer.

Use `prepare.normalize_text` before encoding: lowercase NFKC text, retaining
internal apostrophes and separating other punctuation. The retained train.txt
and metadata.json record training options, corpus/model/source hashes, and
full-training-corpus unknown-token and encode/decode roundtrip checks.

The separate `tokenizer/` directory retains an inspected public Icefall
`lang_bpe_500` asset, pinned to revision
ee9731ed5e6941ad58716704159d08cf46d34f35 of
https://huggingface.co/csukuangfj/icefall-asr-librispeech-pruned-transducer-stateless7-2022-11-11 .
Despite its directory name, its actual algorithm is **UNIGRAM**, not BPE.
It uses uppercase text and its own IDs; do not mix it with `bpe500/`.
Root metadata.json describes this public asset and the absence of an explicit
model-asset license in its publication metadata.

Run `prepare.py` with Python and SentencePiece to fetch, train and validate.
It refuses to overwrite an existing locally trained model. All checks and
artifacts are local-only; no acoustic training or W&B upload is performed.