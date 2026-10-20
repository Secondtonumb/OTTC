# TIMIT BPE500: matched 50-epoch experiments

This recipe implements the user's 2026-10-20 instruction to redo the short
experiments for 50 epochs, with recognition and segmentation as the outcomes.
The old 10/40-epoch experiments, source snapshots and checkpoints are unchanged.
Their zero-LR or different-schedule checkpoints are not imported into this
matched fresh-initialization comparison.

## Training and submission

For an already allocated PBS GPU interactive terminal:

```bash
bash reproduction/timit_bpe500_50ep_20261020/run_jsd_interactive.sh --replace-queued
```

This defaults to `ottc_jsd`; add `--recipe spottc_jsd` for SPOTTC-JSD.
`--replace-queued` explicitly cancels only matching pending batch copies after
successful preflight. Running jobs and jobs with dependent successors are never
cancelled. Without this flag the launcher refuses duplicate queued work.
The default budget is 6000 seconds, capped by the current allocation's remaining
walltime minus twenty minutes. The same command resumes the existing matched
checkpoint toward 50 epochs. Logs and a launcher source snapshot stay in jobs/.
`--self-test` checks launcher safety without a GPU or queue modifications.

```bash
source reproduction/libri960_continue5_20261017/environment.sh
"$LIBRI_PYTHON" reproduction/timit_bpe500_50ep_20261020/train.py --check
"$LIBRI_PYTHON" reproduction/timit_bpe500_50ep_20261020/train.py --submit-all
```

Order: CTC, OTTC-CE, original OTTC-JSD, SPOTTC-CE, original SPOTTC-JSD.
PBS `afterok` dependencies run one job at a time. This is scheduling order,
not weight transfer or a recognition-quality gate. Repeated submission reuses
active jobs and completed matching runs. `--submit --recipe NAME` submits an
individual recipe; `--dry-run` previews commands. A failed predecessor blocks
later jobs. No old job is cancelled and no old artifacts are removed.

All five use the same 4620/400/192 TIMIT manifests, BPE500 tokenizer, official
Wav2Vec2-large-LV60 SSL initialization, seed 42, physical batch64 (tail12),
one forward/backward, no gradient accumulation, learning rate 2e-4, weight decay
0.005, 500 warmup steps and exactly 50 epochs/3650 optimizer steps. There is no
microbatch splitting or automatic OOM fallback. OTTC retains the original
regfrozen switch after epoch30. SPOTTC uses existing S3T2 state-uniform targets
and 1498 outputs, including states for unused special IDs; no loss is replaced.

Each job requests one GPU for two hours; training has a 6000-second shared
budget with twenty minutes reserved for saving/evaluation. A budget pause
returns 75 with complete resume state, never marks fewer than 50 epochs as
complete, and does not advance dependent jobs. Resume the same recipe if needed.

## Recognition and segmentation

Every epoch records full-dev WER/CER and text-free **word** boundary P/R/F1 and
R-value at 10/20/30ms. BPE pieces are merged into words and compared to original
TIMIT `.wrd` timestamps, with full dev400/test192 coverage. Source annotation
hashes and complete references are retained. Original overlapping intervals,
zero-duration words and transcript discrepancies are preserved and counted;
no utterances are silently removed.

The clock is native 20ms frame index. Token spans extend from onset to the next
token onset, assigning blanks to the previous token, with the last token ending
at audio duration. SPOTTC uses the existing state-reset span decoder. Unique
word starts/ends within 20ms of utterance edges are excluded. The matcher and
R-value are the frozen IF-MDD directional-greedy implementation, with its source
hash recorded. No lexical match is required for boundary scoring.

These are audio-only greedy word-segmentation scores, **not phoneme segmentation
or forced alignment**. Reference text/timing is never given to the decoder.
They must not be compared directly to phone-boundary or MFA supplied-text scores.

After 50 epochs, best-dev, last5 and last10 are evaluated. All dev evaluations
finish before test inference. Recognition selects minimum dev WER; segmentation
separately selects maximum dev R-value at 20ms. Both selections and every
candidate's dev/test metrics are saved. Each candidate uses its own joint/frozen
phase when evaluated. Different objectives' raw loss scales are not comparable.

## Retention and tracking

Every completed epoch atomically saves named weights and model/source identity.
The existing retention callback keeps the dev-best3, last10 and detected
evaluation dependencies, preserving metadata and deletion provenance. Best-dev
weights and the latest two complete optimizer/scheduler/RNG states are retained;
incomplete checkpoint directories are preserved but excluded from resume.
Model hashes are verified on resume. Source/data changes reject active contracts.

Training scalar logging retains the normal W&B defaults and a separate stable
run ID per recipe. Evaluation artifacts, word segments, reports, references,
configs and predictions remain local with `OTTC_EVAL_UPLOAD=0`.