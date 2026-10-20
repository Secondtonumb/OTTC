# Evaluation uploads (user update, 2026-10-19)

Tests, smoke/probe runs, checkpoint comparisons, forced alignment, recognition,
segmentation evaluation, and their reports are local-only by default. Do not
upload their artifacts, predictions, audio, tables, code snapshots, or metrics
to W&B/CoreWeave, and do not sync old offline evaluation runs unless the user
explicitly requests it. This supersedes older evaluation-upload requirements
below. Preserve complete local metrics, predictions, configs, and provenance.
Training scalar curves remain enabled as before; do not upload large evaluation
artifacts from a training run. Use reproduction/wandb_policy.py for evaluation
initialization and artifact logging. The default evaluation_uploads setting is
false; only an explicit user request may enable OTTC_EVAL_UPLOAD=1.
Keep completed source snapshots and active training/resume contracts unchanged.

# Experiment tracking

The user requested W&B curves for existing experiments and W&B logging for
all future experiments in this repository. Enable W&B by default for new
training runs, alongside local logs. Use `reproduction/wandb_config.json`
for project/entity defaults and honor explicit user overrides.

Record raw training/evaluation loss, optimizer steps, epochs, learning rate,
recognition metrics and experiment configuration. Keep each independent
experiment in its own run and reuse its saved run ID for checkpoint resumes.
Upload final recognition/segmentation benchmark summaries. If compute-node
networking is unavailable, retain offline W&B logs and sync when possible.

Historical imports must be marked as backfills and retain original recording
intervals. Do not interpolate missing loss points or present import timestamps
as training timestamps. Raw losses with different definitions/reductions are
not directly comparable. Existing trained checkpoints and source snapshots
must remain unchanged when adding tracking.

Reuse completed CE models for the same dataset and model/phoneset/training
configuration. Do not retrain them for evaluation or checkpoint averaging.
Resume unfinished matching runs from their checkpoints. Check the actual
data manifest and model configuration when matching earlier experiments;
dataset name alone does not make single-head, four-head, WavLM and Zapformer
weights interchangeable.

The user's main experimental focus is OTTC with the original upstream jsd
objective. Prioritize its recognition, segmentation and checkpoint-average
results in reporting; reuse matching CE experiments as the comparison.
Do not interpret reporting priority as cancelling other authorized jobs.

For Zapformer JSD, directly reuse the previously reproduced original OTTC
implementation. Do not write a replacement divergence formula or silently
substitute IF-MDD's standard JSD. Record any inventory adaptation explicitly.

When comparing alignment with the existing native MFA topline, reuse the
historical IF-MDD raw-PHN scorer and record its source hash. Keep full-set
boundary results separate from equal-phone-count TSE subsets and report a
common subset for direct TSE comparisons. Preserve separate text-free metrics
and state word-transcript versus supplied-phone input differences.

The archived IF-MDD/TIMIT_MFA CTMs are not verified as the MFA 3.0 baseline
in arXiv:2609.13694v1 Table II. Their closure-merge scores match that paper's
Kaldi TDNN-F row. Treat their model identity as unverified; numeric replay or
a directory name does not establish identity. Cite the paper MFA values
(dev TSE 30.73 ms / R-value 87.83%; test 31.95 ms / 87.44%) separately from
local reproduction and do not claim OTTC exceeds that topline from these CTMs.

Record segmentation R-value alongside precision/recall/F1 in development and
test, keeping text-free and forced alignment separate. Reuse the frozen
IF-MDD R-value formula; report the boundary protocol explicitly. Historical
R-values may be derived from saved P/R without retraining or inference, but
must be marked as backfills with original epochs/steps and no interpolation.

# Checkpoint averaging for future training

The user requested this on 2026-10-17 after the CIF TIMIT runs retained only
best/latest weights. Before starting any new training recipe, ensure its
trainer atomically saves an independently named model-weight checkpoint at
every completed epoch, including epoch, optimizer step and model/config
identity. Keep the full resumable latest state and best-dev checkpoint as
well. Per-epoch averaging checkpoints need not duplicate optimizer state.
Retain at least the final ten epoch checkpoints; do not overwrite each epoch
with a single resume.pt and treat that as support for checkpoint averaging.
Do not delete existing artifacts as part of implementing this policy.

For completed future runs with sufficient checkpoints, evaluate averages of
the final five and final ten epochs alongside the single best-dev model.
Choose any reported average configuration using development data only and
report test results separately. Record member paths, epochs, steps and hashes,
and upload recognition/segmentation summaries to W&B with local results.
An average of best plus final is a separate two-point experiment, not a
last-five/last-ten average. Preserve existing completed source snapshots and
active resume/source-hash contracts; apply trainer changes in a new recipe
when necessary. This request does not authorize restarting completed runs.
