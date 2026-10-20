# TIMIT40 CIF: segmentation with CTC / upstream-JSD OTTC auxiliary supervision

Three independent configurations with the same CIF main task:

| Experiment | Training objective |
|---|---|
| `no_ctc` | `(sum token CE + sum L1 quantity) / target phones` |
| `ctc_0p5` | `(sum token CE + sum L1 quantity + 0.5 sum CTC) / target phones` |
| `ottc_jsd_0p5` | `(sum token CE + sum L1 quantity) / target phones + 0.5 native OTTC sum / utterances` |

The CE uses uniform label smoothing 0.1. Quantity uses the **unscaled** sigmoid
weight sum. The optional CTC classifier consumes encoder frames, before CIF.
The no-CTC variant does not construct a CTC classifier or call CTC loss.
The third variant replaces auxiliary CTC with the previously reproduced original
upstream OTTC `jsd=True`, imported directly from
`reproduction/zapformer_s3t2_native_20261012/archived_ot_loss.py` (SHA256
`165f62cbc096a7f5d9b218763fe8848d21bb69a4a17379ef077625ea3992e80f`).
It uses independent frame-phone and softmax-mass heads on the shared encoder,
uniform target mass per phone occurrence, and the same42-ID inventory. This
adapts only the inventory/head wiring; no divergence formula is rewritten.
Upstream `jsd` is not standard JSD. CIF mass is not reused as OTTC mass.
CTC and OTTC auxiliary branches are mutually exclusive. Raw OTTC already has
unit transport mass per utterance, so its loss is averaged over utterances;
CTC is normalized by phones alongside CE. Equal0.5 coefficients do not imply
matched gradient strengths. This initial coefficient is unretuned, and raw
CTC/OTTC losses must not be compared directly. No variant uses BERT distillation.
This adds a CIF baseline to the OTTC repository; it does not change its OT loss.

References: [CIF paper](https://arxiv.org/abs/1905.11235),
[CIF-PyTorch](https://github.com/MingLunHan/CIF-PyTorch), and
[CIF-HieraDist criterion](https://github.com/MingLunHan/CIF-HieraDist/blob/main/fairseq/criterions/cif.py).
The new code is an independent implementation, not a vendored copy of Fairseq.

当前训练目标（用户确认）：**三组各40epochs**。已经完成的5轮是中间进度，
从现有checkpoint和同一W&B ID继续；不是将总训练缩短为5轮。
交互队列每用户只能同时运行一个作业，且已分配节点不能延长walltime。
CTC在interactive `3566826.opbs`继续，节点结束后由依赖作业 `3566985.opbs`
接力；无辅助损失为 `3566983.opbs`，upstream-JSD OTTC为 `3566984.opbs`。
三组均移除365步的临时上限，保持40轮/2920步的LR计划。
`active_runs.json`仍指向原`pilots/3566826.opbs/`路径，以保留所有已完成进度。

验证状态（2026-10-16）：11项算法/梯度/损失/解码/保存加载测试已在两个环境通过。
真实 TIMIT 音频的 tiny smoke 已跑通无CTC与CTC0.5两组；无CTC的2步后续训到4步，
与连续4步的参数、优化器、LR调度、RNG和dev/test结果逐项一致。
证据见 [CPU_VALIDATION.json](validation/CPU_VALIDATION.json)。这些是实现验证，
不是正式分段结果。原排队探测 `3566634.opbs` 已撤销，改用用户指定的interactive
节点 `3566826.opbs`（mg0032 / GH200）。新增OTTC的原生值/梯度与loss归一化测试
已在GPU通过：当前共13项测试。提交与进展见 [SUBMITTED.json](SUBMITTED.json)。

## Matched data and model

- Reuse `reproduction/multihead_timit/data/{train-train,train-dev,test}.jsonl`:
  train4620 / dev400 / core-test192, including the existing training SA utterances.
- Only `targets.ottc_arpabet` and `references.ottc_arpabet` are selected. This is a
  **single-head TIMIT40** model, using the frozen 42-ID vocabulary from the
  completed original OTTC CE processor. Other inventories are not combined.
- Waveforms, utterance normalization, source hashes and IDs are checked/read;
  manifests, completed CE/JSD models, and historical source snapshots are untouched.
- Wav2Vec2-large-LV60 starts from the same local official pretrained encoder as
  original single-head OTTC; frozen CNN, trainable transformer, gradient checkpointing.
- CIF predictor: conv(width3, channels256), LayerNorm/ReLU/dropout, sigmoid.
  Threshold is fixed at the paper's recommended 1.0. Training rescales mass to
  target phone count; inference uses predicted mass, with tail firing at >0.5.
- AR decoder: previous phone embedding + previous CIF feature -> input projection
  -> 2 causal self-attention layers (dim256, 4heads, FFN2048) -> concatenate
  current CIF feature -> phone logits. No cross-attention or external LM.
- Explicit adaptation for phone segmentation: **no EOS target**, to preserve the
  existing phone occurrence sequence and quantity count; private input-only BOS
  embedding. The output vocabulary stays exactly 42 IDs. CTC blank is ID0;
  blank is suppressed during CIF phone decoding. Unknown predictions count as errors.
- Defaults: seed42, batch64, AdamW LR2e-4, weight decay.005, warmup500,
  linear decay over40 epochs, BF16 GPU autocast with FP32 CIF/loss.
  This CIF recipe has no OTTC `regfrozen` alignment snapshot. These architecture,
  precision and loss differences must be disclosed when comparing to OTTC results.

## Segmentation is the primary evaluation

Each epoch evaluates **dev400**; select maximum dev text-free R-value@20ms.
At completion, evaluate the selected checkpoint on dev400 and core-test192.
No test checkpoint selection. Full-set boundary results are independent of PER.

Recognition and segmentation receive **audio only**: no reference phones, phone
counts, timestamps or length scaling. Supervised validation CE is a separate
forward with reference lengths and must not be confused with recognition.

Every CIF firing closes a segment. Internal events are all firings except the
last closing event, regardless of phone identity. Consecutive repeated phones
retain distinct events; there is no CTC repeat collapse. No extra end boundary
is fabricated when CIF emits zero/one segment.

Boundary reports include Precision/Recall/F1/**R-value** at10/20/30ms, raw TIMIT40
PER and the existing no-sil normalization (remove sil/q, dx->d). Match frozen
IF-MDD directional greedy code via `reproduction/segmentation_metrics.py`, recording
the frozen scorer SHA256. References are all internal phone occurrence starts,
including silence; utterance endpoints excluded. No supplied-phone FA is reported.

Two timestamp variants share the same duration/T encoder clock:

- `text_free`: fractional cumulative mass crossing, assuming uniform mass inside
  each encoder cell; this is an interpolation assumption, not subframe observations.
- `text_free_frame`: right endpoint of the firing cell, with no interpolation.

These match the native TIMIT scorer's clock/matching conventions. The older
Wav2Vec2 single-head evaluator uses a fixed20ms clock,20ms edge exclusion and
one-to-one matching, so its historical numbers are **not** directly identical
protocol controls. Rescore existing CE/JSD predictions/checkpoints under a common
protocol before making improvement claims. No MFA topline claim is made here.

Outputs contain per-utterance hypothesis/reference tokens, all predicted
boundaries, reference boundaries, predicted quantity, duration and frame counts.
Metrics are saved locally and logged to independent W&B runs, using
`reproduction/wandb_config.json` and explicit environment overrides. Offline
logging falls back when networking fails. Final benchmark summaries are uploaded.

## Correctness and bounded probes

```bash
python -m unittest reproduction.cif_timit_20261016.test_cif -v
qsub reproduction/cif_timit_20261016/profile.pbs
```

The probe runs all three **full pretrained encoder** variants at batch64 for4 updates on
the64 longest training utterances, then scores16dev/16test diagnostic utterances.
It records loading/transfer, forward/backward, optimizer times, memory, gradient
norm, actual loss components and continuous GPU telemetry. Each probe has its own
W&B ID. Probe weights are discarded; they are not production initialization.
`PROBE_COMPLETE.json` requires all requested finite updates. The resulting
segmentation metrics are **partial smoke checks**, not trained benchmark results.
The probe is a longest-batch safety/timing check, not a full-epoch throughput study.

Historical diagnostics used `pilot_threeway.sh` for five epochs (365updates),
preserving the40-epoch LR schedule and resumable state. Those five epochs are
intermediate checkpoints, not final experiments. **The current requested runs
must use the40-epoch continuation commands below or `run40.pbs`, without
`--max-steps 365`.** All4620training utterances are visited each epoch; full
dev400 / test192 scoring is retained.

CPU checks with real TIMIT waveforms can use a tiny random model:

```bash
python -m reproduction.cif_timit_20261016.train \
  --output reproduction/cif_timit_20261016/validation/tiny_no_ctc \
  --tiny --ctc-weight 0 --epochs 2 --batch-size 4 --train-limit 16 --eval-limit 4 \
  --max-steps 2 --warmup-steps 0 --evaluate-on-stop --wandb-mode offline
```

Repeat exactly that command with `--max-steps 4` to test continuation from
`resume.pt`; use a separate output and `--ctc-weight 0.5` for the CTC variant.
Tiny results only establish implementation correctness, never model performance.

## Training inside an allocated GPU shell

After reviewing the full-model probe, the separate resumable commands are:

```bash
CIF_CTC_WEIGHT=0 bash reproduction/cif_timit_20261016/interactive.sh
CIF_CTC_WEIGHT=0.5 bash reproduction/cif_timit_20261016/interactive.sh
CIF_OTTC_WEIGHT=0.5 bash reproduction/cif_timit_20261016/interactive.sh
```

Each invocation saves/stops after3000seconds by default; choose a budget below
the allocation limit via `CIF_MAX_WALL_SECONDS`. Rerunning continues the same
configuration, optimizer, full40-epoch scheduler, RNG, batch cursor and W&B ID.
After the bounded pilots finish, `active_runs.json` points these default commands
to their existing matching checkpoints, so continuation does not restart the
five completed epochs. Explicit `CIF_OUTPUT` always overrides that pointer.
It does not start the second command until the first invocation returns. Changed
data/model/loss/source contracts require a separate output via `CIF_OUTPUT`.
INT/TERM request saving after an optimizer update; SIGKILL cannot save.

`resume.pt` is the latest full training state; `best_dev_r.pt` is selected by
development segmentation. `PAUSED.json` is a bounded unfinished run;
`TRAIN_COMPLETE.json` is written only after every planned epoch and final dev/test
evaluation. Tiny/limited runs remain explicitly identified in runtime/metrics.
