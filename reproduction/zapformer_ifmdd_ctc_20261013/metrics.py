"""Reuse historical TIMIT scoring with the IF-MDD content CTC decoder."""
from pathlib import Path
from collections import Counter
import hashlib
import json
import sys
import torch.distributed as dist
import jiwer

HERE = Path(__file__).resolve().parent
SUITE = HERE.parent / "english_suite"
DATA = HERE.parent / "spottc_jsd_s3t2_framece_libri100_20261012/eval_data"
sys.path.insert(0, str(SUITE))
import mfa_metrics
sys.path.insert(0, str(HERE.parent / "ipa_arpabet_comparison_20261012"))
from rescore import mapped


def evaluate(net, split, rank, world, output, limit=0):
    output.mkdir(parents=True, exist_ok=True)
    old_data, old_greedy = mfa_metrics.DATA, mfa_metrics.greedy
    from speechbrain.decoders.ctc import filter_ctc_output
    mfa_metrics.DATA = DATA
    mfa_metrics.greedy = lambda path: filter_ctc_output(path, blank_id=0)
    try:
        result = mfa_metrics.evaluate(net, "ctc", split=split, rank=rank,
                                      world=world, forced=True, limit=limit, output=output)
    finally:
        mfa_metrics.DATA, mfa_metrics.greedy = old_data, old_greedy
    references = {row["id"]: row for row in
                  (json.loads(line) for line in (DATA / f"{split}.jsonl").open())}
    count = Counter()
    for row in (json.loads(line) for line in (output / f"{split}_predictions.jsonl").open()):
        ref = mapped(references[row["id"]]["tokens"], "english_collapsed")
        hyp = mapped(row["predicted_ids"], "english_collapsed")
        edits = jiwer.process_words(" ".join(ref), " ".join(hyp))
        count.update(reference_phones=len(ref), substitutions=edits.substitutions,
                     deletions=edits.deletions, insertions=edits.insertions, utterances=1)
    if world > 1:
        gathered = [None] * world
        dist.all_gather_object(gathered, dict(count))
        count = Counter()
        for local in gathered:
            count.update(local)
    n = count["reference_phones"]
    result["mapped_arpabet"] = dict(count)
    result["mapped_arpabet"]["PER_percent"] = 100 * sum(
        count[k] for k in ("substitutions", "deletions", "insertions")
    ) / n
    result["protocol"].update(
        subset="official dev400" if split == "development" else "official core-test192",
        recognition="Audio only; SpeechBrain CTC merge repeats before removing blank; speech IDs 3..102",
        training_topology="standard single-token CTC; original MFA inventory and targets",
        decoder="IF-MDD content CTC head; no substate expansion/completion",
        fa="Supplied phones; historical IF-MDD native CTC FSA; previous blank; duration/frames",
        mapped_recognition="Frozen IPA to English-collapsed ARPAbet/TIMIT39 mapping and jiwer",
        test_checkpoint_selection=False,
        adapter_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        source_sha256=hashlib.sha256((SUITE / "mfa_metrics.py").read_bytes()).hexdigest(),
    )
    (output / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    return result
