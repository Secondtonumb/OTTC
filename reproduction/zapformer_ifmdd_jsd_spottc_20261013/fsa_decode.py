"""Exact topology graphs from frozen IF-MDD arcs; recognition uses no transcript."""
import json
from pathlib import Path
import torch
import k2
from task_ottc import criterion, HERE
from utils.subphonetic_topology import token_fst_text, build_utterance_topology_arcs

INVENTORY = {label: int(idx) for label, idx in
             (line.rsplit(' ', 1) for line in Path(json.loads((HERE/'reference.json').read_text())['inventory']['path']).read_text().splitlines())}


def graph(loss_module=None, supplied=None, device='cpu'):
    module = criterion() if loss_module is None else loss_module
    lines = []
    if supplied is None:
        symbolic = token_fst_text(module.phones, module.topology, split_silence=False, blank_label='<blank>')
        labels = {label: i for i, label in enumerate(module.labels)}
        maximum = 0
        for line in symbolic.splitlines():
            parts = line.split()
            if len(parts) == 1:
                continue
            src, dst = map(int, parts[:2]); maximum = max(maximum, src, dst)
            ilabel = labels[parts[2]]
            olabel = 0 if parts[3] == '<eps>' else INVENTORY[parts[3]]
            lines.append((src, dst, ilabel, olabel))
        final = 0; super_final = maximum + 1
    else:
        phone_ids = [int(module.content_to_phone[p]) for p in supplied]
        if not phone_ids or any(p <= 0 for p in phone_ids):
            raise ValueError('Unknown supplied MFA phone')
        n, arcs, _, final = build_utterance_topology_arcs(
            phone_ids, module.phone_to_substates, module.topology, blank_index=0,
            repeat_separator='optional_blank', silence_self_loops='none')
        boundary = 0; next_free = 1; entries = {}
        for occurrence, phone_id in enumerate(phone_ids, 1):
            entries[(boundary, module.phone_to_substates[phone_id][0])] = occurrence
            boundary = next_free + len(module.phone_to_substates[phone_id]); next_free = boundary + 1
        assert boundary == final
        for arc in arcs:
            lines.append((arc.src, arc.dst, arc.label, entries.get((arc.src, arc.label), 0)))
        super_final = n
    lines.append((final, super_final, -1, -1))
    text = '\n'.join(f'{src} {dst} {label} {output} 0.0' for src,dst,label,output in sorted(lines)) + f'\n{super_final}\n'
    fsa = k2.arc_sort(k2.Fsa.from_str(text, acceptor=False).to(device))
    values = fsa.arcs.values()
    fsa.original_src = values[:, 0].clone(); fsa.original_dst = values[:, 1].clone()
    return k2.Fsa.from_fsas([fsa]), text


def decode_fsa(scores, vector):
    dense = k2.DenseFsaVec(scores.log_softmax(-1).unsqueeze(0),
                           torch.tensor([[0, 0, len(scores)]], dtype=torch.int32))
    lattice = k2.intersect_dense(vector, dense, output_beam=8., frame_idx_name='frame_idx')
    best = k2.shortest_path(lattice, use_double_scores=True)
    path=[]; outputs=[]; starts=[]; node=0
    for label, output, frame, src, dst in zip(best.labels.tolist(), best.aux_labels.tolist(),
                                             best.frame_idx.tolist(), best.original_src.tolist(), best.original_dst.tolist()):
        assert src == node; node = dst
        if label == -1:
            continue
        assert frame == len(path)
        path.append(label)
        if output > 0:
            outputs.append(output); starts.append(frame)
    if len(path) != len(scores):
        raise ValueError('No complete topology path')
    return outputs, path, starts


def forced_align(scores, supplied, duration, module):
    vector, _ = graph(module, supplied=supplied, device=scores.device)
    occurrences, path, starts = decode_fsa(scores, vector)
    if occurrences != list(range(1, len(supplied)+1)):
        raise ValueError('Topology FSA did not cover all supplied phone occurrences')
    # Historical IF-MDD previous-blank policy: leading blank goes to first phone,
    # interior blank to previous phone, final blank to last phone.
    segments=[]
    for i, token in enumerate(supplied):
        start = 0 if i == 0 else starts[i]
        end = starts[i+1] if i+1 < len(starts) else len(scores)
        segments.append(dict(token_id=token, start=start*duration/len(scores), end=end*duration/len(scores)))
    return segments, path
