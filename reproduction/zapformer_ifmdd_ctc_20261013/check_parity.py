"""Behavioral parity: loss/gradient, padding, repeats, and original graph DP."""
from pathlib import Path
import hashlib
import importlib.util
import json
import sys
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from task_ctc import ctc_objective, decode, NATIVE
from speechbrain.nnet.losses import ctc_loss
from speechbrain.decoders.ctc import ctc_greedy_decode
from ifmdd_losses import pad_flat_targets, build_s3t2_utterance_arcs


def load_reference():
    root = Path('/work/gm64/m64000/IF-MDD')
    # IF-MDD uses a namespace package. Load its actual topology methods intact.
    sys.path.insert(0, str(root))
    spec = importlib.util.spec_from_file_location('ifmdd_reference_topology', root / 'utils/subphonetic_topology.py')
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main():
    torch.manual_seed(42)
    torch.set_num_threads(4)
    x = torch.randn(3, 31, 105, requires_grad=True)
    z = x.detach().clone().requires_grad_()
    lengths = torch.tensor([31, 23, 17])
    token_lengths = torch.tensor([5, 3, 2])
    tokens = torch.tensor([7, 7, 31, 7, 62, 31, 62, 62, 7, 7])
    padded = pad_flat_targets(tokens, token_lengths)
    actual = ctc_objective(x, lengths, tokens, token_lengths)
    expected = ctc_loss(z.log_softmax(-1), padded, lengths / 31., token_lengths / 5., blank_index=0)
    actual.backward(); expected.backward()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(x.grad, z.grad, rtol=0, atol=0)
    changed = x.detach().clone()
    changed[1, 23:] = torch.randn_like(changed[1, 23:]) * 100
    changed[2, 17:] = torch.randn_like(changed[2, 17:]) * 100
    torch.testing.assert_close(ctc_objective(changed, lengths, tokens, token_lengths), actual.detach(), rtol=0, atol=0)
    fixtures = []
    for path, expected_ids in [([7, 7, 7], [7]), ([7, 7, 0, 7, 7], [7, 7]), ([7, 0, 31], [7, 31])]:
        logits = torch.full((1, len(path), 105), -30.)
        logits[0, torch.arange(len(path)), torch.tensor(path)] = 0.
        assert decode(logits, torch.tensor([len(path)])) == [expected_ids]
        fixtures.append(dict(path=path, hypothesis=expected_ids))
    assert decode(x.detach(), lengths) == ctc_greedy_decode(x.detach(), lengths / 31., blank_id=0)
    # Independent topology source checks: original arcs and full forward sum.
    reference = load_reference()
    phone_ids = [3, 3, 7]
    mapping = {p: [1 + (p-1)*3+s for s in range(3)] for p in phone_ids}
    _, ref_arcs, _, final = reference.build_utterance_topology_arcs(phone_ids, mapping, 's3_t2_star_star')
    arcs, actual_final = build_s3t2_utterance_arcs(phone_ids, 's3_t2_star_star')
    assert final == actual_final
    assert sorted((a.src, a.dst, a.label) for a in ref_arcs) == sorted((a.src, a.dst, a.label) for a in arcs)
    scores = torch.randn(15, 313).log_softmax(-1).requires_grad_()
    table = {(0, 0): scores.new_zeros(())}
    for frame in range(len(scores)):
        for node in range(final + 1):
            incoming = [table[(frame, a.src)] + scores[frame, a.label] for a in arcs if a.dst == node and (frame, a.src) in table]
            if incoming:
                table[(frame + 1, node)] = torch.logsumexp(torch.stack(incoming), 0)
    dp_loss = -table[(len(scores), final)]
    import k2
    lines = [f'{a.src} {a.dst} {a.label} 0.0' for a in sorted(ref_arcs,key=lambda a:(a.src,a.dst,a.label))]
    lines += [f'{final} {final+1} -1 0.0', str(final+1)]
    graph = k2.arc_sort(k2.Fsa.from_fsas([k2.Fsa.from_str('\n'.join(lines), acceptor=True)]))
    dense = k2.DenseFsaVec(scores.unsqueeze(0), torch.tensor([[0,0,len(scores)]], dtype=torch.int32))
    graph_loss = k2.ctc_loss(graph, dense, output_beam=100., reduction='none', use_double_scores=True)[0]
    torch.testing.assert_close(dp_loss, graph_loss.to(dp_loss.dtype), rtol=1e-5, atol=1e-5)
    g1, = torch.autograd.grad(dp_loss, scores, retain_graph=True)
    g2, = torch.autograd.grad(graph_loss, scores)
    torch.testing.assert_close(g1, g2, rtol=1e-4, atol=1e-5)
    report = dict(status='passed', ctc_loss=float(actual.detach()),
                  ctc_loss_and_gradient='exact match to IF-MDD selected SpeechBrain criterion',
                  padding='loss invariant to padded frames', decoder_fixtures=fixtures,
                  original_s3t2ss_graph='exact arc match, adjacent identical targets included',
                  topology_dp_loss=float(dp_loss), topology_k2_loss=float(graph_loss),
                  topology_max_gradient_difference=float((g1-g2).abs().max()),
                  explanation='No graph arithmetic mismatch found in this fixture; scaling, objective, architecture and decoding differences remain separate.',
                  speechbrain_loss_sha256=hashlib.sha256(Path(sys.modules[ctc_loss.__module__].__file__).read_bytes()).hexdigest(),
                  source_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in HERE.glob('*.py')})
    (HERE / 'PARITY_PASS.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
