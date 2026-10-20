"""Render the exact archived decoding FSA, rather than a schematic substitute."""
import hashlib
import json
from pathlib import Path
import subprocess

HERE = Path(__file__).resolve().parent
TOKENS_FILE = Path('/work/gm64/m64000/dataset/english-mfa31-10ms-all-train-v2/tokens.txt')
TOKENS = {int(line.rsplit(' ', 1)[1]): line.rsplit(' ', 1)[0]
          for line in TOKENS_FILE.read_text().splitlines()}


def quote(s):
    return json.dumps(str(s), ensure_ascii=False)


def render(name, title, arcs, final, colors=None):
    nodes = {a[0] for a in arcs} | {a[1] for a in arcs}
    lines = ['digraph FSA {', 'rankdir=LR;',
             'graph [fontname="DejaVu Sans", fontsize=18, labelloc=t, label=' + quote(title) + '];',
             'node [shape=circle, fontname="DejaVu Sans", fontsize=14];',
             'edge [fontname="DejaVu Sans", fontsize=12];',
             'start [shape=point]; start -> 0;']
    for node in sorted(nodes):
        node_label = f'{node}\nphone boundary' if node == 0 else str(node)
        lines.append(f'{node} [label={quote(node_label)}, shape={"doublecircle" if node == final else "circle"}];')
    for src, dst, acoustic, phone, weight in arcs:
        if acoustic == -1:
            label = 'final (no acoustic frame) / -1'
        elif acoustic == 0:
            label = 'blank [0] / eps'
        else:
            token = (acoustic - 1) // 3 + 1
            state = (acoustic - 1) % 3
            label = f'{TOKENS[token]}{state} [{acoustic}] / ' + (f'{TOKENS[phone]} [{phone}]' if phone else 'eps')
        attributes = 'label=' + quote(label)
        if acoustic > 0 and phone > 0:
            attributes += ', color="#c53a30", fontcolor="#a82b23", penwidth=2'
        elif src == dst:
            attributes += ', color="#2877a9", fontcolor="#23698e"'
        if colors and (src, dst) in colors:
            attributes += ', style=bold'
        lines.append(f'{src} -> {dst} [{attributes}];')
    lines.append('}')
    dot_file = HERE / f'{name}.dot'
    dot_file.write_text('\n'.join(lines) + '\n')
    for suffix in ('svg', 'png', 'pdf'):
        subprocess.run(['dot', '-T' + suffix, str(dot_file), '-o', str(HERE / f'{name}.{suffix}')], check=True)


def main():
    fsa = HERE / 'phone_loop.fst.txt'
    rows = [line.split() for line in fsa.read_text().splitlines()]
    final = int(rows[-1][0])
    arcs = [tuple(map(int, row[:4])) + (float(row[4]),) for row in rows[:-1]]
    b_id = next(i for i, phone in TOKENS.items() if phone == 'b')
    b_states = {1 + (b_id - 1) * 3 + i for i in range(3)}
    small = [a for a in arcs if a[2] in b_states or a[2] <= 0]
    render('fsa_b_exact', 'Actual S3T2** decoding FSA: b branch\nArc: acoustic input [ID] / phone output [ID]. Red emits b; blue is a self-loop.', small, final)
    render('fsa_full_exact', 'Actual S3T2** free-phone-loop decoding FSA (104 phones)\nEvery arc copied from phone_loop.fst.txt. No LM, no insertion penalty.', arcs, final)
    # A compact second diagram exposes a full occurrence followed by a restart.
    example = HERE / 'jsd_fs/final_development_predictions.jsonl'
    row = json.loads(example.open().readline())
    path = row['legal_state_path'][:12]
    graph_arcs = {(a[0], a[2]): [] for a in arcs}
    for a in arcs:
        graph_arcs[(a[0], a[2])].append(a)
    # This literal 12-frame segment is shown only after matching the saved path.
    assert path == [19, 21, 0, 19, 19, 21, 0, 19, 19, 21, 19, 21]
    lines = ['digraph Path {', 'rankdir=LR;',
        'graph [fontname="DejaVu Sans", labelloc=t, fontsize=17, label="Saved Viterbi path, first 120 ms: four complete b occurrences"];',
        'node [shape=circle, fontname="DejaVu Sans"]; edge [fontname="DejaVu Sans"];']
    for i in range(13):
        lines.append(f'{i} [label="{i * 10}ms"];')
    for i, acoustic in enumerate(path):
        token = 0 if acoustic == 0 else (acoustic - 1) // 3 + 1
        state = -1 if acoustic == 0 else (acoustic - 1) % 3
        emission = acoustic > 0 and state == 0 and (i == 0 or path[i - 1] != acoustic)
        text = 'blank / eps' if acoustic == 0 else f'{TOKENS[token]}{state} / ' + (TOKENS[token] if emission else 'eps')
        color = '#c53a30' if emission else '#2877a9'
        lines.append(f'{i} -> {i + 1} [label={quote(text)}, color="{color}"];')
    lines.append('}')
    file = HERE / 'fsa_actual_b_path.dot'
    file.write_text('\n'.join(lines) + '\n')
    for suffix in ('svg', 'png', 'pdf'):
        subprocess.run(['dot', '-T' + suffix, str(file), '-o', str(HERE / f'fsa_actual_b_path.{suffix}')], check=True)
    proof = dict(source=str(fsa), source_sha256=hashlib.sha256(fsa.read_bytes()).hexdigest(),
                 total_arcs=len(arcs), total_phone_branches=104, displayed_phone='b', phone_id=b_id,
                 displayed_acoustic_ids=sorted(b_states), final_node=final,
                 b_branch_arcs=small, example_id=row['id'], actual_first_12_frames=path,
                 graph_figures_are_copied_from_actual_fsa=True)
    (HERE / 'FSA_FIGURES.json').write_text(json.dumps(proof, indent=2) + '\n')
    print(json.dumps(proof))


if __name__ == '__main__':
    main()
