"""Eight-rank MFA trainer with original upstream jsd and W&B/dev hooks.

Execute the immutable MFA trainer after a scoped AST patch, as the existing
tracking wrapper does. A different loss contract rejects CE optimizer resumes.
"""
import ast
import hashlib
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SUITE = HERE.parent
sys.path.insert(0, str(SUITE))
from mfa_tracked import SOURCE, Tracker


def instrument():
    tree = ast.parse(SOURCE.read_text(), filename=str(SOURCE))
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and node.module == 'model_10ms':
            node.names = [name for name in node.names if name.name != 'MFAZapformer']
    tree.body.append(ast.parse('from mfa_jsd.model_jsd import MFAZapformer').body[0])
    trainer = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'main')
    # This standalone entry exposes only the implemented OTTC arm.
    for node in ast.walk(trainer):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == 'add_argument':
            if node.args and isinstance(node.args[0], ast.Constant) and node.args[0].value == '--arm':
                for kw in node.keywords:
                    if kw.arg == 'choices':kw.value = ast.parse("['ottc']", mode='eval').body
    assignment = next(n for n in trainer.body if isinstance(n, ast.Assign)
                      and any(isinstance(t, ast.Name) and t.id == 'contract' for t in n.targets))
    trainer.body.insert(trainer.body.index(assignment) + 1, ast.parse(
        "contract.update(ottc_cost='jsd_upstream', jsd_definition='0.5 KL(m||p) + 0.5 KL(m||q_eps)', "
        "jsd_implementation='archived OTTC reproduction, jsd=True', "
        "jsd_acoustic_ids=[0] + list(range(3,105)), jsd_source_sha256=jsd_source_hashes)"
    ).body[0])
    logger = next(n for n in trainer.body if isinstance(n, ast.FunctionDef) and n.name == 'log')
    logger.body.append(ast.parse('suite_tracker.log(record, a.output, a.arm)').body[0])
    checkpoint = next(n for n in trainer.body if isinstance(n, ast.FunctionDef) and n.name == 'checkpoint')
    checkpoint.body.append(ast.parse('suite_tracker.progress(progress)').body[0])
    loop = next(n for n in trainer.body if isinstance(n, ast.For)
                and isinstance(n.target, ast.Name) and n.target.id == 'epoch')
    position = trainer.body.index(loop)
    trainer.body[position:position] = ast.parse(
        "suite_tracker.log({'event':'initialization'}, a.output, a.arm)\n"
        'suite_tracker.development(net, a, progress, initial=True)'
    ).body
    loop.body.append(ast.parse('suite_tracker.development(net, a, progress)').body[0])
    return ast.fix_missing_locations(tree)


def main():
    tracker = Tracker()
    source_hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                     for p in [HERE/'loss.py', HERE/'model_jsd.py', Path(__file__).resolve()]}
    from mfa_jsd.loss import SOURCE as loss_source, SOURCE_SHA256
    source_hashes[str(loss_source)] = SOURCE_SHA256
    namespace = {'__file__': str(SOURCE), '__name__': 'mfa_jsd_source',
                 'suite_tracker': tracker, 'jsd_source_hashes': source_hashes}
    exec(compile(instrument(), str(SOURCE), 'exec'), namespace)
    try:
        namespace['main']()
    except BaseException:
        tracker.finish(1)
        raise
    tracker.finish(0)


if __name__ == '__main__':main()
