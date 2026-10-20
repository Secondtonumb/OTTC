"""Add standard JSD without changing the validated CE/upstream-JSD sources."""
import bootstrap
from bootstrap import HERE
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import train


def configure():
    train.OBJECTIVES['jsd_standard'] = 'jsd_standard'
    original_hashes = train.hashes

    def hashes():
        result = original_hashes()
        source = Path(__file__).resolve()
        result[str(source)] = hashlib.sha256(source.read_bytes()).hexdigest()
        return result

    train.hashes = hashes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['train', 'evaluate', 'budget'])
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--max-wall-seconds', type=float, default=5900)
    args = parser.parse_args()
    configure()
    forwarded = ['--objective', 'jsd_standard'] + (['--smoke'] if args.smoke else [])
    if args.mode in ('train', 'budget'):
        output = train.model_directory('jsd_standard', args.smoke)
        output.mkdir(parents=True, exist_ok=True)
        (output/'standard_jsd_adapter_snapshot.py').write_bytes(Path(__file__).read_bytes())
        if args.mode == 'budget':
            assert not args.smoke
            import budget_train
            sys.argv = ['budget_train.py', '--objective', 'jsd_standard',
                        '--max-wall-seconds', str(args.max_wall_seconds)]
            budget_train.main()
        else:
            sys.argv = ['train.py'] + forwarded
            train.main()
    else:
        # Transformers imports the unrelated installed package named evaluate.
        # Load this experiment's evaluator by absolute path to avoid that cache.
        spec = importlib.util.spec_from_file_location('buckeye_evaluate_standard', HERE/'evaluate.py')
        evaluate = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(evaluate)
        original_publish = evaluate.publish

        def publish(result, output, smoke):
            result['protocol']['inference'] = 'Audio-only FP32 native42-class logits; identical evaluator for CE/upstream JSD/standard JSD'
            result['protocol']['transport_cost'] = 'standard JSD: 0.5 KL(p || m) + 0.5 KL(q || m), m=(p+q)/2; existing jsd_standard branch'
            result['standard_jsd_adapter_sha256'] = train.hashes()[str(Path(__file__).resolve())]
            (output/'standard_jsd_adapter_snapshot.py').write_bytes(Path(__file__).read_bytes())
            (output/'metrics.json').write_text(json.dumps(result, indent=2)+'\n')
            original_publish(result, output, smoke)

        evaluate.publish = publish
        sys.argv = ['evaluate.py'] + forwarded
        evaluate.main()


if __name__ == '__main__':
    main()
