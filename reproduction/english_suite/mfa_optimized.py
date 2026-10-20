"""Apply previously measured microbatch limits without editing the MFA source contract."""
import hashlib
import importlib.util
import json
import os
import shutil
import sys
import time
from pathlib import Path

import microbatching

ROOT = Path(__file__).resolve().parent
FACTORS = {'ctc': 4, 'ottc': 4, 'spottc': 2, 'film_k4': 4}


def main():
    arm = sys.argv[sys.argv.index('--arm')+1]
    original = microbatching.LIMITS[arm]
    microbatching.LIMITS[arm] = tuple(v*FACTORS[arm] for v in original)
    output = Path(sys.argv[sys.argv.index('--output')+1])
    if int(os.environ.get('RANK', '0')) == 0:
        output.mkdir(parents=True, exist_ok=True)
        if (output/'run_config.json').exists():
            history=output/'launch_history'/str(time.time_ns())
            history.mkdir(parents=True)
            for name in ['run_config.json','throughput_change.json']:
                if (output/name).exists():
                    shutil.copy2(output/name,history/name)
        evidence = Path('/work/gm64/m64000/zipa/mfa_miyabi_20260922/batch_probe_3418596.opbs/results.json')
        (output/'throughput_change.json').write_text(json.dumps({
            'arm': arm, 'factor': FACTORS[arm], 'original_limits': original,
            'new_limits': microbatching.LIMITS[arm], 'job': os.environ.get('PBS_JOBID'),
            'evidence': str(evidence), 'evidence_sha256': hashlib.sha256(evidence.read_bytes()).hexdigest(),
            'logical_batch_loss_optimizer_sampler': 'unchanged; microbatch partition only',
            'numerics': 'Different partitions can change stochastic masks and floating-point rounding; not bitwise reproduction',
            'wrapper_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}, indent=2)+'\n')
    spec = importlib.util.spec_from_file_location('mfa_tracking', ROOT/'mfa_tracked.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.main()


if __name__ == '__main__':
    main()
