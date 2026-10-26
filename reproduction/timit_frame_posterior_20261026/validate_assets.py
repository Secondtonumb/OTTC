"""Validate exported browser payload and scripts without changing probabilities."""
import copy
import hashlib
import json
from pathlib import Path
import re
import subprocess
import tempfile

HERE = Path(__file__).resolve().parent
OUTPUT = HERE / 'seed43/FADG0_SI1279'
CONTRACT = HERE.parent / 'timit_posterior_20261026/site_integration'


def main():
    data_path = OUTPUT / 'timit_framewise_FADG0_SI1279_seed43.posterior-v1.json'
    data = json.loads(data_path.read_text())
    assert len(data['methods']) == 4 and len(data['phones']) == 42 and len(data['time_ms']) == 180
    assert data['source_type'] == 'model'
    assert data['time_ms'] == [12.46875+10*i for i in range(180)]
    assert data['metadata']['reference_conditioned_inference'] is False
    assert '/work/' not in data_path.read_text() and '/home/' not in data_path.read_text()
    with tempfile.TemporaryDirectory(prefix='frame_posterior_check_') as temporary:
        tmp = Path(temporary)
        validator = (CONTRACT / 'assets/format.js').read_text().replace('export ', '')
        check = validator + '\n' + r'''
const fs = require('fs');
const data = JSON.parse(fs.readFileSync(process.argv[2], 'utf8'));
validateDataset(data);
const bad = structuredClone(data);
bad.methods[0].posterior[0] = bad.methods[0].posterior[0].map(v=>v/2);
let rejected = false;
try { validateDataset(bad); } catch { rejected = true; }
if (!rejected) throw new Error('Incomplete distribution accepted');
console.log('Dataset validator passed; incomplete distributions rejected');
'''
        script = tmp / 'validate.cjs'
        script.write_text(check)
        subprocess.run(['node', str(script), str(data_path)], check=True)
        html = (OUTPUT / 'posterior_3d_interactive.html').read_text()
        count = 0
        for attrs, body in re.findall(r'<script([^>]*)>([\s\S]*?)</script>', html):
            if 'application/json' in attrs:
                assert json.loads(body) == data
                continue
            assert 'src=' not in attrs
            script = tmp / f'inline-{count}.js'
            script.write_text(body)
            subprocess.run(['node', '--check', str(script)], check=True)
            count += 1
        assert count == 2
        assert 'data:audio/wav;base64,' in html and 'data:image/png;base64,' in html
    report = dict(site_validator_passed=True, incomplete_distribution_rejected=True,
        javascript_syntax_passed=True, scripts=count, methods=[m['id'] for m in data['methods']],
        shape=[4, 180, 42], time_range_ms=[data['time_ms'][0], data['time_ms'][-1]],
        embedded_audio=True, embedded_overview=True, private_paths_in_combined_JSON=False,
        file_sha256=hashlib.sha256(data_path.read_bytes()).hexdigest())
    (OUTPUT / 'SITE_VALIDATION.json').write_text(json.dumps(report, indent=2)+'\n')
    print('SITE_VALIDATION_PASS', report['shape'])


if __name__ == '__main__':
    main()
