"""Select the independently validated full40 source for future evaluations."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / 'reproduction/data/buckeye_full40_20261022'
DEFAULTS = ROOT / 'reproduction/dataset_defaults.json'


def main():
    complete = DATA/'PREPARE_COMPLETE.json'
    report = json.loads(complete.read_text())
    assert report['validated'] and report['source_tracks'] == 255 and report['source_speakers'] == 40
    value = json.loads(DEFAULTS.read_text()) if DEFAULTS.exists() else {}
    value.setdefault('evaluation', {})['buckeye'] = dict(
        dataset_id='buckeye_full40_20261022', data_root=str(DATA.relative_to(ROOT)),
        prepare_complete_sha256=hashlib.sha256(complete.read_bytes()).hexdigest(),
        source_archive='/work/gm64/m64000/dataset/buckeye_miyabi_2026-10-22.tar.gz',
        default_split='test', inventory='ottc_legacy42',
        policy_date='2026-10-24', requested_by='user', evaluation_uploads=False,
        policy='All future Buckeye tests use the new full40 source. Preserve historical '
               'completed artifacts and active resume/source contracts; new evaluations '
               'use reproduction.buckeye_dataset.load_buckeye_rows.',
        raw_annotation_root=str((DATA/'raw/data').relative_to(ROOT)))
    temp = DEFAULTS.with_suffix('.json.tmp')
    temp.write_text(json.dumps(value,indent=2)+'\n')
    temp.replace(DEFAULTS)
    print(DEFAULTS)


if __name__ == '__main__':
    main()
