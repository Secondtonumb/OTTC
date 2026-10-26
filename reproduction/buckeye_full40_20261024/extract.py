"""Stage the verified user package, omitting its duplicate speaker ZIP archives."""
import gzip
import hashlib
import json
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCE = Path('/work/gm64/m64000/dataset/buckeye_miyabi_2026-10-22.tar.gz')
DEST = ROOT / 'reproduction/data/buckeye_full40_20261022/raw'


class HashReader:
    def __init__(self, stream):
        self.stream, self.hash = stream, hashlib.sha256()

    def read(self, n=-1):
        data = self.stream.read(n)
        self.hash.update(data)
        return data


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    expected, checked = {}, {}
    with SOURCE.open('rb') as source:
        wrapped = HashReader(source)
        with gzip.GzipFile(fileobj=wrapped, mode='rb') as gz:
            with tarfile.open(fileobj=gz, mode='r|') as archive:
                for member in archive:
                    if not member.isfile():
                        continue
                    name = member.name.removeprefix('buckeye/')
                    relative = Path(name)
                    if relative.is_absolute() or '..' in relative.parts:
                        raise ValueError(name)
                    if name.startswith('archives/'):
                        continue
                    path = DEST / relative
                    path.parent.mkdir(parents=True, exist_ok=True)
                    temp = path.with_name(path.name + '.tmp')
                    digest = hashlib.sha256()
                    with archive.extractfile(member) as src, temp.open('wb') as dst:
                        while chunk := src.read(4 * 1024 * 1024):
                            digest.update(chunk)
                            dst.write(chunk)
                    if name == 'PACKAGE_CONTENTS.sha256':
                        expected = {line[66:]: line[:64]
                                    for line in temp.read_text().splitlines()}
                    elif digest.hexdigest() != expected[name]:
                        raise ValueError('Package hash mismatch: ' + name)
                    if path.exists():
                        existing = hashlib.sha256(path.read_bytes()).hexdigest()
                        if existing != digest.hexdigest():
                            raise FileExistsError('Refusing to overwrite different data: ' + str(path))
                        temp.unlink()
                    else:
                        temp.replace(path)
                    checked[name] = dict(bytes=member.size, sha256=digest.hexdigest())
                    if len(checked) % 100 == 0:
                        print('staged', len(checked), flush=True)
            while gz.read(4 * 1024 * 1024):
                pass
        while wrapped.read(4 * 1024 * 1024):
            pass
    digest = wrapped.hash.hexdigest()
    assert digest == SOURCE.with_suffix('.gz.sha256').read_text().split()[0]
    missing = sorted(set(expected) - set(checked) - {p for p in expected if p.startswith('archives/')})
    assert not missing, missing
    report = dict(source=str(SOURCE), source_sha256=digest, gzip_crc_passed=True,
                  destination=str(DEST), duplicate_archives_staged=False,
                  verified_files=checked, missing_files=missing)
    path = DEST.parent / 'EXTRACT_COMPLETE.json'
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n')
    print('EXTRACT_COMPLETE', path, flush=True)


if __name__ == '__main__':
    main()
