"""Verify the distributed model/velocity-definition hashes without loading models."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    manifest = json.loads((ROOT/'assets/manifest.json').read_text())
    for name, expected in manifest['files'].items():
        path = ROOT/name
        if not path.is_file():
            raise FileNotFoundError(f'Missing model asset: {name}')
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected['sha256'] or path.stat().st_size != expected['bytes']:
            raise RuntimeError(f'Asset checksum mismatch: {name}')
    print(json.dumps(dict(status='pass', verified_files=len(manifest['files']))))


if __name__ == '__main__':
    main()
