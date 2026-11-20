"""Fetch the unchanged course inputs from a pinned public starter commit."""
import hashlib
import json
from pathlib import Path
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]


def main():
    manifest = json.loads(Path(__file__).with_name('course_data_manifest.json').read_text())
    for relative, expected in manifest['files'].items():
        target = ROOT / relative
        if target.exists():
            if hashlib.sha256(target.read_bytes()).hexdigest() != expected:
                raise ValueError(f'Existing file differs from the course input: {relative}')
            print(f'Verified existing {relative}')
            continue
        url = (f"https://raw.githubusercontent.com/{manifest['repository']}/"
               f"{manifest['revision']}/{relative}")
        request = Request(url, headers={'User-Agent': 'FNLP-Assignment-4-data-setup'})
        with urlopen(request, timeout=60) as response:
            content = response.read()
        if hashlib.sha256(content).hexdigest() != expected:
            raise ValueError(f'Download hash mismatch: {relative}')
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open('xb') as stream:
            stream.write(content)
        print(f'Downloaded and verified {relative}')


if __name__ == '__main__':
    main()
