"""Fetch immutable public source revisions and the one private-overlay wheel."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import urllib.request

ENTRY = Path(__file__).resolve().parents[1]


def fetch(root, name, item):
    target = root / name
    if not target.exists():
        subprocess.run(['git', 'init', '-q', str(target)], check=True)
        subprocess.run(['git', '-C', str(target), 'remote', 'add', 'origin', item['url']], check=True)
        subprocess.run(['git', '-C', str(target), 'fetch', '--depth', '1', 'origin', item['commit']], check=True)
        subprocess.run(['git', '-c', 'core.autocrlf=false', '-C', str(target),
                        'checkout', '--detach', 'FETCH_HEAD'], check=True)
    actual = subprocess.check_output(['git', '-C', str(target), 'rev-parse', 'HEAD'], text=True).strip()
    if actual != item['commit']:
        raise ValueError(f'{name}: expected {item["commit"]}, got {actual}')
    if subprocess.check_output(['git', '-C', str(target), 'status', '--porcelain',
                                '--untracked-files=no', '--ignore-submodules=dirty'], text=True).strip():
        raise ValueError(f'{name}: tracked files changed')
    for relative, expected in item.get('files', {}).items():
        if hashlib.sha256((target / relative).read_bytes()).hexdigest() != expected:
            raise ValueError(f'{name}: source digest differs: {relative}')
    modules = {'InfiniCore': ['third_party/spdlog', 'submodules/InfiniRT', 'submodules/InfiniOps', 'submodules/InfiniCCL'],
               'InfiniLM': ['third_party/spdlog', 'third_party/json']}.get(name, [])
    if modules:
        subprocess.run(['git', '-c', 'core.autocrlf=false', '-C', str(target), 'submodule',
                        'update', '--init', '--recursive', '--depth', '1', '--jobs', '8', '--', *modules], check=True)
    print(f'SOURCE_COMMIT={name} {actual}', flush=True)


def main():
    lock = json.loads((ENTRY / 'source-lock.json').read_bytes())
    work = Path(os.environ['SS5_OPT_WORK']).resolve(strict=True)
    sources = work / 'src'
    sources.mkdir(exist_ok=True)
    for name, item in lock['sources'].items():
        fetch(sources, name, item)
    wheels = work / 'sources'
    wheels.mkdir(exist_ok=True)
    dep = lock['build_dependencies']
    wheel = wheels / dep['numpy_wheel']
    if not wheel.exists():
        with urllib.request.urlopen('https://pypi.org/pypi/numpy/1.26.4/json', timeout=60) as response:
            metadata = json.load(response)
        item, = [f for f in metadata['urls'] if f['filename'] == wheel.name and f['digests']['sha256'] == dep['numpy_sha256']]
        temporary = wheel.with_suffix('.download')
        with urllib.request.urlopen(item['url'], timeout=120) as response, temporary.open('xb') as output:
            while data := response.read(1024 * 1024):
                output.write(data)
        if hashlib.sha256(temporary.read_bytes()).hexdigest() != dep['numpy_sha256']:
            raise ValueError('NumPy download SHA256 differs')
        temporary.rename(wheel)
    if hashlib.sha256(wheel.read_bytes()).hexdigest() != dep['numpy_sha256']:
        raise ValueError('NumPy wheel SHA256 differs')
    (work / 'source-lock.json').write_bytes((ENTRY / 'source-lock.json').read_bytes())


if __name__ == '__main__':
    main()
