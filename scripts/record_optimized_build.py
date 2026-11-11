import hashlib
import importlib.metadata
import json
import os
from pathlib import Path


def main():
    root = Path(os.environ['SS5_OPT_WORK']).resolve(strict=True)
    paths = [*sorted((root / 'infini/lib').glob('*.so'))]
    paths += [p for repo, pattern in [('InfiniCore', '_infinicore*.so'), ('InfiniLM', '_infinilm*.so')]
              for p in (root / 'src' / repo / 'python').rglob(pattern)]
    if len(paths) < 9:
        raise ValueError(f'Incomplete native installation: {paths}')
    record = {'sources': json.loads((root / 'source-lock.json').read_bytes()),
              'runtime_packages': {n: importlib.metadata.version(n) for n in ('torch', 'triton', 'numpy', 'flash-attn')},
              'files': {str(p.relative_to(root)): {'bytes': p.stat().st_size,
                       'sha256': hashlib.sha256(p.read_bytes()).hexdigest()} for p in paths}}
    (root / 'build-manifest.json').write_text(json.dumps(record, indent=2) + '\n')
    print('BUILD_MANIFEST=' + str(root / 'build-manifest.json'))


if __name__ == '__main__':
    main()
