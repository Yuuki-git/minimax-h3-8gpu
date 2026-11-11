"""Prepare a private environment from public source and organizer vendor packages."""
import hashlib
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import subprocess
import shutil
import zipfile
import sys
import urllib.request

ENTRY = Path(__file__).resolve().parents[1]


def fetch(root, name, item, submodules=()):
    target = root / name
    if not target.exists():
        subprocess.run(['git', 'init', '-q', str(target)], check=True)
        subprocess.run(['git', '-C', str(target), 'remote', 'add', 'origin', item['url']], check=True)
        subprocess.run(['git', '-C', str(target), 'fetch', '--depth', '1', 'origin', item['commit']], check=True)
        subprocess.run(['git', '-c', 'core.autocrlf=false', '-C', str(target), 'checkout', '--detach', 'FETCH_HEAD'], check=True)
    actual = subprocess.check_output(['git', '-C', str(target), 'rev-parse', 'HEAD'], text=True).strip()
    if len(item['commit']) == 40 and actual != item['commit']:
        raise ValueError(f'Unexpected {name} commit: {actual}')
    if submodules:
        subprocess.run(['git', '-C', str(target), 'submodule', 'update', '--init', '--recursive',
                        '--depth', '1', '--jobs', '8', '--', *submodules], check=True)
    print(f'SOURCE={name} {actual}', flush=True)


def download(url, target, expected):
    if not target.exists():
        with urllib.request.urlopen(url, timeout=120) as source, target.open('wb') as out:
            while block := source.read(1024 * 1024):
                out.write(block)
    if hashlib.sha256(target.read_bytes()).hexdigest() != expected:
        raise ValueError(f'Download SHA256 differs: {target.name}')


def main():
    if sys.version_info[:2] != (3, 11):
        raise ValueError('Use organizer Python 3.11, optionally through SS5_VENDOR_PYTHON')
    work = Path(os.environ['SS5_OPT_WORK']).resolve(strict=True)
    vendor = {name: metadata.version(name) for name in ('torch', 'triton', 'flash-attn')}
    if vendor != {'torch': '2.9.0+das.opt1.dtk2604', 'triton': '3.5.1+das.opt1.dtk2604.torch290',
                  'flash-attn': '2.6.1+das.opt1.dtk2604.torch290'}:
        raise ValueError(f'Use the organizer gfx936 vendor environment: {vendor}')
    sites = [p for p in sys.path if p and ('site-packages' in p or 'dist-packages' in p)]
    extensions = [p for root in sites for p in Path(root).glob('flash_attn_2_cuda*.so')]
    if not extensions:
        raise ValueError('The organizer vendor FlashAttention extension must be available')
    os.environ['FA_SO_PATH'] = str(extensions[0].resolve())
    runtime = work / 'venv'
    if not runtime.exists():
        subprocess.run([sys.executable, '-B', '-m', 'venv', str(runtime)], check=True)
    python = str(runtime / 'bin/python')
    site = Path(subprocess.check_output([python, '-B', '-c', 'import sysconfig; print(sysconfig.get_path("purelib"))'], text=True).strip())
    (site / 'organizer_readonly.pth').write_text('\n'.join(sites) + '\n')
    constraints = work / 'vendor-constraints.txt'
    constraints.write_text(''.join(f'{name}=={version}\n' for name, version in vendor.items()))
    subprocess.run([python, '-B', '-m', 'pip', 'install', '--constraint', str(constraints),
                    '--requirement', str(ENTRY / 'requirements.txt')], check=True)
    sources = work / 'src'
    sources.mkdir(exist_ok=True)
    lock = json.loads((ENTRY / 'source-lock.json').read_bytes())
    modules = {'InfiniCore': ('third_party/spdlog', 'submodules/InfiniRT', 'submodules/InfiniOps', 'submodules/InfiniCCL'),
               'InfiniLM': ('third_party/spdlog', 'third_party/json')}
    for name, item in lock['sources'].items():
        fetch(sources, name, item, modules.get(name, ()))
    fetch(sources, 'xmake', {'url': 'https://github.com/xmake-io/xmake.git',
                            'commit': lock['build_dependencies']['xmake_commit']}, ('.',))
    fetch(sources, 'xmake-repo', {'url': 'https://github.com/xmake-io/xmake-repo.git',
                                'commit': lock['build_dependencies']['xmake_repo_commit']})
    # Compile torchvision against the preserved vendor Torch if the image lacks it.
    probe = subprocess.run([python, '-B', '-c', 'import torchvision; assert torchvision.__version__.startswith("0.24.0")'])
    if probe.returncode:
        fetch(sources, 'torchvision', {'url': 'https://github.com/pytorch/vision.git', 'commit': 'v0.24.0'})
        subprocess.run([python, '-B', '-m', 'pip', 'install', '--no-deps', '--no-build-isolation',
                        str(sources / 'torchvision')], check=True)
    deps = work / 'deps/xmake-pkgs'
    deps.mkdir(parents=True, exist_ok=True)
    for url, name, sha in (
        ('https://github.com/boostorg/boost/releases/download/boost-1.92.0/boost-1.92.0-cmake.tar.gz',
         'boost-1.92.0.tar.gz', 'f51707c27359a0df0cac1beada86de31bb5eed5e8285592dadec384df99c2984'),
        ('https://github.com/pybind/pybind11/archive/refs/tags/v3.1.0.zip',
         'pybind11-v3.1.0.zip', 'affea1ada7b39fe1d835559fcb78800c7927d514bd480ed01b71b88205a5e536'),
    ):
        download(url, deps / name, sha)
    wheels = work / 'sources'
    wheels.mkdir(exist_ok=True)
    numpy = lock['build_dependencies']
    with urllib.request.urlopen('https://pypi.org/pypi/numpy/1.26.4/json', timeout=60) as response:
        data = json.load(response)
    item, = [x for x in data['urls'] if x['filename'] == numpy['numpy_wheel']]
    download(item['url'], wheels / item['filename'], numpy['numpy_sha256'])
    (work / 'source-lock.json').write_bytes((ENTRY / 'source-lock.json').read_bytes())
    subprocess.run([python, '-B', str(Path(__file__).resolve()), '--codegen'], check=True)
    (work / 'environment.json').write_text(json.dumps({'vendor_python': sys.executable, 'vendor': vendor}, indent=2) + '\n')


def prepare_codegen():
    work = Path(os.environ['SS5_OPT_WORK']).resolve(strict=True)
    lock = json.loads((work / 'source-lock.json').read_bytes())
    source = work / 'src/ninetoothed-ss5'
    venv = work / 'codegen-venv'
    baseline = {'python': sys.version, 'executable': sys.executable,
                'site_paths': [p for p in sys.path if p and ('site-packages' in p or 'dist-packages' in p)]}
    (work / 'baseline-python.json').write_text(json.dumps(baseline, indent=2) + '\n')
    if not venv.exists():
        subprocess.run([sys.executable, '-B', '-m', 'venv', '--without-pip', str(venv)], check=True)
    python = venv / 'bin/python'
    site = Path(subprocess.check_output([str(python), '-B', '-c',
        'import sysconfig;print(sysconfig.get_path("purelib"))'], text=True).strip())
    if not site.resolve().is_relative_to(work):
        raise ValueError('Build environment escaped workspace')
    (site / 'ss5_baseline_readonly.pth').write_text('\n'.join(baseline['site_paths']) + '\n')
    shutil.copytree(source / 'ninetoothed', site / 'ninetoothed', dirs_exist_ok=True)
    metadata = site / 'ninetoothed-0.26.0+ss5.1.dist-info'
    metadata.mkdir(exist_ok=True)
    (metadata / 'METADATA').write_text('Metadata-Version: 2.4\nName: ninetoothed\nVersion: 0.26.0+ss5.1\n'
        'Requires-Python: >=3.10\nRequires-Dist: numpy>=1.26.4\nRequires-Dist: sympy>=1.13.0\nRequires-Dist: triton>=3.0.0\n')
    (metadata / 'INSTALLER').write_text('ss5-locked-source-overlay\n')
    wheel = work / 'sources' / lock['build_dependencies']['numpy_wheel']
    if hashlib.sha256(wheel.read_bytes()).hexdigest() != lock['build_dependencies']['numpy_sha256']:
        raise ValueError('NumPy wheel SHA256 differs')
    with zipfile.ZipFile(wheel) as archive:
        for member in archive.namelist():
            if member.endswith('/'):
                continue
            target = (site / member).resolve()
            if not target.is_relative_to(site.resolve()):
                raise ValueError('Invalid wheel path')
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(archive.read(member))
    print(f'CODEGEN_PYTHON={python}')

if __name__ == '__main__':
    if '--codegen' in sys.argv:
        prepare_codegen()
    else:
        main()
