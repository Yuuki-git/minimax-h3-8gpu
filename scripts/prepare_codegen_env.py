"""Create a build-only source overlay; vendor libraries stay read-only."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile


def main():
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
    main()
