"""Download hash-pinned support inputs without installing them on this machine."""

import argparse
import hashlib
import json
from pathlib import Path
import urllib.request
import zipfile


METRICS = {'python': 'CPython 3.11',
 'platform': 'manylinux_2_17_x86_64',
 'packages': [{'name': 'scikit-image',
               'version': '0.25.2',
               'filename': 'scikit_image-0.25.2-cp311-cp311-manylinux_2_17_x86_64.manylinux2014_x86_64.whl',
               'url': 'https://files.pythonhosted.org/packages/dd/9b/9fb556463a34d9842491d72a421942c8baff4281025859c84fcdb5e7e602/scikit_image-0.25.2-cp311-cp311-manylinux_2_17_x86_64.manylinux2014_x86_64.whl',
               'sha256': '24cc986e1f4187a12aa319f777b36008764e856e5013666a4a83f8df083c2641',
               'size': 14785376,
               'requires_dist': ['numpy>=1.24',
                                 'scipy>=1.11.4',
                                 'networkx>=3.0',
                                 'pillow>=10.1',
                                 'imageio!=2.35.0,>=2.33',
                                 'tifffile>=2022.8.12',
                                 'packaging>=21',
                                 'lazy-loader>=0.4'],
               'requires_python': '>=3.10'},
              {'name': 'scipy',
               'version': '1.15.3',
               'filename': 'scipy-1.15.3-cp311-cp311-manylinux_2_17_x86_64.manylinux2014_x86_64.whl',
               'url': 'https://files.pythonhosted.org/packages/bd/37/89f19c8c05505d0601ed5650156e50eb881ae3918786c8fd7262b4ee66d3/scipy-1.15.3-cp311-cp311-manylinux_2_17_x86_64.manylinux2014_x86_64.whl',
               'sha256': '39cb9c62e471b1bb3750066ecc3a3f3052b37751c7c3dfd0fd7e48900ed52982',
               'size': 37652622,
               'requires_dist': ['numpy<2.5,>=1.23.5'],
               'requires_python': '>=3.10'},
              {'name': 'lazy-loader',
               'version': '0.4',
               'filename': 'lazy_loader-0.4-py3-none-any.whl',
               'url': 'https://files.pythonhosted.org/packages/83/60/d497a310bde3f01cb805196ac61b7ad6dc5dcf8dce66634dc34364b20b4f/lazy_loader-0.4-py3-none-any.whl',
               'sha256': '342aa8e14d543a154047afb4ba8ef17f5563baad3fc610d7b15b213b0f119efc',
               'size': 12097,
               'requires_dist': ['packaging', 'importlib-metadata; python_version < "3.8"'],
               'requires_python': '>=3.7'},
              {'name': 'networkx',
               'version': '3.4.2',
               'filename': 'networkx-3.4.2-py3-none-any.whl',
               'url': 'https://files.pythonhosted.org/packages/b9/54/dd730b32ea14ea797530a4479b2ed46a6fb250f682a9cfb997e968bf0261/networkx-3.4.2-py3-none-any.whl',
               'sha256': 'df5d4365b724cf81b8c6a7312509d0c22386097011ad1abe274afd5e9d3bbc5f',
               'size': 1723263,
               'requires_dist': [],
               'requires_python': '>=3.10'},
              {'name': 'tifffile',
               'version': '2024.8.30',
               'filename': 'tifffile-2024.8.30-py3-none-any.whl',
               'url': 'https://files.pythonhosted.org/packages/3a/4f/73714b1c1d339b1545cac28764e39f88c69468b5e10e51f327f9aa9d55b9/tifffile-2024.8.30-py3-none-any.whl',
               'sha256': '8bc59a8f02a2665cd50a910ec64961c5373bee0b8850ec89d3b7b485bf7be7ad',
               'size': 227262,
               'requires_dist': ['numpy'],
               'requires_python': '>=3.9'}],
 'installer_sha256': '532b49dcf3100aff32b2b41c124f7b960a1925781f3a6ad63af3e830bd0f8224'}

DIFFUSERS = {'name': 'diffusers',
 'version': '0.40.0',
 'filename': 'diffusers-0.40.0-py3-none-any.whl',
 'url': 'https://files.pythonhosted.org/packages/db/df/ffb593ebed2a068d2d6be44261283f39a6b809c0fcdfdcafbd448cbeec77/diffusers-0.40.0-py3-none-any.whl',
 'sha256': '5b5da7c3ddb62152fa4afc577f02e050af688c797375c34fb2d01006da3f3541',
 'size': 5911654,
 'metadata_url': 'https://pypi.org/pypi/diffusers/0.40.0/json'}

CPP = [{'name': 'boost',
  'version': '1.92.0',
  'filename': 'boost-1.92.0.tar.gz',
  'url': 'https://github.com/boostorg/boost/releases/download/boost-1.92.0/boost-1.92.0-cmake.tar.gz',
  'sha256': 'f51707c27359a0df0cac1beada86de31bb5eed5e8285592dadec384df99c2984'},
 {'name': 'pybind11',
  'version': '3.1.0',
  'filename': 'pybind11-v3.1.0.zip',
  'url': 'https://github.com/pybind/pybind11/archive/refs/tags/v3.1.0.zip',
  'sha256': 'affea1ada7b39fe1d835559fcb78800c7927d514bd480ed01b71b88205a5e536'}]


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(item, directory):
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / item["filename"]
    if path.exists():
        if sha256(path) == item["sha256"]:
            return path
        raise ValueError(f"Existing download SHA256 mismatch {path}")
    try:
        with urllib.request.urlopen(item["url"], timeout=120) as source, path.open("xb") as destination:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                destination.write(chunk)
        if sha256(path) != item["sha256"]:
            raise ValueError(f"Downloaded SHA256 mismatch {path}")
    except BaseException:
        path.unlink(missing_ok=True)
        raise
    return path


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=root / "dist/downloads")
    parser.add_argument("--offline", action="store_true", help="Use already downloaded, verified files")
    parser.add_argument("--only", choices=("all", "metrics", "cpp", "diffusers"), default="all")
    args = parser.parse_args()
    destination = root / "dist"
    destination.mkdir(exist_ok=True)
    metrics = METRICS
    diffusers = DIFFUSERS
    cpp = CPP
    groups = [(item, args.cache) for item in metrics["packages"]] if args.only in ("all", "metrics") else []
    if args.only in ("all", "diffusers"):
        groups.append((diffusers, destination))
    if args.only in ("all", "cpp"):
        groups.extend((item, destination / "platform-inputs/xmake-pkgs") for item in cpp)
    files = {}
    for item, directory in groups:
        path = directory / item["filename"]
        if args.offline:
            if not path.is_file() or sha256(path) != item["sha256"]:
                raise ValueError(f"Missing verified offline input {path}")
        else:
            path = download(item, directory)
        files[item["filename"]] = path
        print(f"VERIFIED {path.name}", flush=True)
    if args.only not in ("all", "metrics"):
        return
    helper = root / "scripts/mh3_pyinstall.py"
    if sha256(helper) != metrics["installer_sha256"]:
        raise ValueError("Wheel installer SHA256 mismatch")
    package = destination / "ss5_compare_metrics.pyz"
    with zipfile.ZipFile(package, "w", zipfile.ZIP_DEFLATED, compresslevel=3) as archive:
        archive.write(root / "scripts/ss5_compare_metrics.py", "__main__.py")
        archive.write(helper, "mh3_pyinstall.py")
        archive.writestr("provenance.json", json.dumps(metrics, indent=2) + "\n")
        for item in metrics["packages"]:
            archive.write(files[item["filename"]], "wheels/" + item["filename"])
    package.with_suffix(".pyz.sha256").write_text(f"{sha256(package)}  {package.name}\n", encoding="ascii")
    print(f"METRICS_BUNDLE={package}", flush=True)


if __name__ == "__main__":
    main()
