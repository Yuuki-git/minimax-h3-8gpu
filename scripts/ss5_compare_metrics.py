#!/usr/bin/env python3
"""Install the bundled comparison dependencies in the team's existing venv."""

import argparse
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import sysconfig
import zipfile

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name


CHECK = r'''
import json
import math

import imageio
import lazy_loader
import networkx
import numpy as np
import packaging
import PIL.Image
import scipy
import skimage
import tifffile
from skimage.metrics import structural_similarity

if skimage.__version__ != "0.25.2":
    raise RuntimeError(f"实际 scikit-image 版本不符 {skimage.__version__}")
zero = np.zeros((16, 16), dtype=np.uint8)
white = np.full_like(zero, 255)
identical = structural_similarity(zero, zero, data_range=255)
different = structural_similarity(zero, white, data_range=255)
if identical != 1.0 or not math.isclose(different, 0.0001 / 1.0001, rel_tol=1e-12):
    raise RuntimeError(f"SSIM 验证失败 {identical=} {different=}")
modules = (np, scipy, skimage, lazy_loader, networkx, tifffile, imageio, PIL, packaging)
print(json.dumps({module.__name__: {
    "version": getattr(module, "__version__", None), "path": module.__file__
} for module in modules}, indent=2))
print(f"SS5_COMPARE_SSIM_OK identical={identical} different={different}")
'''


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def location():
    if sys.version_info[:2] != (3, 11) or sys.platform != "linux" or platform.machine() != "x86_64":
        raise ValueError("请使用平台既有的 Linux x86-64 Python 3.11 venv")
    afs = Path(os.environ["IFAB_AFS_HOME"]).resolve(strict=True)
    if Path.cwd().resolve() != afs:
        raise ValueError("请先进入个人 AFS，再安装比较依赖")
    venv = (afs / "venvs/minimax-h3").resolve(strict=True)
    if Path(sys.prefix).resolve() != venv or sys.prefix == sys.base_prefix:
        raise ValueError(f"请使用 {venv}/bin/python")
    site = Path(sysconfig.get_path("purelib")).resolve(strict=True)
    if not site.is_relative_to(venv):
        raise ValueError("安装位置不在团队 venv")
    return venv, site


def missing_packages(packages):
    bundled = {canonicalize_name(item["name"]): item for item in packages}
    requirements = [Requirement("scikit-image==0.25.2")]
    for item in packages:
        for text in item["requires_dist"]:
            requirement = Requirement(text)
            if requirement.marker is None or requirement.marker.evaluate({"extra": ""}):
                requirements.append(requirement)

    missing = set()
    for requirement in requirements:
        name = canonicalize_name(requirement.name)
        try:
            version = metadata.version(name)
        except metadata.PackageNotFoundError:
            if name not in bundled:
                raise ValueError(f"既有基础依赖缺失 {requirement}") from None
            version = bundled[name]["version"]
            missing.add(name)
        if version not in requirement.specifier:
            raise ValueError(f"既有依赖不满足要求 {requirement}，实际版本 {version}")
    return [item for item in packages if canonicalize_name(item["name"]) in missing]


def check_imports(overlay, use_overlay):
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTHONNOUSERSITE="1")
    if use_overlay:
        env["PYTHONPATH"] = os.pathsep.join(filter(None, (str(overlay), env.get("PYTHONPATH"))))
    return subprocess.check_output(
        [sys.executable, "-B", "-c", CHECK], env=env, text=True, stderr=subprocess.STDOUT
    )


def install(bundle, venv, site):
    import fcntl

    overlay_root = venv / "ss5-overlays"
    if overlay_root.resolve() != overlay_root:
        raise ValueError("覆盖目录包含符号链接")
    overlay_root.mkdir(exist_ok=True)
    with (overlay_root / ".compare-metrics.lock").open("a+b") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError("另一个比较依赖安装进程正在运行") from exc

        bundle_hash = sha256(bundle)
        overlay = overlay_root / f"compare-metrics-{bundle_hash[:12]}"
        if overlay.resolve() != overlay:
            raise ValueError("依赖目录包含符号链接")
        with zipfile.ZipFile(bundle) as archive:
            manifest = json.loads(archive.read("provenance.json"))
            helper_data = archive.read("mh3_pyinstall.py")
            if hashlib.sha256(helper_data).hexdigest() != manifest["installer_sha256"]:
                raise ValueError("包安装器校验失败")
            for item in manifest["packages"]:
                name = item["filename"]
                if Path(name).name != name or not name.endswith(".whl"):
                    raise ValueError(f"依赖清单中的文件名不符 {name}")
                data = archive.read(f"wheels/{name}")
                if hashlib.sha256(data).hexdigest() != item["sha256"]:
                    raise ValueError(f"wheel 校验失败 {name}")

            plan = missing_packages(manifest["packages"])
            overlay.mkdir(exist_ok=True)
            installer = {"__name__": "ss5_bundled_pyinstall"}
            exec(compile(helper_data, "mh3_pyinstall.py", "exec"), installer)
            installer["PATHS"] = {key: str(overlay / key) for key in installer["PATHS"]}
            installer["PATHS"].update(purelib=str(overlay), platlib=str(overlay))
            installer["SITE"] = str(overlay)

            for item in plan:
                wheel = overlay / item["filename"]
                if wheel.is_symlink():
                    raise ValueError(f"安装输入包含符号链接 {wheel}")
                wheel.write_bytes(archive.read(f"wheels/{item['filename']}"))
                try:
                    installer["install_wheel"](str(wheel))
                finally:
                    wheel.unlink()

        (overlay / "provenance.json").write_text(
            json.dumps(dict(manifest, bundle_sha256=bundle_hash, installed=plan), indent=2) + "\n",
            encoding="utf-8",
        )
        print(check_imports(overlay, use_overlay=True), end="", flush=True)
        pth = site / "zz_songshan5_compare_metrics.pth"
        if pth.is_symlink():
            raise ValueError("依赖路径文件包含符号链接")
        previous = pth.read_bytes() if pth.exists() else None
        pth.write_text(str(overlay) + "\n", encoding="utf-8")
        try:
            result = check_imports(overlay, use_overlay=False)
        except Exception:
            if previous is None:
                pth.unlink()
            else:
                pth.write_bytes(previous)
            raise
        (overlay / "validation.log").write_text(result, encoding="utf-8")
        print(f"SS5_COMPARE_METRICS_SITE={overlay}")
        print("SS5_COMPARE_METRICS_READY")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=Path(sys.argv[0]))
    args = parser.parse_args()
    previous_umask = os.umask(0o007)
    try:
        venv, site = location()
        install(args.bundle.resolve(strict=True), venv, site)
    except subprocess.CalledProcessError as exc:
        print(exc.output, end="", file=sys.stderr)
        raise SystemExit("SS5_COMPARE_METRICS_FAILED 导入或 SSIM 验证失败") from exc
    except (KeyError, OSError, ValueError, zipfile.BadZipFile) as exc:
        raise SystemExit(f"SS5_COMPARE_METRICS_FAILED {exc}") from exc
    finally:
        os.umask(previous_umask)


if __name__ == "__main__":
    main()
