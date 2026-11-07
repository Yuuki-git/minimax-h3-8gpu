#!/usr/bin/env python3
"""Activate a verified Diffusers release in the existing InfiniFabric venv."""

import argparse
import hashlib
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
import subprocess
import sys
import sysconfig
import zipfile


WHEEL_NAME = "diffusers-0.40.0-py3-none-any.whl"
WHEEL_SHA256 = "5b5da7c3ddb62152fa4afc577f02e050af688c797375c34fb2d01006da3f3541"
PTH_NAME = "zz_songshan5_diffusers_0400.pth"

CHECK = r'''
from importlib import metadata, util
from pathlib import Path
import sys

import diffusers, huggingface_hub, httpx, torch, transformers

overlay, venv = map(Path, sys.argv[1:3])
actual = Path(diffusers.__file__).resolve().parent
if actual != overlay / "diffusers" or diffusers.__version__ != "0.40.0":
    raise RuntimeError(f"实际导入的不是目标 Diffusers：{diffusers.__version__} {actual}")
if metadata.version("diffusers") != "0.40.0":
    raise RuntimeError("Diffusers 的发行元数据未切换到 0.40.0")
if huggingface_hub.__version__ != "1.28.0" or transformers.__version__ != "5.14.1":
    raise RuntimeError("Hub 或 Transformers 与本轮验证的版本不同")
torch_path = Path(torch.__file__).resolve()
if torch.__version__.split("+")[0] != "2.9.0" or torch.version.hip != "6.3.26093" or torch_path.is_relative_to(venv):
    raise RuntimeError(f"实际 Torch 不是原镜像中的海光版本：{torch.__version__}, {torch.version.hip}, {torch_path}")

import diffusers.modular_pipelines.minimax_h3

for mod in (diffusers, huggingface_hub, httpx, transformers, torch):
    print(f"{mod.__name__}: {mod.__version__} {mod.__file__}")
print(f"HIP: {torch.version.hip}")
print(f"torchvision: {'可查到，尚未执行 NMS 检查' if util.find_spec('torchvision') else '待 02b 编译'}")
'''


def location():
    afs = os.environ.get("IFAB_AFS_HOME")
    if not afs or sys.version_info[:2] != (3, 11):
        raise RuntimeError("请在 InfiniFabric 登录工作区使用既有的 Python 3.11 venv")
    venv = (Path(afs) / "venvs/minimax-h3").resolve(strict=True)
    if Path(sys.prefix).resolve() != venv or sys.prefix == sys.base_prefix:
        raise RuntimeError(f"请使用 {venv}/bin/python")
    site = Path(sysconfig.get_path("purelib")).resolve(strict=True)
    if not site.is_relative_to(venv):
        raise RuntimeError("写入目标不是当前 venv")
    overlay = venv / "ss5-overlays" / f"diffusers-0.40.0-{WHEEL_SHA256[:12]}"
    if overlay.resolve() != overlay:
        raise RuntimeError("拒绝通过符号链接写入覆盖目录")
    return venv, overlay, site / PTH_NAME


def unpack(wheel, overlay):
    if wheel.name != WHEEL_NAME:
        raise RuntimeError(f"请使用随附官方 wheel：{WHEEL_NAME}")
    with wheel.open("rb") as source:
        digest = hashlib.sha256()
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
        if digest.hexdigest() != WHEEL_SHA256:
            raise RuntimeError("wheel SHA256 不一致，未写入 venv")
        source.seek(0)
        with zipfile.ZipFile(source) as archive:
            for entry in archive.infolist():
                rel = PurePosixPath(entry.filename)
                if (rel.is_absolute() or ".." in rel.parts or "\\" in entry.filename or not rel.parts
                        or rel.parts[0] not in {"diffusers", "diffusers-0.40.0.dist-info"}
                        or stat.S_ISLNK(entry.external_attr >> 16)):
                    raise RuntimeError(f"wheel 中有意外路径：{entry.filename}")
                target = overlay.joinpath(*rel.parts)
                if target.resolve() != target:
                    raise RuntimeError(f"拒绝写入符号链接：{target}")
                if entry.is_dir():
                    target.mkdir(parents=True, exist_ok=True)
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(entry) as src, target.open("wb") as dst:
                    shutil.copyfileobj(src, dst)


def verify(venv, overlay, *, try_overlay=False):
    env = dict(os.environ, PYTHONNOUSERSITE="1", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
               DIFFUSERS_SLOW_IMPORT="0")
    env.pop("PYTHONPATH", None)
    if try_overlay:
        env["PYTHONPATH"] = str(overlay)
    # The login workspace has no DCU. Do not import FlashAttention or touch the device.
    subprocess.run([sys.executable, "-B", "-u", "-c", CHECK, str(overlay), str(venv)],
                   env=env, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--wheel", type=Path, help="activate the bundled official wheel")
    mode.add_argument("--verify", action="store_true", help="check the active environment without writing")
    mode.add_argument("--rollback", action="store_true", help="remove this script's path entry only")
    args = parser.parse_args()
    try:
        venv, overlay, link = location()
        contents = f"import sys; sys.path.insert(0, {str(overlay)!r})\n"
        if link.is_symlink() or (link.exists() and link.read_text(encoding="utf-8") != contents):
            raise RuntimeError(f"未识别的路径文件，未修改：{link}")
        if args.rollback:
            link.unlink(missing_ok=True)
            print("SS5_ROLLBACK_OK：仅撤下本工具创建的 .pth")
            return 0
        if args.wheel and not link.exists():
            print(f"验证官方 wheel 并解包到独立目录：{overlay}", flush=True)
            unpack(args.wheel, overlay)
            verify(venv, overlay, try_overlay=True)
            try:
                stream = link.open("x", encoding="utf-8")
            except FileExistsError:
                raise RuntimeError("路径文件被另一个运行创建；请先检查现有状态") from None
            try:
                with stream:
                    stream.write(contents)
                verify(venv, overlay)
            except BaseException:
                link.unlink(missing_ok=True)
                raise
        else:
            verify(venv, overlay)
    except (OSError, RuntimeError, zipfile.BadZipFile, subprocess.CalledProcessError) as exc:
        print(f"SS5_STAGE02_FAILED: {exc}", file=sys.stderr, flush=True)
        return 1
    print("SS5_STAGE02_OK", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
