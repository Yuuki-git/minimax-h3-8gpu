#!/usr/bin/env python3
"""Save the built FL2VA runtime to personal AFS and restore it in a new session."""

import argparse
import hashlib
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import tempfile
import uuid
import zipfile


TREES = ("infini/lib", "src/InfiniCore/python", "src/InfiniLM/python")
ENTRIES = (
    "src/InfiniLM/examples/minimax_h3.py",
    "src/InfiniLM/examples/minimax_h3_compare.py",
)
LIBRARIES = ("infinirt", "infiniop", "infiniccl", "infinicore_cpp_api")
BUNDLE = "fl2va-runtime.zip"
CHECKSUM = "runtime.sha256"


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check_runtime(root):
    required = (*ENTRIES, *(f"infini/lib/lib{name}.so" for name in LIBRARIES))
    for item in required:
        path = root / item
        if not path.is_file() or not path.stat().st_size:
            raise ValueError(f"缺少编译或运行产物 {item}")
    for repo, package in (("InfiniCore", "infinicore"), ("InfiniLM", "infinilm")):
        tree = root / f"src/{repo}/python/{package}"
        if not (tree / "__init__.py").is_file():
            raise ValueError(f"缺少 {package} 的 Python 包")
        if not any(path.is_file() and path.stat().st_size for path in tree.rglob(f"_{package}*.so")):
            raise ValueError(f"缺少 {package} 的 Python 原生扩展")


def runtime_paths(root):
    for tree in TREES:
        for path in sorted((root / tree).rglob("*")):
            if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc":
                yield path
    yield from (root / entry for entry in ENTRIES)


def check_group_access(path, gid, required):
    info = path.stat()
    mode = stat.S_IMODE(info.st_mode)
    if info.st_gid != gid or mode & required != required or mode & 0o007:
        raise ValueError(f"{path.name} 未继承个人 AFS 的组或权限，gid={info.st_gid} mode={mode:o}")


def save(workspace, afs, log=None):
    import fcntl

    if Path.cwd().resolve() != afs:
        raise ValueError("请先进入个人 AFS，再执行 save")
    afs_gid = afs.stat().st_gid
    if afs_gid not in {os.getegid(), *os.getgroups()}:
        raise ValueError(f"当前会话未加入个人 AFS 的组 gid={afs_gid}")
    root = workspace / "ss5-v3-work"
    if root.resolve(strict=True) != root:
        raise ValueError("编译目录不能是符号链接")
    if log is not None:
        log = log.resolve(strict=True)
        if not log.is_relative_to(workspace) or not log.is_file():
            raise ValueError("附加日志须是当前会话工作区里的文件")
    with (root / "run.lock").open("rb") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_SH | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError("请等当前构建或推理命令返回再保存") from exc
        check_runtime(root)
        # Session UIDs can differ across nodes. Inherit the AFS group at creation.
        previous_umask = os.umask(0o007)
        try:
            dest = afs / f"ss5-fl2va-runtime-{uuid.uuid4().hex[:12]}"
            dest.mkdir(mode=0o770)
            print(f"RUNTIME_SAVE_DIR={dest}", flush=True)
            check_group_access(dest, afs_gid, 0o050)
            bundle = dest / BUNDLE
            with zipfile.ZipFile(bundle, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=3) as archive:
                for path in runtime_paths(root):
                    target = path.resolve(strict=True)
                    if not target.is_relative_to(root):
                        raise ValueError(f"运行文件引用了会话外路径 {path.relative_to(root)}")
                    archive.write(target, path.relative_to(root).as_posix())
            check_group_access(bundle, afs_gid, 0o040)
            if log is not None:
                shutil.copyfile(log, dest / "run.log")
                check_group_access(dest / "run.log", afs_gid, 0o040)
            (dest / CHECKSUM).write_text(f"{sha256(bundle)}  {BUNDLE}\n", encoding="ascii")
            check_group_access(dest / CHECKSUM, afs_gid, 0o040)
        finally:
            os.umask(previous_umask)
    print(f"SS5_RUNTIME_GROUP_OK={afs_gid}", flush=True)
    print(f"SS5_RUNTIME_SAVED={dest}", flush=True)


def restore(workspace, bundle_dir):
    if Path.cwd().resolve() != workspace:
        raise ValueError("请先进入 IFAB_WORKSPACE_ROOT，再执行 restore")
    root = workspace / "ss5-v3-work"
    if os.path.lexists(root):
        raise ValueError(f"{root} 已存在，请在空白会话恢复")
    bundle = bundle_dir / BUNDLE
    checksum = (bundle_dir / CHECKSUM).read_text(encoding="ascii").strip().split()
    if (len(checksum) != 2 or checksum[1] != BUNDLE or not re.fullmatch(r"[0-9a-f]{64}", checksum[0])
            or sha256(bundle) != checksum[0]):
        raise ValueError("运行包 SHA256 不匹配")
    with tempfile.TemporaryDirectory(prefix="ss5-restore-", dir=workspace) as temp:
        stage = Path(temp)
        if stage.resolve().parent != workspace:
            raise ValueError("恢复目录不在当前工作区")
        with zipfile.ZipFile(bundle) as archive:
            names = archive.namelist()
            if len(names) != len(set(names)):
                raise ValueError("运行包中存在重名文件")
            for name in names:
                path = PurePosixPath(name)
                if (path.is_absolute() or path.as_posix() != name or ".." in path.parts or "\\" in name or ":" in name
                        or not (name in ENTRIES or any(name.startswith(tree + "/") for tree in TREES))):
                    raise ValueError(f"运行包中存在意外路径 {name}")
            archive.extractall(stage)
        check_runtime(stage)
        (stage / "tmp").mkdir()
        (stage / "cache").mkdir()
        (stage / "run.lock").touch()
        if os.path.lexists(root):
            raise ValueError(f"{root} 已存在")
        stage.rename(root)
    print(f"SS5_RUNTIME_RESTORED={root}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    save_parser = sub.add_parser("save")
    save_parser.add_argument("--log", type=Path, help="保留本次会话的运行日志")
    restore_parser = sub.add_parser("restore")
    restore_parser.add_argument("bundle_dir", type=Path)
    args = parser.parse_args()
    try:
        workspace = Path(os.environ["IFAB_WORKSPACE_ROOT"]).resolve(strict=True)
        afs = Path(os.environ["IFAB_AFS_HOME"]).resolve(strict=True)
        if workspace == workspace.parent or afs == afs.parent or workspace.is_relative_to(afs):
            raise ValueError("临时工作区与个人 AFS 路径不符")
        if args.command == "save":
            save(workspace, afs, args.log)
        else:
            restore(workspace, args.bundle_dir.resolve(strict=True))
    except (KeyError, OSError, ValueError, zipfile.BadZipFile) as error:
        parser.exit(1, f"SS5_RUNTIME_FAILED: {error}\n")


if __name__ == "__main__":
    main()
