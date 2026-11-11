#!/usr/bin/env python3
"""Extract the two pinned xmake packages without a cross-directory rename."""

import hashlib
import os
from pathlib import Path
import re
import shutil
import stat
import subprocess
import sys
import zipfile


BOOST_SHA256 = "f51707c27359a0df0cac1beada86de31bb5eed5e8285592dadec384df99c2984"
PYBIND_SHA256 = "affea1ada7b39fe1d835559fcb78800c7927d514bd480ed01b71b88205a5e536"
DOC_LINK = "libs/decimal/doc/modules/ROOT/examples"
TAR_PREFIX = r"(?:/usr/bin/)?tar: "
LINK_CHMOD = re.compile(
    TAR_PREFIX + r"(?:boost-1\.92\.0/)?" + re.escape(DOC_LINK)
    + r": Cannot change mode to [rwxstST-]{9}: Bad file descriptor"
)
TAR_SUMMARY = re.compile(TAR_PREFIX + r"Exiting with failure status due to previous errors")


def digest(path):
    sha = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            sha.update(chunk)
    return sha.hexdigest()


def extract_boost(program, args, target):
    result = subprocess.run(
        [program, "--strip-components=1", *args],
        stderr=subprocess.PIPE,
        env=dict(os.environ, LC_ALL="C"),
    )
    if result.returncode:
        lines = [line for line in result.stderr.decode(errors="replace").splitlines() if line.strip()]
        link = target / DOC_LINK
        if (result.returncode != 2 or len(lines) != 2 or not LINK_CHMOD.fullmatch(lines[0])
                or not TAR_SUMMARY.fullmatch(lines[1]) or not link.is_symlink()
                or os.readlink(link) != "../../../example"):
            sys.stderr.buffer.write(result.stderr)
            return result.returncode if result.returncode > 0 else 128 - result.returncode
        print(f"SS5_BOOST_TAR_MODE_FIX: {DOC_LINK}", flush=True)
    if not (target / "CMakeLists.txt").is_file() or not (target / "libs").is_dir():
        raise ValueError("Boost source root missing after extraction")
    print("SS5_XMAKE_FLAT_SOURCE=boost-1.92.0", flush=True)
    return 0


def extract_pybind(archive, target):
    with zipfile.ZipFile(archive) as source:
        for member in source.infolist():
            root, slash, filename = member.filename.partition("/")
            if root != "pybind11-3.1.0" or not slash:
                raise ValueError(f"unexpected ZIP root: {member.filename}")
            if not filename and member.is_dir():
                continue
            parts = filename.removesuffix("/").split("/")
            mode = member.external_attr >> 16
            if any(p in {"", ".", ".."} or "\\" in p or ":" in p for p in parts):
                raise ValueError(f"unexpected ZIP path: {member.filename}")
            if stat.S_IFMT(mode) not in {0, stat.S_IFREG, stat.S_IFDIR}:
                raise ValueError(f"unexpected ZIP file type: {member.filename}")
            output = target.joinpath(*parts)
            if member.is_dir():
                output.mkdir(parents=True, exist_ok=True)
            else:
                output.parent.mkdir(parents=True, exist_ok=True)
                with source.open(member) as src, output.open("xb") as dst:
                    shutil.copyfileobj(src, dst)
                output.chmod((mode & 0o777) or 0o644)
    if not (target / "include/pybind11/pybind11.h").is_file():
        raise ValueError("pybind11 headers missing after extraction")
    print("SS5_XMAKE_FLAT_SOURCE=pybind11-3.1.0", flush=True)
    return 0


def main():
    tool = Path(sys.argv[0]).name
    args = sys.argv[1:]
    program = os.environ[f"SS5_SYSTEM_{tool.upper()}"]
    if (tool == "tar" and len(args) == 4 and args[0] == "-xf" and args[2] == "-C"
            and Path(args[1]).name in {"boost-1.92.0.tar.gz", "boost-1.92.0-cmake.tar.gz"}):
        archive, target, wanted = Path(args[1]), Path(args[3]), BOOST_SHA256
    elif tool == "unzip" and args[:1] == ["-o"] and args[-2:-1] == ["-d"]:
        inputs = args[1:-2]
        if inputs[:1] == ["-q"]:
            inputs = inputs[1:]
        if len(inputs) != 1 or Path(inputs[0]).name not in {"v3.1.0.zip", "pybind11-v3.1.0.zip"}:
            os.execv(program, [program, *args])
        archive, target, wanted = Path(inputs[0]), Path(args[-1]), PYBIND_SHA256
    else:
        os.execv(program, [program, *args])

    cache = Path(os.environ["XMAKE_PKG_CACHEDIR"]).resolve(strict=True)
    target = target.resolve(strict=True)
    if target.name != "source.tmp" or not target.is_relative_to(cache):
        os.execv(program, [program, *args])
    if digest(archive) != wanted:
        raise ValueError(f"unexpected SHA256 for {archive.name}")
    if not target.is_dir() or any(target.iterdir()):
        raise ValueError(f"expected empty xmake extraction directory: {target}")
    if tool == "tar":
        return extract_boost(program, args, target)
    return extract_pybind(archive, target)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (KeyError, OSError, ValueError, zipfile.BadZipFile) as exc:
        print(f"SS5_XMAKE_EXTRACT_FAILED: {exc}", file=sys.stderr)
        raise SystemExit(1)
