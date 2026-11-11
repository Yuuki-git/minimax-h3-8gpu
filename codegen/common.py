"""Shared filesystem helpers retained from the accepted generator."""
import hashlib
import json
from pathlib import Path


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for data in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(data)
    return digest.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    allow_nan=False) + "\n", encoding="utf-8")


def checked_child(path, allowed_root):
    resolved = Path(path).resolve()
    if resolved == allowed_root or not resolved.is_relative_to(allowed_root):
        raise ValueError(f"Write target is outside the terminal directory/parent: {resolved}")
    return resolved

import os
import subprocess

def logged_command(command, directory, filename, timeout):
    with (directory / filename).open("wb") as log:
        return subprocess.run(command, cwd=directory, stdout=log, stderr=subprocess.STDOUT,
                              timeout=timeout, check=False).returncode


def dtk_roots():
    roots = [Path(path) for path in ("/opt/dtk/cuda/cuda", "/opt/dtk", "/opt/hyhal", "/opt/rocm", "/usr/local/cuda")]
    roots[:0] = [Path(os.environ[key]) for key in ("HYGON_CUDA_HOME", "DTK_ROOT", "CUDA_HOME")
                 if os.environ.get(key) and Path(os.environ[key]).is_absolute()]
    return list(dict.fromkeys(roots))
