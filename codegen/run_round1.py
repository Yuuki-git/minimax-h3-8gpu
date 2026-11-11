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
