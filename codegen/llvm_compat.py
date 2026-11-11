"""Process-local adaptation of LLVM 19 GEP flags for a legacy AMD consumer.

Only `nuw`/`nusw` immediately following a getelementptr opcode are removed.
Quoted strings, comments, all operands/types and arithmetic flags are retained.
The vendor backend and numeric kernel sources on disk remain untouched.
"""

import functools
import hashlib
import inspect
import json
from pathlib import Path
import re
import shutil
import subprocess
import threading
import traceback


_GEP = re.compile(r"(?<![\w%@$.!-])getelementptr(?P<space>\s+)(?P<flags>(?:(?:inbounds|nuw|nusw)\b\s+)+)")
_NEW_FLAGS = re.compile(r"\b(?:nuw|nusw)\b")


def _source_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _segments(text):
    """Yield LLVM code and unchanged quoted/comment segments, including escapes."""
    start, index, size = 0, 0, len(text)
    while index < size:
        if text[index] not in {'"', ';'}:
            index += 1
            continue
        if start < index:
            yield True, text[start:index]
        begin = index
        if text[index] == ';':
            end = text.find('\n', index)
            index = size if end < 0 else end
        else:
            index += 1
            while index < size:
                if text[index] == '\\':
                    index += 2
                elif text[index] == '"':
                    index += 1
                    break
                else:
                    index += 1
        yield False, text[begin:index]
        start = index
    if start < size:
        yield True, text[start:]


def legacy_gep_flags(source):
    if not isinstance(source, str):
        raise TypeError("AMD make_amdgcn input must be LLVM IR text")
    counts = {"nuw": 0, "nusw": 0, "affected_geps": 0}

    def replace(match):
        flags = match.group("flags")
        removed = _NEW_FLAGS.findall(flags)
        if not removed:
            return match.group(0)
        for flag in removed:
            counts[flag] += 1
        counts["affected_geps"] += 1
        # Remove flag words only; retaining whitespace also retains line numbers.
        return "getelementptr" + match.group("space") + _NEW_FLAGS.sub("", flags)

    result = "".join(_GEP.sub(replace, part) if code else part for code, part in _segments(source))
    return result, counts


def _tool_versions():
    result = {}
    for name in ("clang", "llvm-as", "llc"):
        path = shutil.which(name)
        if path is None:
            result[name] = {"available": False}
            continue
        try:
            completed = subprocess.run([path, "--version"], capture_output=True, text=True,
                                       errors="replace", timeout=5)
            result[name] = {"path": path, "exit_code": completed.returncode,
                            "version_output": (completed.stdout + completed.stderr)[:2000]}
        except (OSError, subprocess.TimeoutExpired) as exc:
            result[name] = {"path": path, "error": str(exc)}
    return result


def install_legacy_gep_adapter(destination, case, compiler_module=None):
    if re.fullmatch(r"[a-z][a-z0-9_]*", case) is None:
        raise ValueError("Invalid compiler evidence case name")
    if compiler_module is None:
        from triton.backends.amd import compiler as compiler_module
    backend = compiler_module.HIPBackend
    descriptor = inspect.getattr_static(backend, "make_amdgcn")
    if not isinstance(descriptor, staticmethod):
        raise RuntimeError("Unsupported vendor AMD make_amdgcn descriptor; refusing an unverified patch")
    original = descriptor.__func__
    inspect.signature(original).bind("llvm text", {}, object())
    if getattr(original, "_ss5_legacy_gep_adapter", False):
        raise RuntimeError("LLVM GEP adapter already installed in this worker")
    destination = Path(destination).resolve(strict=True)
    directory = (destination / "compiler-ir" / case).resolve()
    if not directory.is_relative_to(destination):
        raise ValueError("Compiler evidence path escaped this run")
    directory.mkdir(parents=True, exist_ok=True)
    journal = directory / "compiler-compat.json"
    record = {"policy": "remove only LLVM19 GEP nuw/nusw tokens before gfx936 make_amdgcn",
              "vendor_source_modified": False, "numeric_source_modified": False,
              "vendor_compiler_source": str(compiler_module.__file__),
              "vendor_compiler_sha256": _source_hash(compiler_module.__file__),
              "original_stage_source_sha256": hashlib.sha256(inspect.getsource(original).encode()).hexdigest(),
              "adapter_sha256": _source_hash(__file__),
              "tool_versions": _tool_versions(), "calls": []}
    lock = threading.Lock()

    def flush():
        journal.write_text(json.dumps(record, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")

    flush()

    @functools.wraps(original)
    def compatible(source, metadata, options):
        arch = str(getattr(options, "arch", ""))
        if arch.split(":", 1)[0] != "gfx936":
            raise RuntimeError(f"This compatibility experiment is restricted to gfx936, got {arch}")
        converted, counts = legacy_gep_flags(source)
        before = hashlib.sha256(source.encode()).hexdigest()
        after = hashlib.sha256(converted.encode()).hexdigest()
        with lock:
            index = len(record["calls"]) + 1
            raw_path = directory / f"{index:03d}-original.ll"
            compatible_path = directory / f"{index:03d}-legacy.ll"
            raw_path.write_text(source, encoding="utf-8", newline="\n")
            compatible_path.write_text(converted, encoding="utf-8", newline="\n")
            call = {"arch": arch, "original_ir": str(raw_path), "compatible_ir": str(compatible_path),
                    "original_sha256": before, "compatible_sha256": after,
                    "removed_flags": counts, "status": "calling_vendor_make_amdgcn"}
            record["calls"].append(call)
            flush()  # Survives a native crash in the call below.
        print(f"SS5_LLVM_GEP_COMPAT case={case} nuw={counts['nuw']} nusw={counts['nusw']} affected={counts['affected_geps']}", flush=True)
        try:
            result = original(converted, metadata, options)
        except BaseException:
            with lock:
                call["status"] = "python_exception"
                call["error"] = traceback.format_exc()
                flush()
            raise
        with lock:
            call["status"] = "amdgcn_generated"
            flush()
        return result

    compatible._ss5_legacy_gep_adapter = True
    backend.make_amdgcn = staticmethod(compatible)
    return {"installed": True, "journal": str(journal),
            "policy": record["policy"], "vendor_source_modified": False,
            "vendor_compiler_sha256": record["vendor_compiler_sha256"],
            "adapter_sha256": record["adapter_sha256"]}
