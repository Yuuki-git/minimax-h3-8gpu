"""Run the unchanged vendor compiler through the workspace adapters."""
import importlib
import json
from pathlib import Path
import runpy
import shutil
import sys
from unittest.mock import patch
from common import checked_child, sha256, write_json

def binary_evidence(compiled, output):
    metadata = json.loads(json.dumps(compiled.metadata._asdict(), default=str))
    code = compiled.asm["hsaco"]
    if code[:6] != b"\x7fELF\x02\x01" or int.from_bytes(code[18:20], "little") != 224:
        raise ValueError("Expected a little-endian AMDGPU ELF64 code object")
    binary = output / "gate.hsaco"
    binary.write_bytes(code)
    metadata.update(hsaco_sha256=sha256(binary), hsaco_bytes=len(code))
    write_json(output / "metadata.json", metadata)
    return metadata


def compile_child(directory, case, arguments):
    import triton
    from llvm_compat import install_legacy_gep_adapter

    install_legacy_gep_adapter(directory, case)
    output = directory / "native" / case
    output.mkdir()
    stub = checked_child(arguments[arguments.index("--out-path") + 1], directory)
    source = checked_child(arguments[0], directory)
    shutil.copyfile(source, output / "gate.py")
    tool = importlib.util.find_spec("triton.tools.compile")
    write_json(output / "compiler.json", {"argv": arguments, "tool": tool.origin, "tool_sha256": sha256(tool.origin)})
    sys.argv = [tool.origin, *arguments]
    compiler = triton.compile

    def capture_compile(*args, **kwargs):
        compiled = compiler(*args, **kwargs)
        binary_evidence(compiled, output)
        return compiled

    try:
        with patch.object(triton, "compile", capture_compile):
            runpy.run_module("triton.tools.compile", run_name="__main__", alter_sys=True)
    finally:
        for path in sorted(stub.parent.glob(stub.name + ".*")):
            if path.suffix in (".c", ".cpp", ".h"):
                shutil.copyfile(path, output / path.name)

if __name__ == "__main__":
    mode, directory, case, *arguments = sys.argv[1:]
    if mode != "compiler":
        raise ValueError(mode)
    compile_child(Path(directory).resolve(strict=True), case, arguments)
