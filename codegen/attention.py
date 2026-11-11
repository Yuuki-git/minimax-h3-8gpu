"""The accepted fixed H3 attention definition and AOT packaging."""
import importlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import patch
from common import sha256, dtk_roots, logged_command
from hip import adapt_vendor_cpp
DIM, HEADS, SEQUENCE = 128, 7, 27570
SCHEDULES = ((64, 64, 4, 1),)

ENTRY = "ss5_s4_attention"
NAMES = ("query", "key", "value", "scale", "output")
SHAPE = (1, HEADS, SEQUENCE, DIM)
STRIDE = (SEQUENCE * HEADS * DIM, DIM, HEADS * DIM, 1)
V_STRIDE = (SEQUENCE * HEADS * DIM * 3, DIM, HEADS * DIM * 3, 1)


def definitions():
    import ninetoothed as nt
    import pinned_sdpa_v5 as v5
    from fixed_sdpa_h3 import application_without_kv_cache

    m, n, _, _ = SCHEDULES[0]
    arrange_v5, application_v5, original = v5.premake(
        False, emb_dim=DIM, is_causal=False, with_attn_mask=False,
        causal_variant=v5.CausalVariant.UPPER_LEFT, dtype=nt.bfloat16,
        block_size_m=m, block_size_n=n,
    )
    tensors = tuple(nt.Tensor(0, dtype=nt.float32, name=name) if name == "scale"
                    else nt.Tensor(shape=(None, None, None, DIM), dtype=nt.bfloat16, name=name)
                    for name in NAMES)

    def arrange(query, key, value, scale, output):
        args = (query, key, value, *original[3:9], scale, output, *original[11:])
        arranged = arrange_v5(*args)
        return (*arranged[:3], *arranged[5:7])

    return (arrange, application_without_kv_cache, tensors), (arrange_v5, application_v5, original)


def tensor_guard(name, strides):
    predicates = [f"!{name}.data", f"!{name}.shape", f"!{name}.strides",
                  f"reinterpret_cast<uintptr_t>({name}.data) % 16"]
    predicates.extend(f"{name}.shape[{dim}] != {value}" for dim, value in enumerate(SHAPE))
    predicates.extend(f"{name}.strides[{dim}] != {value}" for dim, value in enumerate(strides))
    return f'    if ({" || ".join(predicates)}) {{ launch_phase = "tensor:{name}"; return hipErrorInvalidValue; }}\n'


def compile_attention(destination, definition, report):
    import ninetoothed as nt
    from triton.backends.amd import driver

    aot = importlib.import_module("ninetoothed.aot")
    patched = json.loads((destination / "aot-patch.json").read_bytes())
    if sha256(aot.__file__) != patched["after_sha256"]:
        raise ValueError("NineToothed imported an unexpected AOT packager")
    output = destination / "aot" / ENTRY
    output.mkdir(parents=True)
    raw = destination / "native" / "aot_attention"
    raw.parent.mkdir(exist_ok=True)
    library = output / f"lib{ENTRY}.so"
    report["compiler_calls"] = []

    def run_tool(command, **kwargs):
        if command[:3] == ["python", "-m", "triton.tools.compile"]:
            command = [sys.executable, "-B", str(Path(__file__).with_name("compiler.py")),
                       "compiler", str(destination), raw.name, *command[3:]]
            report["compiler_calls"].append([str(arg) for arg in command])
            return subprocess.run(command, **kwargs)
        if command[0] != "nvcc":
            raise ValueError("Unexpected pinned AOT build command")
        runtime = Path(driver._get_path_to_hip_runtime_dylib()).resolve(strict=True)
        cxx = shutil.which("c++") or shutil.which("clang++")
        if cxx is None:
            raise RuntimeError("No host C++ compiler")
        roots = [Path(driver.__file__).parent / "include", *(root / "include" for root in dtk_roots())]
        includes = [f"-I{root}" for root in roots if (root / "hip/hip_runtime_api.h").is_file()]
        shutil.copyfile(aot._HEADER_PATH, output / "ninetoothed.h")
        for path in sorted(output.glob(f"{ENTRY}.*")):
            source = path.read_text(encoding="utf-8")
            path.write_text(source.replace(f'#include "{aot._HEADER_PATH}"', '#include "ninetoothed.h"'),
                            encoding="utf-8", newline="\n")
        build = [cxx, "-std=c++17", "-O2", "-shared", "-fPIC", "-pthread", "-D__HIP_PLATFORM_AMD__",
                 "-Werror=return-type", *includes, *sorted(str(p) for p in output.glob("*.cpp")), str(runtime),
                 "-Wl,--no-undefined", f"-Wl,-soname,{library.name}", "-o", str(library)]
        report["hip_runtime"] = {"path": str(runtime), "sha256": sha256(runtime)}
        report["compiler_calls"].append(build)
        if logged_command(build, output, "build.log", 120):
            raise RuntimeError(f"Generated HIP build failed, see {output / 'build.log'}")

    def variants(names, tensors, find):
        ranks = tuple(find(tensors, name).source.ndim for name in names)
        if tuple(names) != NAMES or ranks != (4, 4, 4, 0, 4):
            raise ValueError(f"Unexpected H3 AOT argument contract {tuple(names)}, {ranks}")
        return [("gfx936", (), tuple((name, 3) for name in names if name != "scale"), nt.int32, nt.int32)]

    def pack(source, header, wrapper, original, name, symbol, suffix):
        meta = json.loads((raw / "metadata.json").read_bytes())
        code = (raw / "gate.hsaco").read_bytes()
        if sha256(raw / "gate.hsaco") != meta["hsaco_sha256"]:
            raise ValueError("Code object hash differs")
        normalized, resources = adapt_vendor_cpp(source, symbol, name, meta, code)
        if meta["shared"] <= 0 or (meta["num_warps"], meta["num_stages"]) != SCHEDULES[0][2:]:
            raise ValueError("Unexpected attention LDS or schedule")
        report.update(device_binary=meta, elf_resources=resources)
        if wrapper.count(f"return {original}(") != 1 or wrapper.count(f"launch_{name}(") != 1:
            raise ValueError("Unexpected generated attention launch wrapper")
        if "double" not in wrapper or "scale.data" not in wrapper:
            raise ValueError("Expected the pinned AOT double storage for the FP32 scale input")
        guards = "".join(tensor_guard(n, V_STRIDE if n == "value" else STRIDE) for n in NAMES if n != "scale")
        guards += ('    if (!scale.data || reinterpret_cast<uintptr_t>(scale.data) % alignof(double)) '
                   '{ launch_phase = "scale"; return hipErrorInvalidValue; }\n')
        wrapper = wrapper.replace(f"return {original}(", f"return {symbol}(", 1)
        wrapper = wrapper.replace(f"launch_{name}(", f"launch_{name}_{suffix}(", 1)
        wrapper = wrapper.replace("{\n", "{\n" + guards, 1)
        return {f"{name}.{suffix}.cpp": f'{normalized}\n#include "{header}"\nextern "C" {wrapper}\n'}

    _, _, warps, stages = SCHEDULES[0]
    with (patch.object(aot, "_ss5_hip_variant", pack, create=True),
          patch.object(aot, "_enumerate_variant_specs", variants),
          patch.object(aot, "subprocess", SimpleNamespace(run=run_tool)),
          patch.object(aot, "_load_launch_func", lambda *args: library)):
        nt.make(*definition, caller="cuda", kernel_name=ENTRY, output_dir=output, num_warps=warps, num_stages=stages)
    header = (output / f"{ENTRY}.h").read_text(encoding="utf-8")
    params = ", ".join(["NineToothedStream stream", *(f"NineToothedTensor {n}" for n in NAMES)])
    if f"launch_{ENTRY}({params})" not in header:
        raise ValueError("Generated attention host argument order differs")
    report["launcher_sha256"] = sha256(library)
    return library
