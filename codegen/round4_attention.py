"""Compile H3's selected SDPA through NineToothed and the DTK AOT tool."""

import ctypes as ct
import importlib
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import patch

from round1_worker import DIM, HEADS, SEQUENCE, core_bridge, error_metrics, input_digest, load_core, tensor_info
from round2_aot_worker import NineToothedTensor, dtk_roots, logged_command
from round3_attention import DISTRIBUTIONS, SCHEDULES, inputs_for
from round3_hip_aot import adapt_vendor_cpp, core_stream, exact
from run_round1 import sha256


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
            command = [sys.executable, "-B", str(Path(__file__).with_name("round2_aot_worker.py")),
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


def configure_lds(handle, required, report):
    import triton

    configure = getattr(handle, f"ss5_configure_lds_{ENTRY}")
    configure.argtypes = (ct.c_int, ct.c_uint)
    configure.restype = ct.c_int
    limits = []
    report["lds_validation"] = {"required_bytes": required, "max_shared_mem_by_device": limits}
    for device in range(8):
        limit = triton.runtime.driver.active.utils.get_device_properties(device)["max_shared_mem"]
        if type(limit) is not int or not required <= limit <= 0xFFFF_FFFF:
            raise ValueError(f"Invalid LDS limit on device {device}: {limit!r} for {required} required bytes")
        limits.append(limit)
        status = configure(device, limit)
        if status:
            raise RuntimeError(f"Configuring LDS for device {device} returned {status}")


def view4(tensor):
    return NineToothedTensor(tensor.data_ptr(), (ct.c_uint64 * 4)(*tensor.shape), (ct.c_int64 * 4)(*tensor.stride()))


def validate_attention(torch, settings, original, library, report):
    import ninetoothed as nt
    from infinicore.context import sync_stream
    from pinned_sdpa_v5 import CausalVariant

    core = load_core(torch, settings, report)
    device = torch.device("cuda", settings["device"])
    core.set_device(core.device("cuda", settings["device"]))
    pointer, external = core_stream(torch, device)
    handle = ct.CDLL(str(library))
    required = report["device_binary"]["shared"] + report["elf_resources"]["group_segment_fixed_size"]
    configure_lds(handle, required, report)
    launch = getattr(handle, f"launch_{ENTRY}")
    launch.argtypes = (ct.c_void_p,) + (NineToothedTensor,) * len(NAMES)
    launch.restype = ct.c_int
    failure = getattr(handle, f"ss5_launch_phase_{ENTRY}")
    failure.argtypes = ()
    failure.restype = ct.c_char_p
    _, _, warps, stages = SCHEDULES[0]
    jit = nt.make(*original, kernel_name="ss5_s4_attention_jit", num_warps=warps, num_stages=stages, max_num_configs=1)
    scale = ct.c_double(ct.c_float(1 / math.sqrt(DIM)).value)
    scalar = NineToothedTensor(ct.addressof(scale), None, None)
    mask = torch.empty((1, HEADS, SEQUENCE, SEQUENCE), device="meta")
    checks = []
    report["core_stream"] = pointer
    report["checks"] = checks
    for distribution in DISTRIBUTIONS:
        q, k, v = inputs_for(torch, device, distribution)
        expected, actual = torch.full_like(q, float("nan")), torch.full_like(q, float("nan"))
        qh, kh, vh, eh, ah = (x.permute(0, 2, 1, 3) for x in (q, k, v, expected, actual))
        torch.cuda.synchronize(device)
        with torch.cuda.stream(external):
            jit(qh, kh, vh, mask, False, scale.value, eh, False, CausalVariant.UPPER_LEFT)
            status = launch(pointer, view4(qh), view4(kh), view4(vh), scalar, view4(ah))
            if status:
                raise RuntimeError(f"Attention launch at {failure().decode('ascii')} returned {status}")
            # Exercise a downstream Core op on the same non-default stream.
            consumed = torch.empty_like(actual)
            core.add(core_bridge(core, actual), core_bridge(core, q), out=core_bridge(core, consumed))
        sync_stream()
        checks.append({"distribution": distribution, "input_sha256": input_digest((q, k, v)),
                       "kernel_inputs": [tensor_info(x) for x in (qh, kh, vh)],
                       "error_vs_jit": error_metrics(actual, expected),
                       "consumer_vs_jit": error_metrics(consumed, expected + q)})
    return all(exact(item["error_vs_jit"]) and exact(item["consumer_vs_jit"]) for item in checks)
