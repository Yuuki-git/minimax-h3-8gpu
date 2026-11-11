"""Adapt the pinned NineToothed AOT packager to DTK's generated HIP C++."""

import ctypes as ct
from concurrent.futures import ThreadPoolExecutor
from functools import partial
import importlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import threading
import traceback
from types import SimpleNamespace
from unittest.mock import patch

from round1_worker import HIDDEN, SEQUENCE, core_bridge, error_metrics, load_core, tensor_info
from round2_aot_worker import (NineToothedTensor, dtk_roots, logged_command, stream_pointer,
                               validate_core_stream)
from round3_metadata import kernel_resources
from run_round1 import checked_child, sha256, write_json


WHEEL_AOT_SHA256 = "ba6fca1cd9f38867555c152a44877477650bd9c740c009e79d0856e2d7e18536"


def patch_wheel_aot(destination):
    paths = list((destination / "venv/lib").glob("python*/site-packages/ninetoothed/aot.py"))
    if len(paths) != 1 or sha256(paths[0]) != WHEEL_AOT_SHA256:
        raise ValueError("Unexpected aot.py in isolated NineToothed wheel")
    path = checked_child(paths[0], destination)
    original = path.read_text(encoding="utf-8")
    patched = original
    changes = (
        ('    c_source_file = output_contents[c_source_file_name]\n',
         '    hip_source = output_contents.get(f"{kernel_name}.{signature_hash}.cpp")\n'
         '    c_source_file = output_contents[c_source_file_name] if hip_source is None else hip_source\n'),
        ('    launch_func_unparsed = unparser.unparse(launch_func)\n',
         '    launch_func_unparsed = unparser.unparse(launch_func)\n'
         '    if hip_source is not None:\n'
         '        return _ss5_hip_variant(hip_source, _HEADER_PATH, launch_func_unparsed,\n'
         '                                func.__name__, kernel_name, kernel_name_with_hash, variant_suffix)\n'),
        ('    body_lines = prelude_lines + branches\n',
         '    body_lines = prelude_lines + branches\n'
         '    if len(variant_specs) == 1 and variant_specs[0][0] == "gfx936":\n'
         '        body_lines = [f"    return launch_{kernel_name}_gfx936({call_args});"]\n'),
    )
    for old, new in changes:
        if patched.count(old) != 1:
            raise ValueError("Pinned aot.py patch anchor changed")
        patched = patched.replace(old, new, 1)
    compile(patched, str(path), "exec")
    path.write_text(patched, encoding="utf-8", newline="\n")
    evidence = destination / "aot-patch"
    evidence.mkdir()
    (evidence / "aot.original.py").write_text(original, encoding="utf-8", newline="\n")
    shutil.copyfile(path, evidence / "aot.patched.py")
    write_json(destination / "aot-patch.json", {"before_sha256": WHEEL_AOT_SHA256,
                                               "after_sha256": sha256(path), "patched_candidate_file": str(path)})


def adapt_vendor_cpp(source, symbol, entry, metadata, code):
    blob = re.search(r"^unsigned char HSACO_NAME\[(\d+)\] = \{([^}]+)\};$", source, re.MULTILINE)
    signature = re.search(rf"^hipError_t {re.escape(symbol)}\(([^\n]+)\) \{{\n(.*?)^\}}$",
                          source, re.MULTILINE | re.DOTALL)
    if not blob or not signature:
        raise ValueError("Vendor HIP C++ template changed")
    embedded = bytes(int(value, 16) for value in re.findall(r"\b0x([a-fA-F0-9]{2})\b", blob[2]))
    if len(embedded) != int(blob[1]) or embedded != code:
        raise ValueError("Vendor wrapper embeds a different code object")
    body = signature[2]
    grid = re.search(r"(^    unsigned int gX = .*?^    void \*args\[\d+\] = \{[^\n]+\};$)", body, re.MULTILINE | re.DOTALL)
    if not grid or body.count("hipModuleLaunchKernel(") != 1:
        raise ValueError("Vendor grid/argument template changed")
    if (metadata["arch"] != "gfx936" or metadata["warp_size"] != 64 or metadata["num_ctas"] != 1 or
            metadata["cluster_dims"] != [1, 1, 1] or
            metadata.get("launch_cooperative_grid") or metadata.get("profile_scratch_size") or
            metadata.get("global_scratch_size") or metadata["name"] != entry):
        raise ValueError("Unexpected AMD launch metadata")
    resources = kernel_resources(code, entry)
    shared = metadata["shared"]
    fixed = resources["group_segment_fixed_size"]
    if type(shared) is not int or shared < 0 or type(fixed) is not int or fixed < 0:
        raise ValueError("Invalid static or dynamic LDS byte count")
    if resources["wavefront_size"] != metadata["warp_size"] or metadata["num_warps"] not in (4, 8):
        raise ValueError("Unexpected wave size or workgroup size")
    threads = metadata["num_warps"] * metadata["warp_size"]
    state = Path(__file__).with_name("hip_module_state.inc").read_text(encoding="utf-8")
    return f"""{source.splitlines()[0]}
{source.splitlines()[1]}
#include <hip/hip_runtime_api.h>
#include <cstdint>
#include <mutex>

namespace {{
static constexpr unsigned char image[{len(code)}] = {{{blob[2]}}};
static constexpr char entry[] = "{entry}";
static constexpr unsigned required_lds = {shared + fixed};
{state}
static hipError_t {symbol}({signature[1]}) {{
    hipFunction_t function;
    auto status = function_for_device(&function);
    if (status != hipSuccess) {{
        return status;
    }}
{grid[1]}
    if (gX == 0 || gY == 0 || gZ == 0) {{
        launch_phase = "grid";
        return hipErrorInvalidValue;
    }}
    launch_phase = "hipModuleLaunchKernel";
    return hipModuleLaunchKernel(function, gX, gY, gZ, {threads}, 1, 1, {shared}, stream, args, nullptr);
}}
}}
extern "C" const char *ss5_launch_phase_{entry}() {{ return launch_phase; }}
extern "C" int ss5_configure_lds_{entry}(int device, unsigned limit) {{ return configure_lds_limit(device, limit); }}
""", resources


def pack_variant(source, header, wrapper, original, name, symbol, suffix, *, raw, tensors, width, rows, report):
    metadata = json.loads((raw / "metadata.json").read_text(encoding="utf-8"))
    code = (raw / "gate.hsaco").read_bytes()
    if sha256(raw / "gate.hsaco") != metadata["hsaco_sha256"]:
        raise ValueError("AOT code object digest differs")
    normalized, resources = adapt_vendor_cpp(source, symbol, name, metadata, code)
    report["device_binary"] = metadata
    report["elf_resources"] = resources
    if wrapper.count(f"return {original}(") != 1 or wrapper.count(f"launch_{name}(") != 1:
        raise ValueError("Unexpected NineToothed launch wrapper")
    wrapper = wrapper.replace(f"return {original}(", f"return {symbol}(", 1)
    wrapper = wrapper.replace(f"launch_{name}(", f"launch_{name}_{suffix}(", 1)
    # The AOT signature specializes stride[-1] to 1 and applies 16-byte/divisible hints.
    guards = []
    for tensor in tensors:
        bounds = [f"!{tensor}.data", f"!{tensor}.shape", f"!{tensor}.strides",
                  f"reinterpret_cast<uintptr_t>({tensor}.data) % 16", f"{tensor}.shape[1] != {width}",
                  f"{tensor}.shape[0] == 0", f"{tensor}.shape[0] != {tensors[0]}.shape[0]",
                  f"{tensor}.strides[0] < {width}", f"{tensor}.strides[0] % 16", f"{tensor}.strides[1] != 1",
                  f"{tensor}.strides[0] > INT32_MAX", f"{tensor}.shape[0] > (INT32_MAX - {width}) / {tensor}.strides[0]"]
        if rows is not None:
            bounds.append(f"{tensor}.shape[0] != {rows}")
        guard = " ||\n        ".join(bounds)
        guards.append(f'    if ({guard}) {{\n        launch_phase = "tensor:{tensor}";\n'
                      '        return hipErrorInvalidValue;\n    }\n')
    wrapper = wrapper.replace("{\n", "{\n" + "".join(guards), 1)
    return {f"{name}.{suffix}.cpp": f'{normalized}\n#include "{header}"\nextern "C" {wrapper}\n'}


def build_aot(destination, case, definition, tensor_names, width, rows, report):
    import ninetoothed as nt
    import triton
    from triton.backends.amd import driver

    aot = importlib.import_module("ninetoothed.aot")
    if sha256(aot.__file__) != json.loads((destination / "aot-patch.json").read_text(encoding="utf-8"))["after_sha256"]:
        raise ValueError("NineToothed imported a different AOT packager")
    name = f"ss5_s3_{case.removeprefix('aot_')}"
    output = destination / "aot" / case
    output.mkdir(parents=True)
    (destination / "native").mkdir(exist_ok=True)
    variant = f"{case}_variant"
    raw = destination / "native" / variant
    report["compiler_calls"] = []

    def run_tool(command, **kwargs):
        if command[:3] == ["python", "-m", "triton.tools.compile"]:
            command = [sys.executable, "-B", str(Path(__file__).with_name("round2_aot_worker.py")),
                       "compiler", str(destination), variant, *command[3:]]
            report["compiler_calls"].append([str(part) for part in command])
            return subprocess.run(command, **kwargs)
        if command[0] != "nvcc":
            raise ValueError("Unexpected command in pinned NineToothed AOT")
        runtime = Path(driver._get_path_to_hip_runtime_dylib()).resolve(strict=True)
        cxx = shutil.which("c++") or shutil.which("clang++")
        if cxx is None:
            raise RuntimeError("No host C++ compiler")
        roots = [Path(driver.__file__).parent / "include", *(root / "include" for root in dtk_roots())]
        includes = [f"-I{root}" for root in roots if (root / "hip/hip_runtime_api.h").is_file()]
        shutil.copyfile(aot._HEADER_PATH, output / "ninetoothed.h")
        for path in sorted(output.glob(f"{name}.*")):
            source = path.read_text(encoding="utf-8")
            path.write_text(source.replace(f'#include "{aot._HEADER_PATH}"', '#include "ninetoothed.h"'),
                            encoding="utf-8", newline="\n")
        build = [cxx, "-std=c++17", "-O2", "-shared", "-fPIC", "-pthread", "-D__HIP_PLATFORM_AMD__",
                 "-Werror=return-type", *includes, *sorted(str(p) for p in output.glob("*.cpp")), str(runtime),
                 "-Wl,--no-undefined", "-o", str(output / f"{name}.so")]
        report["hip_runtime"] = {"path": str(runtime), "sha256": sha256(runtime)}
        report["compiler_calls"].append(build)
        if logged_command(build, output, "build.log", 120):
            raise RuntimeError(f"Generated HIP C++ build failed, see {output / 'build.log'}")

    def fixed_variant(names, tensors, find):
        if tuple(names) != tensor_names or any(find(tensors, name).source.ndim != 2 for name in names):
            raise ValueError("Unexpected tensor ABI for the S3 fixed probe variant")
        return [("gfx936", (), tuple((name, 1) for name in names), nt.int32, nt.int32)]

    build_variant = partial(pack_variant, raw=raw, tensors=tensor_names, width=width, rows=rows, report=report)
    with (patch.object(aot, "_ss5_hip_variant", build_variant, create=True),
          patch.object(aot, "_enumerate_variant_specs", fixed_variant),
          patch.object(aot, "subprocess", SimpleNamespace(run=run_tool)),
          patch.object(aot, "_load_launch_func", lambda *args: output / f"{name}.so")):
        library = nt.make(*definition, caller="cuda", kernel_name=name, output_dir=output, num_warps=4, num_stages=1)
    header = (output / f"{name}.h").read_text(encoding="utf-8")
    expected = ", ".join(["NineToothedStream stream", *(f"NineToothedTensor {n}" for n in tensor_names)])
    if f"launch_{name}({expected})" not in header:
        raise ValueError("Generated host tensor argument order differs")
    report["launcher_sha256"] = sha256(library)
    handle = ct.CDLL(str(library))
    launch = getattr(handle, f"launch_{name}")
    launch.argtypes = (ct.c_void_p,) + (NineToothedTensor,) * len(tensor_names)
    launch.restype = ct.c_int
    failure_phase = getattr(handle, f"ss5_launch_phase_{name}")
    failure_phase.argtypes = ()
    failure_phase.restype = ct.c_char_p
    required_lds = report["device_binary"]["shared"] + report["elf_resources"]["group_segment_fixed_size"]
    if required_lds:
        report["phase"] = "configure_device_lds_limits"
        get_properties = triton.runtime.driver.active.utils.get_device_properties
        configure_lds = getattr(handle, f"ss5_configure_lds_{name}")
        configure_lds.argtypes = (ct.c_int, ct.c_uint)
        configure_lds.restype = ct.c_int
        limits = []
        report["lds_validation"] = {"required_bytes": required_lds, "max_shared_mem_by_device": limits,
                                    "source": "triton.runtime.driver.active.utils.get_device_properties"}
        for device in range(8):
            limit = get_properties(device)["max_shared_mem"]
            if type(limit) is not int or not 0 < limit <= 0xFFFF_FFFF:
                raise ValueError(f"Invalid max_shared_mem for device {device}: {limit!r}")
            limits.append(limit)
            status = configure_lds(device, limit)
            if status:
                raise RuntimeError(f"Configuring LDS limit for device {device} returned {status}")
    failures = []
    report["launch_failures"] = failures

    def call(stream, *tensors):
        status = launch(stream, *(NineToothedTensor.from_torch(t) for t in tensors))
        if status:
            phase = failure_phase().decode("ascii")
            failures.append({"phase": phase, "status": status, "stream": stream,
                             "tensors": {name: {"shape": list(t.shape), "stride": list(t.stride()),
                                                "device": str(getattr(t, "device", None)),
                                                "address_mod_16": t.data_ptr() % 16}
                                         for name, t in zip(tensor_names, tensors, strict=True)}})
            raise RuntimeError(f"Generated HIP launch at {phase} returned {status}")

    return call


def core_stream(torch, device):
    from infinicore.context import get_stream

    pointer = stream_pointer(get_stream())
    if not pointer or pointer == torch.cuda.default_stream(device).cuda_stream:
        raise ValueError("Core stream is unavailable or default")
    external = torch.cuda.ExternalStream(pointer, device=device)
    if external.cuda_stream != pointer:
        raise ValueError("Core and Torch external stream pointers differ")
    return pointer, external


def exact(metrics):
    return metrics["exact_equal"] and metrics["nonfinite_actual"] == metrics["nonfinite_reference"] == 0


def validate_dot(torch, settings, definition, launch, report):
    import ninetoothed as nt
    from infinicore.context import sync_stream

    if report["device_binary"]["shared"] <= 0:
        raise ValueError("AOT dot requires positive metadata.shared for the LDS test")
    core = load_core(torch, settings, report)
    for index in range(8):
        arch = torch.cuda.get_device_properties(index).gcnArchName.split(":", 1)[0]
        if arch != "gfx936":
            raise ValueError(f"Unexpected target on device {index}: {arch}")
    device = torch.device("cuda", settings["device"])
    rng = torch.Generator(device="cpu").manual_seed(42)
    left, right = (torch.randn((128, 128), dtype=torch.bfloat16, generator=rng) for _ in range(2))
    jit = nt.make(*definition, kernel_name="ss5_s3_dot_jit", num_warps=4, num_stages=1, max_num_configs=1)
    expected = torch.empty_like(left, device=device)
    jit(left.to(device), right.to(device), expected)
    torch.cuda.synchronize(device)
    expected = expected.cpu()
    torch.testing.assert_close(expected.float(), left.float() @ right.float(), rtol=0.02, atol=0.0625)

    first_launch = threading.Barrier(8, timeout=90)

    def rank_test(index):
        attempted = False
        try:
            torch.cuda.set_device(index)
            core.set_device(core.device("cuda", index))
            rank_device = torch.device("cuda", index)
            pointer, external = core_stream(torch, rank_device)
            with torch.cuda.stream(external):
                a, b = left.to(rank_device), right.to(rank_device)
                output = torch.full_like(a, float("nan"))
                first_launch.wait()
                attempted = True
                try:
                    launch(pointer, a, b, output)
                finally:
                    sync_stream()
            return {"device": index, "first_launch_attempted": attempted, "core_stream": pointer,
                    "error_vs_jit": error_metrics(output, expected)}
        except Exception:
            if not attempted:
                first_launch.abort()
            return {"device": index, "first_launch_attempted": attempted, "error": traceback.format_exc()}

    with ThreadPoolExecutor(max_workers=8) as threads:
        checks = list(threads.map(rank_test, range(8)))
    report["first_launch_barrier_participants"] = sum(check["first_launch_attempted"] for check in checks)
    report["rank_checks"] = checks
    return all("error_vs_jit" in check and exact(check["error_vs_jit"]) for check in checks)


def validate_adaln(torch, settings, launch, report):
    from infinicore.context import sync_stream

    core = load_core(torch, settings, report)
    device = torch.device("cuda", settings["device"])
    pointer, external = core_stream(torch, device)
    report["core_stream"] = pointer
    checks = []
    report["adaln_checks"] = checks
    rng = torch.Generator(device=device).manual_seed(20261110)
    with torch.cuda.stream(external):
        x = torch.randn((SEQUENCE, HIDDEN), dtype=torch.bfloat16, device=device, generator=rng)
        prod, partial, expected, output, consumed, expected_consumed = (torch.empty_like(x) for _ in range(6))
        cx, cp, ci, ce, co, cc, cr = (core_bridge(core, tensor) for tensor in
                                      (x, prod, partial, expected, output, consumed, expected_consumed))
        modulation = None
        for layout, slots, offset in (("msa", 6, 0), ("mlp", 6, 3), ("final_norm", 2, 0)):
            if modulation is None or modulation.shape[1] != slots * HIDDEN:
                modulation = torch.randn((SEQUENCE, slots * HIDDEN), dtype=x.dtype, device=device, generator=rng)
            shift = modulation[:, offset * HIDDEN:(offset + 1) * HIDDEN]
            scale = modulation[:, (offset + 1) * HIDDEN:(offset + 2) * HIDDEN]
            if shift.stride() != scale.stride() or shift.stride() != (slots * HIDDEN, 1):
                raise ValueError("Unexpected H3 modulation stride")
            cs, cb = (core_bridge(core, tensor) for tensor in (scale, shift))
            output.fill_(float("nan"))
            try:
                core.mul(cx, cs, out=cp)
                core.add(cx, cp, out=ci)
                core.add(ci, cb, out=ce)
                launch(pointer, x, scale, shift, output)
                core.add(co, cx, out=cc)
                core.add(ce, cx, out=cr)
            finally:
                sync_stream()
            metrics = error_metrics(output, expected)
            consumer = error_metrics(consumed, expected_consumed)
            checks.append({"layout": layout, "x": tensor_info(x), "scale": tensor_info(scale), "shift_offset": offset,
                           "output": metrics, "consumer": consumer})
            if not exact(metrics) or not exact(consumer):
                return False
    return True


def aot_probe(torch, case, settings, destination, report):
    import ninetoothed as nt
    import round1_kernels as kernels

    arrangement, application, names = {
        "aot_dot": (kernels.dot_arrangement, kernels.dot_application, ("left", "right", "output")),
        "aot_adaln": (kernels.elementwise_arrangement, kernels.adaln_application,
                      ("normalized", "scale", "shift", "output")),
        "aot_gate": (kernels.elementwise_arrangement, kernels.gate_application,
                     ("residual", "branch", "gate", "output")),
    }[case]
    dot = case == "aot_dot"
    width = 128 if dot else HIDDEN

    def definition():
        shape = (None, None) if dot else (None, HIDDEN)
        return arrangement, application, tuple(nt.Tensor(shape=shape, dtype=nt.bfloat16, name=name) for name in names)

    report["phase"] = "public_ninetoothed_aot_device_and_hip_host_build"
    launch = build_aot(destination, case, definition(), names, width, 128 if dot else None, report)
    report["phase"] = "core_stream_validation"
    if dot:
        return validate_dot(torch, settings, definition(), launch, report)
    if case == "aot_adaln":
        return validate_adaln(torch, settings, launch, report)
    return validate_core_stream(torch, settings, report, lambda stream, operands: launch(stream, *(operands[name] for name in names)))
