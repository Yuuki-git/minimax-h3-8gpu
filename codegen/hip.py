"""Package NineToothed generated HIP code for gfx936."""
from functools import partial
import importlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import patch
from common import dtk_roots, logged_command, sha256
from resources import kernel_resources

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
            command = [sys.executable, "-B", str(Path(__file__).with_name("compiler.py")),
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
    return library
