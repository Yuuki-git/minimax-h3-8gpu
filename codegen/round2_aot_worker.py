"""Probe DTK, compile a NineToothed gate, and check the Core stream ABI."""

import ast
import ctypes as ct
import hashlib
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import runpy
import shutil
import subprocess
import sys
import traceback
from types import SimpleNamespace
from unittest.mock import patch

from round1_worker import HIDDEN, SEQUENCE, core_bridge, error_metrics, load_core, load_module
from run_round1 import checked_child, sha256, write_json


KERNEL = "ss5_r2_gate"
ARGUMENTS = ("residual", "branch", "gate", "output")
WARPS, STAGES, WAVE_SIZE = 4, 1, 64
ARCH = "gfx936"
BLOCK = 1024  # Verified against the fixed round-one arrangement by the grid guard.
DRIVER_SYMBOLS = ("cuInit", "cuCtxGetId", "cuModuleLoadData", "cuModuleLoadDataEx",
                  "cuModuleGetFunction", "cuModuleUnload", "cuLaunchKernel")
HOST_SYMBOLS = (*DRIVER_SYMBOLS, "hipModuleLoadData", "hipModuleLoadDataEx",
                "hipModuleGetFunction", "hipModuleLaunchKernel", "hipModuleUnload")


def logged_command(command, directory, filename, timeout):
    with (directory / filename).open("wb") as log:
        return subprocess.run(command, cwd=directory, stdout=log, stderr=subprocess.STDOUT,
                              timeout=timeout, check=False).returncode


def dtk_roots():
    roots = [Path(path) for path in ("/opt/dtk/cuda/cuda", "/opt/dtk", "/opt/hyhal", "/opt/rocm", "/usr/local/cuda")]
    roots[:0] = [Path(os.environ[key]) for key in ("HYGON_CUDA_HOME", "DTK_ROOT", "CUDA_HOME")
                 if os.environ.get(key) and Path(os.environ[key]).is_absolute()]
    return list(dict.fromkeys(roots))


def driver_probe(directory, report):
    output = directory / "driver"
    output.mkdir()
    roots = dtk_roots()
    includes = list(dict.fromkeys(root / "include" for root in roots if (root / "include/cuda.h").is_file()))
    libraries = list(dict.fromkeys(root / sub for root in roots for sub in ("lib64", "lib")
                                  if (root / sub).is_dir()))
    report["cuda_headers"] = [{"path": str(path / "cuda.h"), "sha256": sha256(path / "cuda.h")} for path in includes]
    report["tools"] = {name: shutil.which(name) for name in ("cc", "nvcc", "clang", "llc", "readelf")}
    for name, tool in report["tools"].items():
        if tool:
            logged_command([tool, "--version"], output, f"{name}-version.log", 10)
    checks = {}
    for library in ("libcuda.so", "libcuda.so.1", "libamdhip64.so", "libamdhip64.so.6"):
        try:
            handle = ct.CDLL(library)
            checks[library] = {"symbols": [name for name in HOST_SYMBOLS if hasattr(handle, name)]}
        except OSError as exc:
            checks[library] = {"load_error": str(exc)}
    report["driver_dynamic_load"] = checks
    if not report["tools"]["cc"]:
        report["link_probe"] = "C compiler unavailable"
        return False
    source = output / "cuda-driver.c"
    # Compile and link only. Taking symbol addresses requires the actual driver API.
    declarations = "\n".join(f"void (*volatile probe_{name})(void) = (void (*)(void))&{name};" for name in DRIVER_SYMBOLS)
    source.write_text(f"#include <cuda.h>\n{declarations}\nint main(void) {{ return 0; }}\n", encoding="utf-8")
    flags = [f"-I{path}" for path in includes]
    cc = report["tools"]["cc"]
    status = logged_command([cc, *flags, "-c", source.name, "-o", "driver.o"], output, "headers.log", 20)
    report["header_compile_exit"] = status
    if status == 0:
        flags = [f"-L{path}" for path in libraries]
        report["driver_link_exit"] = logged_command([cc, "driver.o", *flags, "-lcuda", "-o", "driver-link"],
                                                    output, "link.log", 20)
    return report.get("driver_link_exit") == 0


def gate_definition():
    import ninetoothed as nt
    from round1_kernels import elementwise_arrangement, gate_application

    tensors = tuple(nt.Tensor(shape=(None, HIDDEN), dtype=nt.bfloat16, name=name) for name in ARGUMENTS)
    return elementwise_arrangement, gate_application, tensors


def binary_evidence(compiled, output):
    metadata = json.loads(json.dumps(compiled.metadata._asdict(), default=str))
    code = compiled.asm["hsaco"]
    if code[:6] != b"\x7fELF\x02\x01" or int.from_bytes(code[18:20], "little") != 224:
        raise ValueError("Expected a little-endian AMDGPU ELF64 code object")
    binary = output / "gate.hsaco"
    binary.write_bytes(code)
    llir = compiled.asm.get("llir")
    if llir:
        (output / "gate.ll").write_text(llir, encoding="utf-8")
    readelf = shutil.which("readelf")
    if readelf:
        logged_command([readelf, "-h", "-n", binary.name], output, "code-object.log", 20)
    metadata.update(hsaco_sha256=sha256(binary), hsaco_bytes=len(code))
    write_json(output / "metadata.json", metadata)
    return metadata


def offline_probe(directory, report):
    import triton
    from triton.backends.compiler import GPUTarget
    from triton.compiler import ASTSource
    from ninetoothed import Tensor
    from ninetoothed.generation import CodeGenerator
    from round1_llvm_compat import install_legacy_gep_adapter

    output = directory / "offline"
    output.mkdir()
    report["phase"] = "fixed_tile_ninetoothed_generation"
    arrangement, application, tensors = gate_definition()
    application.__annotations__ = dict(zip(ARGUMENTS, arrangement(*tensors), strict=True))
    # The pinned constructor queries device properties solely for auto-tuning
    # bounds. This fixed 1024-element schedule has no tuneable symbols.
    generator = CodeGenerator.__new__(CodeGenerator)
    ast.NodeTransformer.__init__(generator)
    generator._min_num_elements = 1
    generator._max_num_elements = 2**15
    source = generator(application, caller="cuda", kernel_name=KERNEL, num_warps=WARPS,
                       num_stages=STAGES, max_num_configs=None, prettify=False)
    shutil.copyfile(source, output / "gate.py")
    module = load_module("ss5_round2_offline_gate", source)
    kernel = getattr(module, KERNEL)
    signature = {}
    for arg in kernel.arg_names:
        if Tensor.pointer_pattern().fullmatch(arg):
            signature[arg] = "*bf16"
        elif Tensor.size_pattern().fullmatch(arg) or Tensor.stride_pattern().fullmatch(arg):
            signature[arg] = "i32"
        else:
            raise ValueError(f"Unexpected generated gate parameter: {arg}")
    target = GPUTarget("hip", ARCH, WAVE_SIZE)
    write_json(output / "compile-contract.json", {
        "entry": KERNEL, "arguments": list(signature.items()), "target": str(target),
        "num_warps": WARPS, "num_stages": STAGES, "grid_ast": ast.unparse(generator.raw_grid),
        "source_sha256": sha256(source), "specialization": "no constexpr or divisibility hints in the compile signature",
    })
    report["phase"] = "explicit_target_device_compile"
    install_legacy_gep_adapter(directory, "offline")
    compiled = triton.compile(ASTSource(kernel, signature), target=target,
                              options={"num_warps": WARPS, "num_stages": STAGES})
    if compiled.metadata.target != target:
        raise ValueError("Compiled device target differs from gfx936 wave64")
    report["device_binary"] = binary_evidence(compiled, output)
    return True


def compile_child(directory, case, arguments):
    import triton
    from round1_llvm_compat import install_legacy_gep_adapter

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


class NineToothedTensor(ct.Structure):
    _fields_ = [("data", ct.c_void_p), ("shape", ct.POINTER(ct.c_uint64)), ("strides", ct.POINTER(ct.c_int64))]

    @classmethod
    def from_torch(cls, tensor):
        return cls(tensor.data_ptr(), (ct.c_uint64 * 2)(*tensor.shape), (ct.c_int64 * 2)(*tensor.stride()))


def stream_pointer(stream):
    if isinstance(stream, int):
        return stream
    get_name = ct.pythonapi.PyCapsule_GetName
    get_name.argtypes, get_name.restype = (ct.py_object,), ct.c_char_p
    get_pointer = ct.pythonapi.PyCapsule_GetPointer
    get_pointer.argtypes, get_pointer.restype = (ct.py_object, ct.c_char_p), ct.c_void_p
    return get_pointer(stream, get_name(stream))


def validate_core_stream(torch, settings, report, launch):
    from infinicore.context import get_stream, sync_stream

    core = load_core(torch, settings, report)
    device = torch.device("cuda", settings["device"])
    stream = stream_pointer(get_stream())
    if not stream or stream == torch.cuda.default_stream(device).cuda_stream:
        raise RuntimeError("Core did not return a non-default stream")
    external = torch.cuda.ExternalStream(stream, device=device)
    report["stream"] = {"core_pointer": stream, "torch_external_pointer": external.cuda_stream,
                        "scope": "Core producer, native AOT launch, Core consumers, then Core stream synchronize"}
    if external.cuda_stream != stream:
        raise ValueError("Torch ExternalStream pointer differs from Core stream")
    results = []
    report["gate_checks"] = results
    with torch.cuda.stream(external):
        for rows, offset in ((33, 5), (SEQUENCE, 2)):
            rng = torch.Generator(device=device).manual_seed(rows)
            options = {"dtype": torch.bfloat16, "device": device}
            residual, branch_seed = (torch.randn((rows, HIDDEN), generator=rng, **options) for _ in range(2))
            modulation = torch.randn((rows, 6 * HIDDEN), generator=rng, **options)
            gate = modulation[:, offset * HIDDEN:(offset + 1) * HIDDEN]
            branch, product, expected, output, consumed, expected_consumer = (
                torch.empty_like(residual) for _ in range(6))
            output.fill_(float("nan"))
            tensors = (residual, branch_seed, gate, branch, product, expected, output, consumed, expected_consumer)
            cr, cs, cg, cb, cp, ce, co, cc, cec = (core_bridge(core, tensor) for tensor in tensors)
            if gate.stride() != (6 * HIDDEN, 1):
                raise ValueError("Gate no longer has the H3 modulation stride")
            operands = dict(zip(ARGUMENTS, (residual, branch, gate, output), strict=True))
            try:
                core.add(cs, cr, out=cb)
                core.mul(cg, cb, out=cp)
                core.add(cr, cp, out=ce)
                result = launch(stream, operands)
                if result:
                    raise RuntimeError(f"AOT launch returned {result}")
                core.add(co, cr, out=cc)
                core.add(ce, cr, out=cec)
            finally:
                sync_stream()
            metrics = error_metrics(output, expected)
            consumer_metrics = error_metrics(consumed, expected_consumer)
            results.append({"shape": [rows, HIDDEN], "gate_offset": offset, "gate_stride": list(gate.stride()),
                            "output": metrics, "consumer": consumer_metrics})
            if not all(m["exact_equal"] and m["nonfinite_actual"] == m["nonfinite_reference"] == 0
                       for m in (metrics, consumer_metrics)):
                return False
    return True


def select_device(directory, report):
    import torch
    from triton.runtime import driver

    settings = json.loads((directory / "settings.json").read_text(encoding="utf-8"))
    device = settings["device"]
    if not torch.cuda.is_available() or torch.cuda.device_count() != 8:
        raise RuntimeError("Run the native probe in a job with 8 visible GPUs")
    torch.cuda.set_device(device)
    torch.set_num_threads(8)
    target = driver.active.get_current_target()
    report["target"] = str(target)
    if (target.backend, target.arch, target.warp_size) != ("hip", ARCH, WAVE_SIZE):
        raise ValueError("Expected gfx936 wave64 on the selected GPU")
    return torch, settings


def native_probe(directory, report):
    import ninetoothed as nt

    torch, settings = select_device(directory, report)
    output = directory / "native"
    output.mkdir()
    aot = importlib.import_module("ninetoothed.aot")
    compiler_calls = []
    report["compiler_calls"] = compiler_calls

    def run_tool(command, **kwargs):
        if command[:3] == ["python", "-m", "triton.tools.compile"]:
            case = f"aot_variant_{len(compiler_calls):02d}"
            command = [sys.executable, "-B", __file__, "compiler", str(directory), case, *command[3:]]
        elif command[0] == "nvcc":
            command = list(command)
            command[command.index("--threads") + 1] = "8"
        else:
            raise ValueError("Unexpected command in pinned NineToothed AOT")
        compiler_calls.append([str(part) for part in command])
        print(f"AOT_TOOL={Path(command[0]).name} INDEX={len(compiler_calls)}", flush=True)
        return subprocess.run(command, **kwargs)

    report["phase"] = "ninetoothed_public_aot"
    try:
        with patch.object(aot, "subprocess", SimpleNamespace(run=run_tool)):
            nt.make(*gate_definition(), caller="cuda", kernel_name=KERNEL, output_dir=output,
                    num_warps=WARPS, num_stages=STAGES)
    finally:
        header = checked_child(aot._HEADER_PATH, directory)
        if header.is_file():
            shutil.copyfile(header, output / "ninetoothed.h")
        report["launcher_api_symbols"] = sorted({symbol for path in output.rglob("*")
            if path.is_file() and path.suffix in (".c", ".cpp", ".h")
            for symbol in HOST_SYMBOLS if re.search(r"\b" + symbol + r"\b", path.read_text(encoding="utf-8"))})

    report["phase"] = "core_stream_validation"
    header = (output / f"{KERNEL}.h").read_text(encoding="utf-8")
    declaration = re.search(rf"launch_{KERNEL}\(([^)]*)\)", header)
    params = [param.split() for param in declaration.group(1).split(",")] if declaration else []
    if (len(params) != 5 or params[0] != ["NineToothedStream", "stream"] or
            any(len(param) != 2 or param[0] != "NineToothedTensor" for param in params[1:])):
        raise ValueError("Unexpected generated host launch signature")
    order = [param[1] for param in params[1:]]
    if set(order) != set(ARGUMENTS):
        raise ValueError(f"Unexpected generated tensor argument names: {order}")
    report["launch_signature"] = params
    library = output / f"{KERNEL}.so"
    report["launcher_sha256"] = sha256(library)
    handle = ct.CDLL(str(library))
    launch_native = getattr(handle, f"launch_{KERNEL}")
    launch_native.argtypes = (ct.c_void_p,) + (NineToothedTensor,) * 4
    launch_native.restype = ct.c_int

    def launch(stream, operands):
        return launch_native(stream, *(NineToothedTensor.from_torch(operands[name]) for name in order))

    return validate_core_stream(torch, settings, report, launch)


def hip_launch_contract(output, metadata):
    """Allow only the fixed 32-bit gate ABI emitted by the pinned generator."""
    contract = json.loads((output / "compile-contract.json").read_text(encoding="utf-8"))
    fields = {"pointer": "*bf16", "size_0": "i32", "stride_0": "i32", "stride_1": "i32"}
    expected = {f"ninetoothed_{name}_{field}": dtype for name in ARGUMENTS for field, dtype in fields.items()}
    arguments = contract["arguments"]
    if len(arguments) != len(expected) or dict(arguments) != expected:
        raise ValueError("Offline argument names or types differ from the fixed gate ABI")
    grid = f"lambda meta: ((ninetoothed_residual_size_0 * {HIDDEN} - {BLOCK - 1} - 1 + {BLOCK} - 1) // {BLOCK} + 1,)"
    if contract["grid_ast"] != grid or contract["entry"] != KERNEL:
        raise ValueError("Unexpected NineToothed grid or entry symbol")
    if (metadata["num_warps"] != WARPS or metadata["warp_size"] != WAVE_SIZE or
            metadata.get("global_scratch_size", 0) or metadata.get("profile_scratch_size", 0) or
            metadata.get("launch_cooperative_grid", False)):
        raise ValueError("Offline metadata requires a launch mode outside the fixed gate ABI")
    ir = (output / "gate.ll").read_text(encoding="utf-8")
    definition = re.search(rf"^define amdgpu_kernel void @{KERNEL}\((.*)\)[^\n]*\{{", ir, re.MULTILINE)
    params = re.split(r",\s+(?=(?:ptr addrspace\(1\)|i32)\s)", definition.group(1)) if definition else []
    llvm_types = ["ptr addrspace(1)" if dtype == "*bf16" else "i32" for _, dtype in arguments]
    llvm_types += ["ptr addrspace(1)"] * 2
    if (len(params) != len(llvm_types) or
            any(not param.startswith(ty + " ") for param, ty in zip(params, llvm_types)) or
            any("readnone" not in param.split() for param in params[-2:])):
        raise ValueError("Offline LLVM ABI differs from the checked argument types and two unused scratch slots")
    return [name.removeprefix("ninetoothed_").split("_", 1) for name, _ in arguments]


def hip_probe(directory, report):
    from triton.backends.amd import driver

    torch, settings = select_device(directory, report)
    offline = directory / "offline"
    metadata = json.loads((offline / "metadata.json").read_text(encoding="utf-8"))
    binary = offline / "gate.hsaco"
    if sha256(binary) != metadata["hsaco_sha256"]:
        raise ValueError("Offline binary hash differs")
    arguments = hip_launch_contract(offline, metadata)
    report["checked_kernel_arguments"] = arguments
    output = directory / "hip"
    output.mkdir()
    source = output / "hip_gate_launcher.cc"
    shutil.copyfile(Path(__file__).with_name(source.name), source)
    cxx = shutil.which("c++") or shutil.which("clang++")
    include_roots = [Path(driver.__file__).parent / "include", *(root / "include" for root in dtk_roots())]
    includes = [f"-I{root}" for root in include_roots if (root / "hip/hip_runtime_api.h").is_file()]
    runtime = Path(driver._get_path_to_hip_runtime_dylib()).resolve(strict=True)
    report["hip_runtime"] = {"path": str(runtime), "sha256": sha256(runtime)}
    report["phase"] = "build_experimental_hip_launcher"
    if not cxx or logged_command([cxx, "-shared", "-fPIC", "-D__HIP_PLATFORM_AMD__", *includes,
                                  source.name, str(runtime), "-Wl,--no-undefined", "-o", "launcher.so"],
                                  output, "build.log", 60):
        raise RuntimeError("Experimental HIP launcher build failed")
    library = output / "launcher.so"
    report["launcher_sha256"] = sha256(library)
    handle = ct.CDLL(str(library))
    pointer = ct.c_void_p
    handle.ss5_load.argtypes = (pointer, ct.c_char_p, ct.POINTER(pointer), ct.POINTER(pointer))
    handle.ss5_launch.argtypes = (pointer, pointer, ct.c_uint, ct.c_uint, ct.c_uint, ct.POINTER(pointer))
    handle.ss5_unload.argtypes = (pointer,)
    image = ct.create_string_buffer(binary.read_bytes())
    module, function = pointer(), pointer()
    status = handle.ss5_load(image, KERNEL.encode("ascii"), ct.byref(module), ct.byref(function))
    if status:
        raise RuntimeError(f"hipModuleLoadData/hipModuleGetFunction returned {status}")

    def launch(stream, operands):
        rows, columns = operands["residual"].shape
        scalars = {"size_0": lambda t: t.shape[0], "stride_0": lambda t: t.stride(0), "stride_1": lambda t: t.stride(1)}
        values = [pointer(operands[name].data_ptr()) if field == "pointer"
                  else ct.c_int32(scalars[field](operands[name])) for name, field in arguments]
        values.extend((pointer(), pointer()))
        params = (pointer * len(values))(*(ct.addressof(value) for value in values))
        grid = (rows * columns + BLOCK - 1) // BLOCK
        return handle.ss5_launch(function, stream, grid, WAVE_SIZE * WARPS, metadata["shared"], params)

    report["phase"] = "experimental_hip_core_stream_validation"
    try:
        matched = validate_core_stream(torch, settings, report, launch)
    finally:
        report["hip_module_unload_exit"] = handle.ss5_unload(module)
    return matched and report["hip_module_unload_exit"] == 0


def main():
    case, destination, *extra = sys.argv[1:]
    directory = Path(destination).resolve(strict=True)
    if case == "compiler":
        compile_child(directory, extra[0], extra[1:])
        return 0
    probes = {"driver": driver_probe, "offline": offline_probe, "native": native_probe, "hip": hip_probe}
    report = {"case": case, "completed": False, "validation_passed": False, "phase": "preflight"}
    try:
        if case != "driver":
            import triton
            from triton.runtime.cache import get_cache_manager
            from round1_triton_cache import WorkspaceCacheManager

            report["environment"] = {name: importlib.metadata.version(name) for name in ("torch", "triton", "ninetoothed", "numpy")}
            report["triton_module_sha256"] = sha256(triton.__file__)
            manager = get_cache_manager(hashlib.sha256(b"ss5-round2-aot-preflight").hexdigest())
            if not isinstance(manager, WorkspaceCacheManager):
                raise RuntimeError("Vendor Triton did not select the workspace cache adapter")
            blob = manager.put(b"ss5-round2-aot-preflight", "probe.bin")
            if manager.get_file("probe.bin") != blob or Path(blob).read_bytes() != b"ss5-round2-aot-preflight":
                raise RuntimeError("Workspace cache publication preflight failed")
        report["validation_passed"] = probes[case](directory, report)
        report["completed"] = True
    except Exception:
        report["error"] = traceback.format_exc()
        print(report["error"], file=sys.stderr, flush=True)
    write_json(directory / f"{case}.json", report)
    print(json.dumps({key: report[key] for key in ("case", "phase", "completed", "validation_passed")}), flush=True)
    return 0 if report["validation_passed"] else 3


if __name__ == "__main__":
    sys.exit(main())
