"""One isolated GPU experiment. Parent configures every writable path first."""

import gc
import hashlib
import importlib.metadata
import importlib.util
import json
import math
from pathlib import Path
import statistics
import sys
import threading
import time
import traceback


SEQUENCE, HEADS, DIM, HIDDEN = 27570, 7, 128, 5376


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for data in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(data)
    return digest.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False,
                                    allow_nan=False) + "\n", encoding="utf-8")


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def tensor_info(value):
    return {"shape": list(value.shape), "stride": list(value.stride()),
            "dtype": str(value.dtype), "device": str(value.device)}


def error_metrics(actual, reference, chunk=262144):
    """Bound CPU scratch memory; BF16 and FP32 references are both supported."""
    import torch

    if actual.shape != reference.shape:
        raise ValueError("Reference and output shapes differ")
    actual, reference = actual.detach().cpu().reshape(-1), reference.detach().cpu().reshape(-1)
    count, bad_actual, bad_reference, unequal = actual.numel(), 0, 0, 0
    abs_sum = squared = ref_squared = act_squared = inner = maximum = 0.0
    for start in range(0, count, chunk):
        a = actual[start:start + chunk].double()
        b = reference[start:start + chunk].double()
        good_a, good_b = torch.isfinite(a), torch.isfinite(b)
        bad_actual += int((~good_a).sum())
        bad_reference += int((~good_b).sum())
        unequal += int((a != b).sum())
        valid = good_a & good_b
        a, b = a[valid], b[valid]
        if not a.numel():
            continue
        difference = a - b
        maximum = max(maximum, float(difference.abs().max()))
        abs_sum += float(difference.abs().sum())
        squared += float(difference.square().sum())
        ref_squared += float(b.square().sum())
        act_squared += float(a.square().sum())
        inner += float((a * b).sum())
    finite = bad_actual == bad_reference == 0
    return {"elements": count, "nonfinite_actual": bad_actual,
            "nonfinite_reference": bad_reference, "exact_equal": unequal == 0,
            "unequal_elements": unequal, "max_abs": maximum if finite else None,
            "mae": abs_sum / count if finite else None,
            "rmse": math.sqrt(squared / count) if finite else None,
            "relative_rmse": math.sqrt(squared / ref_squared) if finite and ref_squared else None,
            "cosine": inner / math.sqrt(ref_squared * act_squared) if finite and ref_squared * act_squared else None}


def synchronized_benchmark(torch, function, settings):
    """Use completed wall time: Core and Torch need not share a stream."""
    device = settings["device"]
    started = time.perf_counter()
    function()
    torch.cuda.synchronize(device)
    first_call_ms = (time.perf_counter() - started) * 1000
    for _ in range(settings["warmup"]):
        function()
        torch.cuda.synchronize(device)
    samples = []
    for _ in range(settings["repeats"]):
        torch.cuda.synchronize(device)
        started = time.perf_counter()
        function()
        torch.cuda.synchronize(device)
        samples.append((time.perf_counter() - started) * 1000)

    # Memory is a SEPARATE pass: sampling does not contaminate timing samples.
    gc.collect()
    torch.cuda.synchronize(device)
    baseline = torch.cuda.memory_allocated(device)
    torch.cuda.reset_peak_memory_stats(device)
    samples_used = [torch.cuda.mem_get_info(device)[1] - torch.cuda.mem_get_info(device)[0]]
    stop = threading.Event()

    def sample_memory():
        torch.cuda.set_device(device)
        while not stop.is_set():
            free, total = torch.cuda.mem_get_info(device)
            samples_used.append(total - free)
            stop.wait(0.005)

    sampler = threading.Thread(target=sample_memory, daemon=True)
    sampler.start()
    try:
        function()
        torch.cuda.synchronize(device)
    finally:
        stop.set()
        sampler.join()
    return {"timing_scope": "single-GPU host call plus completion synchronize; warmed; no profiler",
            "first_call_including_jit_ms": first_call_ms, "samples_ms": samples,
            "median_ms": statistics.median(samples), "min_ms": min(samples),
            "max_ms": max(samples),
            "memory": {"baseline_torch_allocated_bytes": baseline,
                       "peak_torch_allocated_bytes": torch.cuda.max_memory_allocated(device),
                       "peak_torch_reserved_bytes": torch.cuda.max_memory_reserved(device),
                       "incremental_torch_allocated_bytes": torch.cuda.max_memory_allocated(device) - baseline,
                       "sampled_whole_device_used_peak_bytes": max(samples_used),
                       "whole_device_sample_count": len(samples_used), "sampling_interval_ms": 5,
                       "scope": "Torch peak exact for its allocator; device samples are a lower bound, include all users, and may miss short native allocations"}}


def generated_source(handle, report):
    path = Path(handle._source)
    report["generated_source"] = {"path": str(path), "sha256": sha256(path),
                                  "contains_tl_dot": "dot(" in path.read_text(encoding="utf-8")}


def core_bridge(core, tensor):
    # Fixed Core from_torch uses from_blob and discards strides. Do not use it!
    wrapped = core.strided_from_blob(tensor.data_ptr(), list(tensor.shape), list(tensor.stride()),
                                     dtype=core.bfloat16,
                                     device=core.device("cuda", tensor.device.index or 0))
    if list(wrapped.stride()) != list(tensor.stride()):
        raise ValueError("Core bridge discarded physical strides")
    return wrapped


def load_core(torch, settings, report):
    import infinicore as core

    core.set_device(core.device("cuda", settings["device"]))
    expected = settings["baseline_libraries"]
    loaded = []
    for line in Path("/proc/self/maps").read_text(encoding="utf-8").splitlines():
        fields = line.split(maxsplit=5)
        if len(fields) != 6 or not fields[5].startswith("/"):
            continue
        path = Path(fields[5])
        if path.name in expected and path not in [Path(x["path"]) for x in loaded]:
            digest = sha256(path)
            if digest != expected[path.name]:
                raise ValueError(f"Unexpected baseline binary: {path.name}")
            loaded.append({"path": str(path), "sha256": digest})
    required = {"libinfinirt.so", "libinfiniop.so", "libinfinicore_cpp_api.so"}
    if not required <= {Path(x["path"]).name for x in loaded}:
        raise ValueError("Cannot establish loaded fixed Core binary identity")
    report["loaded_core_libraries"] = loaded
    report["core_python"] = {"path": core.__file__, "sha256": sha256(core.__file__)}
    return core


def attention_inputs(torch, device):
    generator = torch.Generator(device=device).manual_seed(42)
    return tuple(torch.randn((1, SEQUENCE, HEADS, DIM), dtype=torch.bfloat16,
                             device=device, generator=generator) for _ in range(3))


def input_digest(tensors):
    # These are synthetic, seeded inputs, not user data or model weights.
    return [hashlib.sha256(x.cpu().view(__import__("torch").uint8).numpy().tobytes()).hexdigest()
            for x in tensors]


def fp32_attention_reference(torch, q, k, v, query_chunk=128):
    """Full FP32 reference, chunked over Q; all K/V and all output rows used."""
    qh, kh, vh = (x.permute(0, 2, 1, 3).float() for x in (q, k, v))
    result = torch.empty(q.shape, dtype=torch.float32, device="cpu")
    for start in range(0, q.shape[1], query_chunk):
        scores = torch.matmul(qh[:, :, start:start + query_chunk], kh.transpose(-1, -2))
        scores.mul_(1 / math.sqrt(q.shape[-1]))
        probability = torch.softmax(scores, dim=-1)
        block = torch.matmul(probability, vh).permute(0, 2, 1, 3)
        result[:, start:start + query_chunk].copy_(block.cpu())
        del scores, probability, block
    return result


def profile_core_mha(torch, function, report):
    try:
        from torch.autograd import DeviceType
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                torch.profiler.ProfilerActivity.CUDA],
                                    record_shapes=True) as profiler:
            function()
            torch.cuda.synchronize()
        groups = {}
        for event in profiler.events():
            if event.device_type != DeviceType.CUDA:
                continue
            item = groups.setdefault(event.name, {"name": event.name, "count": 0,
                                                  "device_event_ms_sum": 0.0})
            item["count"] += 1
            item["device_event_ms_sum"] += (event.time_range.end - event.time_range.start) / 1000
        report["isolated_mha_profile"] = sorted(groups.values(), key=lambda x: -x["device_event_ms_sum"])
        report["isolated_mha_event_counts"] = {
            "softmax": sum(x["count"] for name, x in groups.items() if "softmax" in name.lower()),
            "aten_scalar_mul": sum(x["count"] for name, x in groups.items() if "at::native::" in name and "MulFunctor" in name),
            "gemm_family": sum(x["count"] for name, x in groups.items() if name.startswith(("Cijk_", "Cij_")) or "gemm" in name.lower()),
        }
        report["isolated_mha_profile_scope"] = "one extra call after untraced timing; use to verify GEMM/scale/softmax kernel-family attribution"
    except Exception:
        report["isolated_mha_profile_error"] = traceback.format_exc()


def attention_case(torch, case, settings, destination, report):
    device = torch.device("cuda", settings["device"])
    q, k, v = attention_inputs(torch, device)
    report["inputs"] = [tensor_info(x) for x in (q, k, v)]
    report["input_sha256"] = input_digest((q, k, v))
    report["input_scope"] = "seeded Gaussian BF16 data at real local shape; not captured model activations"
    torch.cuda.synchronize(device)
    reference_path = destination / "large/fp32-attention-reference.pt"
    if case == "reference":
        started = time.perf_counter()
        result = fp32_attention_reference(torch, q, k, v)
        torch.cuda.synchronize(device)
        report["reference_compute_seconds"] = time.perf_counter() - started
        reference_path.parent.mkdir(exist_ok=True)
        torch.save({"output": result, "input_sha256": report["input_sha256"]}, reference_path)
        report.update(reference_sha256=sha256(reference_path), reference_scope="full output, FP32 QK/scale/softmax/PV; Q chunks of 128; TF32 disabled", validation_passed=True)
        return
    output = torch.empty_like(q)
    scale = 1 / math.sqrt(DIM)
    if case == "core_mha":
        core = load_core(torch, settings, report)
        cq, ck, cv, co = (core_bridge(core, x) for x in (q, k, v, output))
        function = lambda: core.mha(cq, ck, cv, scale=scale, is_causal=False, out=co)
    elif case == "nt_attention":
        import round1_kernels
        pinned = load_module("ss5_pinned_sdpa", Path(__file__).parent / "pinned_sdpa.py")
        kernel = round1_kernels.make_attention(pinned)
        generated_source(kernel, report)
        meta_mask = torch.empty((1, HEADS, SEQUENCE, SEQUENCE), device="meta")
        qh, kh, vh, oh = (x.permute(0, 2, 1, 3) for x in (q, k, v, output))
        report["kernel_inputs"] = [tensor_info(x) for x in (qh, kh, vh)]
        function = lambda: kernel(qh, kh, vh, meta_mask, False, scale, oh, False, pinned.CausalVariant.UPPER_LEFT)
    elif case == "vendor_flash":
        import flash_attn_2_cuda
        from flash_attn import flash_attn_func
        report["vendor_extension"] = {"path": flash_attn_2_cuda.__file__, "sha256": sha256(flash_attn_2_cuda.__file__)}
        report["use_scope"] = "performance reference only; never registered into the formal model path"

        def function():
            nonlocal output
            output = flash_attn_func(q, k, v, dropout_p=0.0, softmax_scale=scale, causal=False)
    else:
        raise ValueError(case)
    report["output_allocation"] = "vendor API allocates output; Core/NT use preallocated output; allocation policies are explicitly reported"
    report["benchmark"] = synchronized_benchmark(torch, function, settings)
    if case == "core_mha":
        profile_core_mha(torch, function, report)
    if not reference_path.is_file():
        report.update(validation_passed=False, validation_unavailable="full FP32 reference job did not complete")
        return
    reference = torch.load(reference_path, map_location="cpu", weights_only=True)
    if reference["input_sha256"] != report["input_sha256"]:
        raise ValueError("FP32 reference inputs differ from benchmark inputs")
    report["error_vs_full_fp32"] = metrics = error_metrics(output, reference["output"])
    tolerance = settings["attention_tolerance"]
    report["validation_passed"] = (metrics["nonfinite_actual"] == metrics["nonfinite_reference"] == 0
                                   and metrics["relative_rmse"] is not None
                                   and metrics["relative_rmse"] <= tolerance["relative_rmse"]
                                   and metrics["max_abs"] <= tolerance["max_abs"])


def fusion_case(torch, case, settings, report):
    device = torch.device("cuda", settings["device"])
    generator = torch.Generator(device=device).manual_seed(20261109)
    x, branch = (torch.randn((SEQUENCE, HIDDEN), device=device, dtype=torch.bfloat16,
                             generator=generator) for _ in range(2))
    modulation = torch.randn((SEQUENCE, 6 * HIDDEN), device=device,
                              dtype=torch.bfloat16, generator=generator)
    offset = 0 if case.endswith("msa") else 3
    shift, scale, gate = tuple(modulation[:, (offset + i) * HIDDEN:(offset + i + 1) * HIDDEN] for i in range(3))
    report["inputs"] = {name: tensor_info(value) for name, value in
                        (("x", x), ("branch", branch), ("shift", shift), ("scale", scale), ("gate", gate))}
    if any(t.stride() != (6 * HIDDEN, 1) for t in (shift, scale, gate)):
        raise ValueError("Unexpected modulation slice stride")
    is_adaln = "adaln" in case
    output = torch.empty_like(x)
    # Torch reference preserves the same BF16 operation boundaries as fixed LM.
    reference = (x + x * scale) + shift if is_adaln else x + gate * branch
    torch.cuda.synchronize(device)
    if case.startswith("core_"):
        core = load_core(torch, settings, report)
        temporary = torch.empty_like(x)
        intermediate = torch.empty_like(x) if is_adaln else None
        cx, cb, cs, cc, cg, co, ct = (core_bridge(core, t) for t in
                                     (x, branch, shift, scale, gate, output, temporary))
        if is_adaln:
            ci = core_bridge(core, intermediate)

            def function():
                core.mul(cx, cc, out=ct)
                core.add(cx, ct, out=ci)
                core.add(ci, cs, out=co)
        else:
            def function():
                core.mul(cg, cb, out=ct)
                core.add(cx, ct, out=co)
    else:
        import round1_kernels
        kernel = round1_kernels.make_elementwise("adaln" if is_adaln else "gate")
        generated_source(kernel, report)
        function = (lambda: kernel(x, scale, shift, output)) if is_adaln else (lambda: kernel(x, branch, gate, output))
    report["output_allocation"] = "preallocated output; Core uses 2 AdaLN/1 residual scratch tensors, NT uses none; original Core call sequence, not original model allocation overhead"
    report["benchmark"] = synchronized_benchmark(torch, function, settings)
    report["error_vs_bf16_sequence"] = metrics = error_metrics(output, reference)
    report["validation_passed"] = metrics["exact_equal"] and metrics["nonfinite_actual"] == 0


def smoke_case(torch, case, settings, report):
    import round1_kernels

    device = torch.device("cuda", settings["device"])
    generator = torch.Generator(device=device).manual_seed(42)
    if case == "smoke_elementwise":
        left, right = (torch.randn((8193,), device=device, generator=generator) for _ in range(2))
        output = torch.empty_like(left)
        report["phase"] = "ninetoothed_elementwise_source_generation"
        kernel = round1_kernels.make_elementwise("add")
        reference = left + right
    else:
        left, right = (torch.randn((128, 128), device=device, dtype=torch.bfloat16,
                                   generator=generator) for _ in range(2))
        output = torch.empty_like(left)
        report["phase"] = "ninetoothed_dot_source_generation"
        kernel = round1_kernels.make_dot()
        reference = left.float() @ right.float()
    generated_source(kernel, report)
    report["inputs"] = [tensor_info(left), tensor_info(right)]
    report["phase"] = "jit_compile_execute_and_measure"
    report["benchmark"] = synchronized_benchmark(torch, lambda: kernel(left, right, output), settings)
    report["phase"] = "numerical_validation"
    report["error"] = metrics = error_metrics(output, reference)
    if case == "smoke_elementwise":
        report["validation_passed"] = metrics["exact_equal"] and metrics["nonfinite_actual"] == 0
    else:
        torch.testing.assert_close(output.float(), reference, rtol=0.02, atol=0.0625)
        report["validation_passed"] = bool(report["generated_source"]["contains_tl_dot"])


def main():
    case, directory = sys.argv[1:]
    destination = Path(directory)
    settings = json.loads((destination / "settings.json").read_text(encoding="utf-8"))
    report = {"case": case, "completed": False, "validation_passed": False,
              "platform_run": True, "device": settings["device"]}
    exit_code = 0
    try:
        report["phase"] = "import_torch"
        import torch
        report["phase"] = "import_triton"
        import triton
        report["phase"] = "device_preflight"
        report["cuda_available"] = torch.cuda.is_available()
        report["visible_gpu_count"] = torch.cuda.device_count()
        if not report["cuda_available"] or report["visible_gpu_count"] != 8:
            raise RuntimeError("Expected this job's 8 visible GPUs")
        torch.cuda.set_device(settings["device"])
        properties = torch.cuda.get_device_properties(settings["device"])
        if "gfx936" not in str(getattr(properties, "gcnArchName", "")):
            raise RuntimeError("Selected GPU is not gfx936")
        torch.set_num_threads(8)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")
        report["phase"] = "environment_inventory"
        report["environment"] = {
            "python": sys.version, "torch": torch.__version__, "hip": torch.version.hip,
            "triton": getattr(triton, "__version__", None), "triton_module": triton.__file__,
            "triton_module_sha256": sha256(triton.__file__),
            "ninetoothed": importlib.metadata.version("ninetoothed"),
            "gpu_name": properties.name, "gcnArchName": properties.gcnArchName,
            "total_memory": properties.total_memory, "tf32": False,
        }
        report["phase"] = "triton_cache_preflight"
        from triton.runtime import cache as runtime_cache
        from round1_triton_cache import WorkspaceCacheManager
        manager = runtime_cache.get_cache_manager(hashlib.sha256(b"ss5-round1-cache-preflight-v3").hexdigest())
        if not isinstance(manager, WorkspaceCacheManager):
            raise RuntimeError("Vendor Triton did not select the workspace cache hook")
        probe = manager.put(b"ss5-round1-cache-preflight-v3", "ss5-cache-probe.bin")
        if manager.get_file("ss5-cache-probe.bin") != probe or Path(probe).read_bytes() != b"ss5-round1-cache-preflight-v3":
            raise RuntimeError("Workspace cache publication preflight failed")
        report["triton_cache"] = {
            "class": f"{type(manager).__module__}.{type(manager).__name__}",
            "cache_dir": manager.cache_dir, "preflight_passed": True,
            "adapter_sha256": sha256(Path(__file__).parent / "round1_triton_cache.py"),
            "vendor_cache_source": runtime_cache.__file__,
            "vendor_cache_source_sha256": sha256(runtime_cache.__file__),
            "publication": "immutable content-addressed files, locked verified index, no rename",
        }
        import numpy as np
        report["environment"]["numpy"] = np.__version__
        if np.__version__ != "1.26.4":
            raise RuntimeError("Expected candidate-only NumPy 1.26.4 overlay")
        report["phase"] = "triton_target_inventory"
        report["environment"]["triton_target"] = str(triton.runtime.driver.active.get_current_target())
        if case.startswith(("smoke_", "nt_")):
            report["phase"] = "install_llvm_gep_compatibility"
            from round1_llvm_compat import install_legacy_gep_adapter
            report["compiler_compatibility"] = install_legacy_gep_adapter(destination, case)
        report["phase"] = "operator_setup"
        if case.startswith("smoke_"):
            smoke_case(torch, case, settings, report)
        elif "adaln" in case or "gate" in case:
            fusion_case(torch, case, settings, report)
        else:
            attention_case(torch, case, settings, destination, report)
        report["completed"] = True
        report["phase"] = "completed"
        if not report["validation_passed"]:
            exit_code = 3
    except BaseException:
        report["error"] = traceback.format_exc()
        exit_code = 2
    write_json(destination / f"{case}.json", report)
    print(json.dumps({"case": case, "completed": report["completed"],
                      "validation_passed": report["validation_passed"],
                      "median_ms": report.get("benchmark", {}).get("median_ms")}), flush=True)
    if report.get("error"):
        print(report["error"], flush=True)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
