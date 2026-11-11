"""V5 SDPA schedule scan with model V stride and matched FP32 references."""

import dataclasses
import inspect
import json
import math
from pathlib import Path
import time
from unittest.mock import patch

from round1_worker import (DIM, HEADS, SEQUENCE, attention_inputs, core_bridge, error_metrics,
                           fp32_attention_reference, generated_source, input_digest, load_core,
                           load_module, synchronized_benchmark, tensor_info)
from round3_metadata import kernel_resources
from run_round1 import sha256


SCHEDULES = (
    (64, 64, 4, 1), (128, 64, 4, 1), (128, 64, 8, 1), (128, 32, 4, 1),
    (128, 128, 8, 1), (64, 64, 4, 2), (128, 64, 4, 2), (256, 64, 8, 1),
)
DISTRIBUTIONS = ("normal", "sharp", "tail")
MAX_RELATIVE_RMSE = 0.01


def inputs_for(torch, device, distribution):
    q, k, initial_v = attention_inputs(torch, device)
    # Match the last component of the QKV projection, including the offset.
    packed = torch.empty((1, SEQUENCE, 3 * HEADS, DIM), dtype=q.dtype, device=device)
    v = packed[:, :, 2 * HEADS:, :]
    v.copy_(initial_v)
    if v.stride() != (SEQUENCE * 3 * HEADS * DIM, 3 * HEADS * DIM, DIM, 1):
        raise ValueError("Expected H3's packed V view")
    if distribution == "sharp":
        q.mul_(8)
    elif distribution == "tail":
        q.zero_()
        k.zero_()
        q[..., 0] = 1
        k[..., 0] = -8 * math.sqrt(DIM)
        k[:, -50:, :3, 0] = 10 * math.sqrt(DIM)
        v.fill_(1)
        v[:, :, :3, :].zero_()
        v[:, -50:, :3, :].fill_(1.25)
    return q, k, v


def jit_evidence(torch, function, device, kernel_name, directory):
    from triton.runtime.jit import JITFunction

    original = JITFunction.run
    compiled = []

    def run(kernel, *args, **kwargs):
        result = original(kernel, *args, **kwargs)
        if getattr(getattr(result, "metadata", None), "name", None) == kernel_name:
            compiled.append(result)
        return result

    started = time.perf_counter()
    with patch.object(JITFunction, "run", run):
        function()
        torch.cuda.synchronize(device)
    elapsed = (time.perf_counter() - started) * 1000
    if len(compiled) != 1 or compiled[0] is None:
        raise RuntimeError("Unable to capture the actual SDPA CompiledKernel")
    artifact = compiled[0]
    directory.mkdir(parents=True)
    binary = directory / "attention.hsaco"
    code = artifact.asm["hsaco"]
    binary.write_bytes(code)
    metadata = json.loads(json.dumps(artifact.metadata._asdict(), default=str))
    return {"compile_and_first_launch_ms": elapsed, "hsaco_sha256": sha256(binary), "metadata": metadata,
            "elf_resources": kernel_resources(code, artifact.metadata.name)}


def attention_probe(torch, case, settings, destination, report):
    _, engine, distribution, *config = case.split("_")
    if distribution not in DISTRIBUTIONS:
        raise ValueError(f"Unknown attention distribution {distribution}")
    device = torch.device("cuda", settings["device"])
    q, k, v = inputs_for(torch, device, distribution)
    report["inputs"] = [tensor_info(x) for x in (q, k, v)]
    report["distribution"] = distribution
    report["input_sha256"] = input_digest((q, k, v))
    torch.cuda.synchronize(device)
    reference_path = destination / "large" / f"reference-{distribution}.pt"
    if engine == "reference":
        report["phase"] = "full_fp32_reference"
        started = time.perf_counter()
        reference = fp32_attention_reference(torch, q, k, v)
        report["compute_seconds"] = time.perf_counter() - started
        reference_path.parent.mkdir(exist_ok=True)
        torch.save({"output": reference, "input_sha256": report["input_sha256"]}, reference_path)
        report["reference_sha256"] = sha256(reference_path)
        return True

    output = torch.empty_like(q)
    scale = 1 / math.sqrt(DIM)
    if engine == "core":
        core = load_core(torch, settings, report)
        cq, ck, cv, co = (core_bridge(core, x) for x in (q, k, v, output))
        function = lambda: core.mha(cq, ck, cv, scale=scale, is_causal=False, out=co)
    elif engine == "nt":
        import ninetoothed as nt
        from triton.backends.amd.compiler import HIPOptions

        index = int(config[0])
        block_m, block_n, warps, stages = SCHEDULES[index]
        report["schedule"] = {"index": index, "block_m": block_m, "block_n": block_n,
                              "num_warps": warps, "num_stages": stages}
        report["compiler_options"] = {"vendor_hip_options": [field.name for field in dataclasses.fields(HIPOptions)],
                                      "ninetoothed_make_parameters": list(inspect.signature(nt.make).parameters)}
        pinned = load_module("ss5_s3_pinned_sdpa", Path(__file__).with_name("pinned_sdpa_v5.py"))
        definition = pinned.premake(with_kv_cache=False, emb_dim=DIM, is_causal=False,
                                    with_attn_mask=False, causal_variant=pinned.CausalVariant.UPPER_LEFT,
                                    dtype=nt.bfloat16, block_size_m=block_m, block_size_n=block_n)
        kernel_name = f"ss5_s3_sdpa_{index:02d}"
        kernel = nt.make(*definition, kernel_name=kernel_name, num_warps=warps, num_stages=stages, max_num_configs=1)
        generated_source(kernel, report)
        qh, kh, vh, oh = (x.permute(0, 2, 1, 3) for x in (q, k, v, output))
        meta_mask = torch.empty((1, HEADS, SEQUENCE, SEQUENCE), device="meta")
        report["kernel_inputs"] = [tensor_info(x) for x in (qh, kh, vh)]
        function = lambda: kernel(qh, kh, vh, meta_mask, False, scale, oh, False, pinned.CausalVariant.UPPER_LEFT)
        report["phase"] = "jit_compile_and_capture_resources"
        report["resource_evidence"] = jit_evidence(torch, function, device, kernel_name, destination / "attention" / case)
    else:
        raise ValueError(f"Unknown attention implementation {engine}")

    report["phase"] = "synchronized_benchmark"
    report["benchmark"] = synchronized_benchmark(torch, function, settings)
    report["phase"] = "numerical_validation"
    reference = torch.load(reference_path, map_location="cpu", weights_only=True)
    if reference["input_sha256"] != report["input_sha256"]:
        raise ValueError("Reference and benchmark input hashes differ")
    metrics = error_metrics(output, reference["output"])
    report["error_vs_full_fp32"] = metrics
    finite = metrics["nonfinite_actual"] == metrics["nonfinite_reference"] == 0 and metrics["relative_rmse"] is not None
    if engine == "core":
        return finite
    core = json.loads((destination / f"attn_core_{distribution}.json").read_text(encoding="utf-8"))
    if core["input_sha256"] != report["input_sha256"] or not core["validation_passed"]:
        raise ValueError("Missing valid Core reference on identical inputs")
    core_error = core["error_vs_full_fp32"]["relative_rmse"]
    report["accuracy_gate"] = {"max_relative_rmse": MAX_RELATIVE_RMSE, "matched_core_relative_rmse": core_error}
    return finite and metrics["relative_rmse"] <= min(MAX_RELATIVE_RMSE, core_error)
