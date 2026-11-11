"""NineToothed definitions for the first gfx936 probes, never raw Triton."""

import ninetoothed as nt
import ninetoothed.language as ntl


def elementwise_arrangement(*tensors):
    # Flatten logical indices, preserving each source's physical strides.
    return tuple(tensor.flatten().tile((1024,)) for tensor in tensors)


def add_application(left, right, output):
    output = left + right


def adaln_application(normalized, scale, shift, output):
    # Fixed LM: add(add(x, mul(x, scale)), shift). BF16 at EVERY boundary.
    product = ntl.cast(ntl.cast(normalized, ntl.float32) * ntl.cast(scale, ntl.float32), ntl.bfloat16)
    partial = ntl.cast(ntl.cast(normalized, ntl.float32) + ntl.cast(product, ntl.float32), ntl.bfloat16)
    output = ntl.cast(ntl.cast(partial, ntl.float32) + ntl.cast(shift, ntl.float32), ntl.bfloat16)


def gate_application(residual, branch, gate, output):
    product = ntl.cast(ntl.cast(gate, ntl.float32) * ntl.cast(branch, ntl.float32), ntl.bfloat16)
    output = ntl.cast(ntl.cast(residual, ntl.float32) + ntl.cast(product, ntl.float32), ntl.bfloat16)


def dot_arrangement(left, right, output):
    # Arrange-and-apply GEMM, following the official NineToothed example.
    arranged_output = output.tile((32, 64))
    arranged_left = left.tile((32, 32)).tile((1, -1))
    arranged_left = arranged_left.expand((-1, arranged_output.shape[1]))
    arranged_left.dtype = arranged_left.dtype.squeeze(0)
    arranged_right = right.tile((32, 64)).tile((-1, 1))
    arranged_right = arranged_right.expand((arranged_output.shape[0], -1))
    arranged_right.dtype = arranged_right.dtype.squeeze(1)
    return arranged_left, arranged_right, arranged_output


def dot_application(left, right, output):
    accumulator = ntl.zeros(output.shape, dtype=ntl.float32)
    for index in range(left.shape[0]):
        accumulator += ntl.dot(left[index], right[index])
    output = accumulator


def make_elementwise(kind):
    ndim, dtype, count, application = {
        "add": (1, nt.float32, 3, add_application),
        "adaln": (2, nt.bfloat16, 4, adaln_application),
        "gate": (2, nt.bfloat16, 4, gate_application),
    }[kind]
    return nt.make(elementwise_arrangement, application,
                   tuple(nt.Tensor(ndim, dtype=dtype) for _ in range(count)),
                   kernel_name=f"ss5_r1_{kind}", num_warps=4, num_stages=1,
                   max_num_configs=1)


def make_dot():
    return nt.make(dot_arrangement, dot_application,
                   tuple(nt.Tensor(2, dtype=nt.bfloat16) for _ in range(3)),
                   kernel_name="ss5_r1_bf16_dot", num_warps=4, num_stages=1,
                   max_num_configs=1)


def make_attention(pinned_sdpa):
    # Explicit, bounded schedule; no whole-model tuning in a 60-minute session.
    arrangement, application, tensors = pinned_sdpa.premake(
        with_kv_cache=False, emb_dim=128, is_causal=False,
        with_attn_mask=False, causal_variant=pinned_sdpa.CausalVariant.UPPER_LEFT,
        dtype=nt.bfloat16, block_size_m=64, block_size_n=64,
    )
    return nt.make(arrangement, application, tensors,
                   kernel_name="ss5_r1_ntops_sdpa", num_warps=4, num_stages=1,
                   max_num_configs=1)
