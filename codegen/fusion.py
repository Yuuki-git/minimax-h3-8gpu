"""NineToothed fusion definitions with the accepted BF16 rounding boundaries."""
import ninetoothed as nt
import ninetoothed.language as ntl

def elementwise_arrangement(*tensors):
    # Flatten logical indices, preserving each source's physical strides.
    return tuple(tensor.flatten().tile((1024,)) for tensor in tensors)


def adaln_application(normalized, scale, shift, output):
    # Fixed LM: add(add(x, mul(x, scale)), shift). BF16 at EVERY boundary.
    product = ntl.cast(ntl.cast(normalized, ntl.float32) * ntl.cast(scale, ntl.float32), ntl.bfloat16)
    partial = ntl.cast(ntl.cast(normalized, ntl.float32) + ntl.cast(product, ntl.float32), ntl.bfloat16)
    output = ntl.cast(ntl.cast(partial, ntl.float32) + ntl.cast(shift, ntl.float32), ntl.bfloat16)


def gate_application(residual, branch, gate, output):
    product = ntl.cast(ntl.cast(gate, ntl.float32) * ntl.cast(branch, ntl.float32), ntl.bfloat16)
    output = ntl.cast(ntl.cast(residual, ntl.float32) + ntl.cast(product, ntl.float32), ntl.bfloat16)
