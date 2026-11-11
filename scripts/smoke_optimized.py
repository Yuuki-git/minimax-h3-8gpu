"""Exercise the rebuilt native operators on every gfx936 using Core streams."""
import ctypes as ct
import json
import os


def main():
    import torch
    import infinicore as core
    from infinicore import _ss5_ninetoothed as audit
    from infinicore.lib import _infinicore
    from infinicore.tensor import Tensor

    if os.environ['SS5_NT_ATTN'] + os.environ['SS5_NT_ADALN'] != '11':
        raise ValueError('Run the public-entry smoke with both switches enabled')
    if torch.cuda.device_count() != 8:
        raise ValueError('Eight visible gfx936 GPUs are required')

    def bridge(tensor):
        result = core.strided_from_blob(tensor.data_ptr(), list(tensor.shape), list(tensor.stride()),
                                       dtype=core.bfloat16, device=core.device('cuda', tensor.device.index))
        if list(result.stride()) != list(tensor.stride()):
            raise ValueError('Core tensor lost the physical strides')
        return result

    before = audit.snapshot()
    checks = []
    for rank in range(8):
        if torch.cuda.get_device_properties(rank).gcnArchName.split(':')[0] != 'gfx936':
            raise ValueError(f'Unexpected architecture on GPU {rank}')
        torch.cuda.set_device(rank)
        core.set_device(core.device('cuda', rank))
        stream = core.get_stream()
        if isinstance(stream, int):
            pointer = stream
        else:
            get_name = ct.pythonapi.PyCapsule_GetName
            get_name.argtypes, get_name.restype = (ct.py_object,), ct.c_char_p
            get_pointer = ct.pythonapi.PyCapsule_GetPointer
            get_pointer.argtypes, get_pointer.restype = (ct.py_object, ct.c_char_p), ct.c_void_p
            pointer = get_pointer(stream, get_name(stream))
        if not pointer or pointer == torch.cuda.default_stream(rank).cuda_stream:
            raise ValueError('Core stream is not a non-default device stream')
        external = torch.cuda.ExternalStream(pointer, device=rank)
        x = torch.randn((33, 5376), dtype=torch.bfloat16, device=rank)
        modulation = torch.randn((33, 6 * 5376), dtype=x.dtype, device=rank)
        scale, shift, gate = (modulation[:, i * 5376:(i + 1) * 5376] for i in range(3))
        branch = torch.randn_like(x)
        adaln, residual = torch.empty_like(x), torch.empty_like(x)
        q = torch.zeros((1, 27570, 7, 128), dtype=x.dtype, device=rank)
        k = torch.zeros_like(q)
        packed = torch.ones((1, 27570, 3, 7, 128), dtype=x.dtype, device=rank)
        v = packed[:, :, 2]
        attention = torch.empty_like(q)
        torch.cuda.synchronize(rank)
        with torch.cuda.stream(external):
            nx, ns, nb, ng, nbranch = map(bridge, (x, scale, shift, gate, branch))
            native_adaln = _infinicore._ss5_adaln(nx._underlying, ns._underlying, nb._underlying)
            native_gate = _infinicore._ss5_gated_residual(nx._underlying, nbranch._underlying, ng._underlying)
            if native_adaln is None or native_gate is None:
                raise ValueError('Fused dispatch did not accept the modulation layout')
            bridge(adaln).copy_(Tensor(native_adaln))
            bridge(residual).copy_(Tensor(native_gate))
            core.mha(bridge(q), bridge(k), bridge(v), scale=128 ** -0.5, out=bridge(attention))
            expected_adaln = (x + x * scale) + shift
            expected_gate = x + branch * gate
        core.sync_stream()
        torch.testing.assert_close(adaln, expected_adaln, rtol=0, atol=0)
        torch.testing.assert_close(residual, expected_gate, rtol=0, atol=0)
        torch.testing.assert_close(attention, torch.ones_like(attention), rtol=0, atol=0)
        checks.append({'device': rank, 'core_stream': pointer, 'passed': True})
        print(f'SS5_OPT_SMOKE_DEVICE={rank} PASS', flush=True)
        del x, modulation, scale, shift, gate, branch, adaln, residual, q, k, packed, v, attention
    after = audit.snapshot()
    delta = {k: after['calls'][k] - before['calls'][k] for k in before['calls']}
    if delta != {'attention': 8, 'adaln': 8, 'gate': 8}:
        raise ValueError(f'Unexpected host submission counts: {delta}')
    print(json.dumps({'passed': True, 'devices': checks, 'host_dispatch_calls': delta}), flush=True)


if __name__ == '__main__':
    main()
