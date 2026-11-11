"""Complete device work at the Torch/InfiniLM call boundary."""

import functools


def install_forward_fence(engine_class, synchronize, devices):
    original = engine_class.forward_raw
    if getattr(original, "_ss5_forward_fence", False):
        raise ValueError("Forward fence already installed")
    devices = tuple(devices)
    record = {"devices": list(devices), "calls": 0,
              "pre_synchronizations": 0, "post_synchronizations": 0}

    @functools.wraps(original)
    def fenced(self, *args, **kwargs):
        # Torch-created inputs must be ready before native workers consume them.
        for device in devices:
            synchronize(device)
            record["pre_synchronizations"] += 1
        result = original(self, *args, **kwargs)
        # Complete native writes before any output bridge or buffer reuse.
        for device in devices:
            synchronize(device)
            record["post_synchronizations"] += 1
        record["calls"] += 1
        return result

    fenced._ss5_forward_fence = True
    engine_class.forward_raw = fenced
    return record
