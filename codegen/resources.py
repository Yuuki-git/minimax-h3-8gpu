"""Read the resource note carried by a gfx936 ELF code object."""

import struct


def unpack_msgpack(data):
    offset = 0

    def take(size):
        nonlocal offset
        result = data[offset:offset + size]
        if len(result) != size:
            raise ValueError("Truncated AMDGPU metadata")
        offset += size
        return result

    def integer(size, signed=False):
        return int.from_bytes(take(size), "big", signed=signed)

    def read(depth=0):
        if depth > 24:
            raise ValueError("AMDGPU metadata nesting is too deep")
        tag = integer(1)
        if tag < 0x80:
            return tag
        if tag >= 0xE0:
            return tag - 256
        if 0xA0 <= tag <= 0xBF:
            return take(tag - 0xA0).decode("utf-8")
        if 0x90 <= tag <= 0x9F or tag in (0xDC, 0xDD):
            count = tag - 0x90 if tag < 0xA0 else integer(2 if tag == 0xDC else 4)
            return [read(depth + 1) for _ in range(count)]
        if 0x80 <= tag <= 0x8F or tag in (0xDE, 0xDF):
            count = tag - 0x80 if tag < 0x90 else integer(2 if tag == 0xDE else 4)
            return {read(depth + 1): read(depth + 1) for _ in range(count)}
        if 0xCC <= tag <= 0xD3:
            return integer(1 << ((tag - 0xCC) % 4), signed=tag >= 0xD0)
        if 0xD9 <= tag <= 0xDB:
            return take(integer(1 << (tag - 0xD9))).decode("utf-8")
        if tag in (0xC0, 0xC2, 0xC3):
            return {0xC0: None, 0xC2: False, 0xC3: True}[tag]
        raise ValueError(f"Unsupported AMDGPU MessagePack tag: 0x{tag:02x}")

    value = read()
    if offset != len(data):
        raise ValueError("Trailing bytes in AMDGPU metadata")
    return value


def kernel_resources(code, entry):
    if code[:6] != b"\x7fELF\x02\x01" or struct.unpack_from("<H", code, 18)[0] != 224:
        raise ValueError("Expected a little-endian AMDGPU ELF64 code object")
    sections = struct.unpack_from("<Q", code, 40)[0]
    stride, count = struct.unpack_from("<HH", code, 58)
    if stride < 64 or sections + stride * count > len(code):
        raise ValueError("Invalid AMDGPU ELF section table")
    for index in range(count):
        section = sections + stride * index
        if struct.unpack_from("<I", code, section + 4)[0] != 7:  # SHT_NOTE
            continue
        offset, size = struct.unpack_from("<QQ", code, section + 24)
        end = offset + size
        while offset + 12 <= end:
            name_size, desc_size, note_type = struct.unpack_from("<III", code, offset)
            name_start = offset + 12
            desc_start = name_start + (name_size + 3) // 4 * 4
            offset = desc_start + (desc_size + 3) // 4 * 4
            if offset > end or offset > len(code):
                raise ValueError("Truncated AMDGPU ELF note")
            name = code[name_start:name_start + name_size].rstrip(b"\0")
            if name != b"AMDGPU" or note_type != 32:  # NT_AMDGPU_METADATA
                continue
            note = unpack_msgpack(code[desc_start:desc_start + desc_size])
            matches = [kernel for kernel in note["amdhsa.kernels"] if kernel[".name"] == entry]
            if len(matches) != 1:
                raise ValueError(f"Expected one AMDGPU metadata entry for {entry}")
            return {name: matches[0].get("." + name) for name in (
                "vgpr_count", "vgpr_spill_count", "sgpr_count", "sgpr_spill_count",
                "group_segment_fixed_size", "private_segment_fixed_size", "wavefront_size",
            )}
    raise ValueError("AMDGPU metadata note unavailable")
