#!/usr/bin/env python3
"""Generate the three HIP libraries using the accepted NineToothed AOT path."""
import argparse
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil

from run_round1 import sha256, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', choices=('adaln', 'gate', 'attention'), required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--prefix', type=Path, required=True)
    args = parser.parse_args()
    destination = args.output.resolve(strict=True)
    # compile_child validates that the generated Triton source is below this case.
    # Configure these before importing NineToothed or initializing the vendor driver.
    for variable, relative in {
        'NINETOOTHED_CACHE_DIR': 'cache/ninetoothed', 'TRITON_CACHE_DIR': 'cache/triton',
        'TRITON_DUMP_DIR': 'cache/triton-dump', 'TRITON_OVERRIDE_DIR': 'cache/triton-override',
        'SS5_ROUND1_CACHE_SCOPE': 'cache', 'TMPDIR': 'tmp', 'TMP': 'tmp', 'TEMP': 'tmp',
    }.items():
        path = destination / relative
        path.mkdir(parents=True, exist_ok=True)
        os.environ[variable] = str(path)
    import ninetoothed as nt
    from round2_aot_worker import select_device
    from round3_hip_aot import build_aot
    from round1_llvm_compat import install_legacy_gep_adapter
    import round1_kernels as kernels

    aot = importlib.import_module('ninetoothed.aot')
    expected = 'fc86e4aa739ac6bc61d3e638a18ac960ef9a48cc1d5d876716bd4645c5d5efb3'
    if sha256(aot.__file__) != expected:
        raise ValueError('Locked NineToothed HIP AOT source differs')
    write_json(destination / 'aot-patch.json', {'after_sha256': expected})
    write_json(destination / 'settings.json', {'device': 0})
    report = {'case': args.case, 'completed': False,
              'environment': {n: importlib.metadata.version(n) for n in ('torch', 'triton', 'numpy', 'ninetoothed')}}
    select_device(destination, report)
    report['compiler_compatibility'] = install_legacy_gep_adapter(destination, args.case)
    if args.case == 'attention':
        from round4_attention import definitions, compile_attention
        definition, _ = definitions()
        library = compile_attention(destination, definition, report)
        name = 'ss5_s4_attention'
    else:
        name = 'ss5_s3_' + args.case
        names = ('normalized', 'scale', 'shift', 'output') if args.case == 'adaln' else ('residual', 'branch', 'gate', 'output')
        definition = (kernels.elementwise_arrangement, getattr(kernels, args.case + '_application'),
                      tuple(nt.Tensor(shape=(None, 5376), dtype=nt.bfloat16, name=n) for n in names))
        build_aot(destination, 'aot_' + args.case, definition, names, 5376, None, report)
        library = destination / 'aot' / ('aot_' + args.case) / (name + '.so')
    prefix = args.prefix.resolve(strict=True)
    (prefix / 'lib').mkdir(exist_ok=True)
    (prefix / 'include').mkdir(exist_ok=True)
    target = prefix / 'lib' / ('lib' + name + '.so')
    shutil.copyfile(library, target)
    generated = library.parent
    root = Path(__file__).resolve().parents[1]
    lock = json.loads((root / 'source-lock.json').read_bytes())
    core = prefix.parent / 'src/InfiniCore/src/infinicore/ops/ss5_ninetoothed/generated'
    for header in (name + '.h', 'ninetoothed.h'):
        actual = (generated / header).read_bytes()
        expected_header = lock['sources']['InfiniCore']['files'][str((core / header).relative_to(prefix.parent / 'src/InfiniCore')).replace('\\', '/')]
        if sha256(generated / header) != expected_header:
            raise ValueError(f'Generated ABI differs from accepted Core header: {header}')
        (prefix / 'include' / header).write_bytes(actual)
    report.update(completed=True, installed_library=str(target), installed_sha256=sha256(target))
    write_json(destination / 'build.json', report)
    print(json.dumps({'case': args.case, 'library': str(target), 'sha256': sha256(target)}), flush=True)


if __name__ == '__main__':
    main()
