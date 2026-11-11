"""Use unchanged official FL2VA generation and comparison at the locked sources."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from run_fl2va import COMPARE_SHA256, load_script, worker


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('smoke', 'reference', 'native', 'compare'))
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--first-frame', type=Path, default=Path(__file__).resolve().parents[1] / 'assets/ss5_fl2va_template_first_frame.png')
    parser.add_argument('--model-path', type=Path, default=Path(os.environ.get('IFAB_SHARED_MODELS', '/public/home/aclsylqidf/shared')) / 'MiniMax-H3/diffusers')
    parser.add_argument('--reference-video', type=Path)
    parser.add_argument('--native-video', type=Path)
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    return parser.parse_args()


def run_child(args, work, output):
    started = time.perf_counter()
    if args.mode == 'smoke':
        from smoke_optimized import main
        main()
    elif args.mode == 'compare':
        reference = args.reference_video.resolve(strict=True)
        native = args.native_video.resolve(strict=True)
        path = work / 'src/InfiniLM/examples/minimax_h3_compare.py'
        module = load_script(path, 'h3_official_compare', COMPARE_SHA256)
        sys.argv = [str(path), '--tasks=fl2va', f'--model-path={args.model_path}', f'--output-dir={output}',
                    f'--reference-fl2va-video={reference}', f'--infinilm-fl2va-video={native}']
        if module.main() != 0:
            raise RuntimeError('Official comparer did not complete')
    else:
        args.worker = args.mode
        worker(args, work)
        if args.mode == 'native':
            from infinicore._ss5_ninetoothed import snapshot
            counts = snapshot()
            expected = {'attention': 19600 if os.environ['SS5_NT_ATTN'] == '1' else 0,
                        'adaln': 39592 if os.environ['SS5_NT_ADALN'] == '1' else 0,
                        'gate': 39200 if os.environ['SS5_NT_ADALN'] == '1' else 0}
            if counts['calls'] != expected:
                raise RuntimeError(f'Host dispatch counts differ: {counts}; expected {expected}')
            (output / 'native-dispatch.json').write_text(json.dumps(counts, indent=2) + '\n')
    elapsed = time.perf_counter() - started
    (output / 'worker-time.json').write_text(json.dumps({'worker_seconds': elapsed,
        'scope': 'includes Python imports and validation; not generation.pipeline_total'}) + '\n')


def main():
    args = parse_args()
    work = Path(os.environ['SS5_OPT_WORK']).resolve(strict=True)
    workspace = Path(os.environ['IFAB_WORKSPACE_ROOT']).resolve(strict=True)
    if not (work / 'build-manifest.json').is_file():
        raise ValueError('Run build_optimized.sh first')
    if args.mode == 'compare' and (not args.reference_video or not args.native_video):
        raise ValueError('compare requires --reference-video and --native-video')
    output = (args.output_dir or work / 'results' / args.mode).resolve()
    if not output.is_relative_to(workspace.parent) or output == workspace.parent:
        raise ValueError('Outputs must stay within the web workspace or its parent')
    args.first_frame = args.first_frame.resolve(strict=True)
    if args.mode != 'smoke':
        args.model_path = args.model_path.resolve(strict=True)
    if args.worker:
        args.output_dir = output
        run_child(args, work, output)
        return
    output.mkdir(parents=True, exist_ok=False)
    (output / 'arguments.json').write_text(json.dumps({'mode': args.mode,
        'source_lock': json.loads((work / 'source-lock.json').read_bytes()),
        'switches': os.environ['SS5_NT_ATTN'] + os.environ['SS5_NT_ADALN'],
        'first_frame_sha256': hashlib.sha256(args.first_frame.read_bytes()).hexdigest(),
        'model_path': str(args.model_path), 'prompt': 'A dog falls asleep', 'seed': 42, 'steps': 50,
        'num_frames': 124, 'gpus': list(range(8))}, indent=2) + '\n')
    command = [sys.executable, '-B', '-u', str(Path(__file__).resolve()), args.mode, '--worker',
               '--output-dir', str(output), '--first-frame', str(args.first_frame), '--model-path', str(args.model_path)]
    if args.reference_video:
        command.extend(['--reference-video', str(args.reference_video.resolve(strict=True))])
    if args.native_video:
        command.extend(['--native-video', str(args.native_video.resolve(strict=True))])
    with (output / 'run.log').open('x', encoding='utf-8') as log:
        process = subprocess.Popen(command, cwd=work / 'src/InfiniLM', stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, encoding='utf-8', errors='replace')
        try:
            for line in process.stdout:
                print(line, end='', flush=True)
                log.write(line)
                log.flush()
            status = process.wait()
        except BaseException:
            process.terminate()
            process.wait()
            raise
        finally:
            process.stdout.close()
    if status:
        raise SystemExit(f'{args.mode} failed with status {status}; log: {output / "run.log"}')
    print(f'SS5_OPT_{args.mode.upper()}_RESULT={output}')


if __name__ == '__main__':
    main()
