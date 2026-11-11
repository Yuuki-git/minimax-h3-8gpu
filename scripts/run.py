"""Use unchanged official FL2VA generation and comparison at the locked sources."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import importlib.util
from types import SimpleNamespace
GENERATOR_SHA256 = "ca54a18a73a7d760d644ca4eb2c3b20678f88dfdbcfa26a62dcd3de617b31af4"
COMPARE_SHA256 = "e143c28e4b3acb885662e0aa238aa00ca17cbcc82077f465f2ce57d0f1225590"

def load_script(path, name, expected_hash):
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected_hash:
        raise ValueError(f"官方脚本与固定提交不符：{path}")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module

def official_worker(args, work):
    examples = work / "src/InfiniLM/examples"
    if args.worker == "compare":
        path = examples / "minimax_h3_compare.py"
        module = load_script(path, "h3_official_compare", COMPARE_SHA256)
        sys.argv = [str(path), "--tasks=fl2va", f"--model-path={args.model_path}",
                    f"--output-dir={args.output_dir}",
                    f"--reference-fl2va-video={args.output_dir / 'reference_fl2va.mp4'}",
                    f"--infinilm-fl2va-video={args.output_dir / 'infinilm_fl2va.mp4'}"]
        status = module.main()
        if status != 0:
            raise RuntimeError(f"官方比较未完成，退出码 {status}")
        return

    import torch

    if torch.cuda.device_count() != 8 or not all(
        torch.cuda.get_device_properties(i).gcnArchName.split(":")[0] == "gfx936"
        for i in range(8)
    ):
        raise ValueError("需要完整的 8 卡 gfx936 会话")
    print(f"[environment] torch={torch.__version__} hip={torch.version.hip}", flush=True)
    counts = None
    if args.worker == "native":
        from infinilm.infer_engine import InferEngine

        from forward_fence import install_forward_fence

        counts = install_forward_fence(InferEngine, torch.cuda.synchronize, range(8))
        print("[native] forward_raw: synchronize all 8 devices before and after", flush=True)
    path = examples / "minimax_h3.py"
    module = load_script(path, "h3_official_generator", GENERATOR_SHA256)
    filename = "reference_fl2va.mp4" if args.worker == "reference" else "infinilm_fl2va.mp4"
    sys.argv = [str(path), "fl2va", f"--model-path={args.model_path}",
                "--text=A dog falls asleep", "--gpus=0,1,2,3,4,5,6,7", "--steps=50",
                "--seed=42", "--num-frames=124", f"--first-frame={args.first_frame}",
                f"--output={args.output_dir / filename}"]
    if args.worker == "reference":
        sys.argv.append("--legacy")
    module.main()
    if counts is not None:
        print(f"[native] forward boundary counts: {json.dumps(counts)}", flush=True)
        if (counts["calls"], counts["pre_synchronizations"], counts["post_synchronizations"]) != (50, 400, 400):
            raise RuntimeError("原生调用或同步次数与固定 50 步配置不符")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('reference', 'native', 'compare'))
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--first-frame', type=Path, default=Path(__file__).resolve().parents[1] / 'assets/ss5_fl2va_template_first_frame.png')
    parser.add_argument('--model-path', type=Path, default=Path(os.environ.get('IFAB_SHARED_MODELS', '/public/home/aclsylqidf/shared')) / 'MiniMax-H3/diffusers')
    parser.add_argument('--reference-video', type=Path)
    parser.add_argument('--native-video', type=Path)
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    return parser.parse_args()


def run_child(args, work, output):
    started = time.perf_counter()
    if args.mode == 'compare':
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
        official_worker(args, work)
    elapsed = time.perf_counter() - started
    (output / 'worker-time.json').write_text(json.dumps({'worker_seconds': elapsed,
        'scope': 'includes Python imports and validation; not generation.pipeline_total'}) + '\n')


def main():
    args = parse_args()
    work = Path(os.environ['SS5_OPT_WORK']).resolve(strict=True)
    workspace = Path(os.environ['IFAB_WORKSPACE_ROOT']).resolve(strict=True)
    if not (work / '.built').is_file():
        raise ValueError('Run build.sh first')
    if args.mode == 'compare' and (not args.reference_video or not args.native_video):
        raise ValueError('compare requires --reference-video and --native-video')
    output = (args.output_dir or work / 'results' / args.mode).resolve()
    if not output.is_relative_to(workspace.parent) or output == workspace.parent:
        raise ValueError('Outputs must stay within the web workspace or its parent')
    args.first_frame = args.first_frame.resolve(strict=True)
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
