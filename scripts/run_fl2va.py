#!/usr/bin/env python3
"""Run official FL2VA generation with the eight-device forward boundary fence."""

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import uuid


CORE_COMMIT = "f32e2fbb2a9b2ca912b5711f4f3836c403196a74"
LM_COMMIT = "65deb91f840ebaef752d028fad2f3217ce0950f4"
GENERATOR_SHA256 = "ca54a18a73a7d760d644ca4eb2c3b20678f88dfdbcfa26a62dcd3de617b31af4"
COMPARE_SHA256 = "e143c28e4b3acb885662e0aa238aa00ca17cbcc82077f465f2ce57d0f1225590"
REPO_ROOT = Path(__file__).resolve().parents[1]


def load_script(path, name, expected_hash):
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected_hash:
        raise ValueError(f"官方脚本与固定提交不符：{path}")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    shared = Path(os.environ.get("IFAB_SHARED_MODELS", "/public/home/aclsylqidf/shared"))
    parser.add_argument("--first-frame", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, default=Path(
        os.environ.get("SS5_FL2VA_MODEL", str(shared / "MiniMax-H3/diffusers"))))
    parser.add_argument("--reference-video", type=Path, help="复用同配置 PyTorch 成片，省略参考生成")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--worker", choices=("reference", "native", "compare"), help=argparse.SUPPRESS)
    return parser.parse_args()


def worker(args, work):
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

        sys.path.insert(0, str(REPO_ROOT / "development"))
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


def run_logged(command, cwd, destination):
    with destination.open("x", encoding="utf-8") as log:
        process = subprocess.Popen(command, cwd=cwd, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")
        try:
            for line in process.stdout:
                print(line, end="", flush=True)
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
        raise RuntimeError(f"运行失败，退出码 {status}，日志：{destination}")


def main():
    args = parse_args()
    workspace = Path(os.environ["IFAB_WORKSPACE_ROOT"]).resolve(strict=True)
    afs = Path(os.environ["IFAB_AFS_HOME"]).resolve(strict=True)
    work = workspace / "ss5-v3-work"
    args.first_frame = args.first_frame.resolve(strict=True)
    args.model_path = args.model_path.resolve(strict=True)
    if args.worker:
        if args.output_dir is None:
            raise ValueError("工作进程缺少输出目录")
        args.output_dir = args.output_dir.resolve(strict=True)
        worker(args, work)
        return
    if not args.first_frame.is_file() or not args.model_path.is_dir():
        raise ValueError("首帧必须为文件，模型路径必须为 Diffusers 权重目录")
    reference = args.reference_video.resolve(strict=True) if args.reference_video else None
    if reference is not None and not reference.is_file():
        raise ValueError("参考视频不是文件")
    destination = (args.output_dir or afs / f"ss5-fl2va-{uuid.uuid4().hex[:12]}").resolve()
    if not (destination.is_relative_to(afs) or destination.is_relative_to(workspace)):
        raise ValueError("输出目录必须位于个人 AFS 或本次会话工作区内")
    destination.mkdir(mode=0o770, parents=True, exist_ok=False)
    print(f"RESULT_DIR={destination}", flush=True)
    record = {"source_commits": {"InfiniCore": CORE_COMMIT, "InfiniLM": LM_COMMIT},
              "model_path": str(args.model_path), "first_frame": str(args.first_frame),
              "first_frame_sha256": hashlib.sha256(args.first_frame.read_bytes()).hexdigest(),
              "prompt": "A dog falls asleep", "seed": 42, "steps": 50, "num_frames": 124,
              "gpus": list(range(8)), "reference_video_reused": str(reference) if reference else None}
    (destination / "arguments.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    if reference is not None:
        shutil.copyfile(reference, destination / "reference_fl2va.mp4")
    modes = ["native", "compare"] if reference is not None else ["reference", "native", "compare"]
    for mode in modes:
        command = [sys.executable, "-B", "-u", str(Path(__file__).resolve()),
                   "--worker", mode, "--first-frame", str(args.first_frame),
                   "--model-path", str(args.model_path), "--output-dir", str(destination)]
        log_name = {"reference": "reference_fl2va.log", "native": "infinilm_fl2va.log", "compare": "run.log"}[mode]
        run_logged(command, work / "src/InfiniLM", destination / log_name)
    print(f"NATIVE_VIDEO={destination / 'infinilm_fl2va.mp4'}", flush=True)
    print(f"COMPARE_LOG={destination / 'run.log'}", flush=True)


if __name__ == "__main__":
    os.umask(0o007)
    try:
        main()
    except (KeyError, OSError, RuntimeError, ValueError) as exc:
        raise SystemExit(f"FL2VA_FAILED: {exc}") from exc
