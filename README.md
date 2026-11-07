# MiniMax-H3 八卡 FL2VA 适配

在 8 张海光 gfx936 上使用 InfiniLM / InfiniCore 运行 MiniMax-H3，生成有声视频，并与 PyTorch 参考结果进行官方 FL2VA 比较。

适配包含上游海光稠密 MHA 的构建配置，以及 Torch 与原生推理交接时的八卡同步。Core/LM 数学源码和官方评测脚本保持原样。

## 环境

- Linux x86_64，8 × gfx936，DTK 位于 `/opt/dtk`；海光 PyTorch 2.9.0 / HIP 6.3.26093。
- Python 3.11 环境，按主办方指南安装模型依赖及匹配的 torchvision 0.24.0、厂商 Triton 和 FlashAttention。Python 依赖版本见 [requirements.txt](requirements.txt)。
- xmake **3.0.5+HEAD.f20295f**，以及 Git、tar、unzip、gzip、make、CMake、g++、readelf、flock 和 diff。

当前脚本使用 InfiniFabric 的目录约定：`IFAB_AFS_HOME` 为持久存储目录，`IFAB_WORKSPACE_ROOT` 为本次作业工作目录；Python 环境位于 `$IFAB_AFS_HOME/venvs/minimax-h3`，xmake 位于 `$IFAB_AFS_HOME/tools/bin/xmake`。其他平台需相应适配目录和运行库路径。

另行准备 MiniMax-H3 的 Diffusers 格式权重和主办方统一首帧。下面的 `/path/to/...` 均替换为自己的实际路径。

## 复现

在可联网的 Linux 机器或 WSL 的 Linux 文件系统中获取代码并准备依赖：

```bash
git clone --branch main --depth 1 https://github.com/Yuuki-git/minimax-h3-8gpu.git
cd minimax-h3-8gpu
bash scripts/reproduce.sh --prepare-only
```

将仓库代码及生成的 `dist/platform-inputs` 复制到目标机器，保持相对目录结构。在满足上述条件的八卡环境中，从仓库根目录运行：

```bash
bash scripts/reproduce.sh --offline \
  --model-path /path/to/MiniMax-H3/diffusers \
  --first-frame /path/to/first_frame.png
```

脚本按并发 8 构建 Core/LM，再串行生成 PyTorch 参考视频、InfiniLM 原生视频，并调用官方 `minimax_h3_compare.py`。固定样例为 `A dog falls asleep`、seed 42、50 步、124 帧，使用 GPU 0–7。

可按需在运行命令中增加：

| 参数 | 用途 |
| --- | --- |
| `--skip-build` | 复用当前作业已完成的构建 |
| `--skip-build --runtime /path/to/runtime-dir` | 在新作业中恢复已保存的编译产物 |
| `--reference-video /path/to/reference.mp4` | 复用同配置 PyTorch 视频，跳过参考生成 |
| `--output-dir /path/to/new-output-dir` | 指定持久目录或作业目录内尚不存在的输出目录 |

编译产物的保存与恢复使用 [ss5_native_checkpoint_v2.py](scripts/ss5_native_checkpoint_v2.py)，可通过 `python3 scripts/ss5_native_checkpoint_v2.py --help` 查看用法。固定上游提交和构建参数见 [源码准备脚本](scripts/prepare_sources.sh) 与 [构建脚本](scripts/ss5_workspace_v3d.sh)。

## 输出

终端打印 `RESULT_DIR`，默认位于 `$IFAB_AFS_HOME/ss5-fl2va-*`。其中 `reference_fl2va.mp4` 和 `infinilm_fl2va.mp4` 分别为参考及原生视频，`run.log` 为官方比较结果，其他日志与 `arguments.json` 记录本次运行。

八卡路径已实测成片，官方平均视频 SSIM 为 **94.687%**。整理后的统一复现入口尚待在新八卡环境完整复跑。
