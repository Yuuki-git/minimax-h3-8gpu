# MiniMax-H3 八卡 FL2VA 适配

本仓库在 `main` 主分支提供海光 gfx936 八卡适配的最小复现代码，包括构建配置、主机同步适配和准备运行脚本。视频、日志和官方比较结果在运行时生成。

原生 InfiniLM/InfiniCore 路径已在主办方环境生成有声视频：统一首帧、`A dog falls asleep`、seed 42、50 步、124 帧，输出为 768 × 928、24 FPS。固定版官方 `minimax_h3_compare.py` 的平均视频 SSIM 为 **94.687%**。

| 适配内容 | 实现位置 |
| --- | --- |
| 启用固定版 Core 已有的海光稠密 MHA | [ss5_workspace_v3d.sh](scripts/ss5_workspace_v3d.sh) 中的独立 xmake 配置 |
| Torch 与原生模型交接时等待全部 8 张卡 | [forward_fence.py](development/forward_fence.py) |
| 参考成片、原生成片及官方 FL2VA 比较 | [run_fl2va.py](scripts/run_fl2va.py) |
| 保存编译产物并在新会话恢复 | [ss5_native_checkpoint_v2.py](scripts/ss5_native_checkpoint_v2.py) |

## 环境

InfiniFabric 的 Linux x86_64 八卡会话，8 × gfx936、Python 3.11.9、镜像自带海光 PyTorch 2.9.0、HIP 6.3.26093，DTK 位于 `/opt/dtk`。需要 Git、tar、unzip、gzip、make、CMake、g++、readelf、flock 和 diff。

沿用主办方指南准备的独立环境 `$IFAB_AFS_HOME/venvs/minimax-h3`，其中已有模型推理依赖及厂商 Triton、FlashAttention。Transformers 为 5.14.1、Hugging Face Hub 为 1.28.0；torchvision 为与镜像 Torch 匹配、包含 `torchvision::nms` 的 0.24.0，按指南源码构建。覆盖与比较依赖版本见 [requirements.txt](requirements.txt)。脚本只在该 venv 中启用 Diffusers 0.40.0 和比较依赖。

还需已有 `$IFAB_AFS_HOME/tools/bin/xmake`，版本为 **3.0.5+HEAD.f20295f**。模型默认读取 `/public/home/aclsylqidf/shared/MiniMax-H3/diffusers`；使用 `--model-path` 可指定其他官方权重目录。首帧使用主办方报告模板中的统一图片，通过 `--first-frame` 指定。

编译在 `$IFAB_WORKSPACE_ROOT/ss5-v3-work` 进行，并发为 8。GPU 任务串行，输出默认保存在个人 AFS。每个新会话需重新构建或恢复编译产物。

## 运行

在有网络的 Linux 机器或 WSL 的 Linux 文件系统中准备固定源码及依赖：

```bash
git clone --branch main --depth 1 https://github.com/Yuuki-git/minimax-h3-8gpu.git
cd minimax-h3-8gpu
bash scripts/reproduce.sh --prepare-only
```

准备结果位于 `dist/platform-inputs`。将整个已准备工程传入个人 AFS，保留源码包及 Git 元数据。在有 8 卡的网页会话中，从工程目录执行：

```bash
bash scripts/reproduce.sh --offline \
  --first-frame "$IFAB_AFS_HOME/ss4-prep/ss5_fl2va_template_first_frame.png"
```

脚本将准备好的输入放入个人 AFS，启用固定 Diffusers 和比较依赖，构建 Core/LM，再依次生成 PyTorch 参考视频、InfiniLM 原生视频，并调用原版官方比较脚本。两路配置均为上述固定样例，原生调用前后同步 GPU 0–7。

当前会话已完成 Core/LM 构建时，可跳过构建：

```bash
bash scripts/reproduce.sh --offline --skip-build \
  --first-frame "$IFAB_AFS_HOME/ss4-prep/ss5_fl2va_template_first_frame.png"
```

已有同配置 PyTorch 参考视频时，可增加 `--reference-video /个人AFS/参考视频.mp4`，只生成原生视频并进行官方比较。

### 保存和恢复

在已完成构建的会话中，进入个人 AFS 保存编译产物。`repo_dir` 指向本工程的绝对路径：

```bash
repo_dir="$IFAB_AFS_HOME/minimax-h3-8gpu"
cd -P -- "$IFAB_AFS_HOME"
python3 -B "$repo_dir/scripts/ss5_native_checkpoint_v2.py" save
```

记下输出的 `SS5_RUNTIME_SAVED` 绝对路径。新开 8 卡会话后，把它填入 `runtime_dir`，即可恢复并运行：

```bash
runtime_dir="$IFAB_AFS_HOME/你保存的运行包目录"
cd -- "$IFAB_AFS_HOME/minimax-h3-8gpu"
bash scripts/reproduce.sh --offline --skip-build --runtime "$runtime_dir" \
  --first-frame "$IFAB_AFS_HOME/ss4-prep/ss5_fl2va_template_first_frame.png"
```

该运行包保存 Core/LM 编译产物及官方入口，Python venv、厂商库和模型权重继续使用个人存储与平台原有内容。

## 代码与结果

[`development`](development) 保存八卡同步适配；[`scripts`](scripts) 保存依赖准备、原生库构建、成片和官方比较入口；[`scripts/reproduce.sh`](scripts/reproduce.sh) 为统一执行入口。

每次运行输出新的 `RESULT_DIR`，默认是个人 AFS 中的 `ss5-fl2va-*`，包含 `reference_fl2va.mp4`、`infinilm_fl2va.mp4`、`run.log`、生成日志及本次参数。下载视频和提交官方比较结果时使用该目录。复用参考视频时，跨轮耗时不能作为同轮加速结论。

本轮八卡成片与官方比较已实测；这里重新整理的最小入口完成本地检查，尚未在新的八卡会话中完整复跑。当前交付范围为八卡运行适配，新增九齿算子优化及模型端到端提速仍需后续接入和测量。

## 固定上游与依赖

| 项目 | 固定版本或提交 | 官方来源 |
| --- | --- | --- |
| InfiniCore | `f32e2fbb2a9b2ca912b5711f4f3836c403196a74` | [InfiniTensor/InfiniCore](https://github.com/InfiniTensor/InfiniCore) |
| InfiniLM | `65deb91f840ebaef752d028fad2f3217ce0950f4` | [InfiniTensor/InfiniLM](https://github.com/InfiniTensor/InfiniLM) |
| xmake | `f20295f255d967af9d91de6f1ed5eeadf17d30a7` | [xmake-io/xmake](https://github.com/xmake-io/xmake) |
| xmake-repo | `bd29d77f242e247589457638272d9936a24a3fe0` | [xmake-io/xmake-repo](https://github.com/xmake-io/xmake-repo) |
| Boost / pybind11 | 1.92.0 / 3.1.0 | [Boost](https://github.com/boostorg/boost)、[pybind11](https://github.com/pybind/pybind11) |
| Diffusers | 0.40.0 | [Hugging Face/Diffusers](https://github.com/huggingface/diffusers) |

固定提交、下载地址和构建参数保存在准备及构建脚本中。Core/LM 官方数学源码、模型权重和官方评测脚本沿用固定版本；海光 MHA 使用上游现有实现。新增或重写的模型数值算子仍按赛题使用九齿编译链。本仓库只保存本队适配和复现源码，第三方依赖在准备时从官方来源获取，各自的许可证声明继续适用。
