# MiniMax H3 八卡九齿优化复现

本工程在 InfiniFabric 的 8 张海光 gfx936 上运行 MiniMax-H3 FL2VA。注意力、AdaLN 和门控残差由九齿编译链生成，接入 InfiniCore / InfiniLM。当前分支仅保留准备环境、生成算子、构建、生成视频和官方比较所需源码。

## 公开源码

| 仓库 | branch | commit |
| --- | --- | --- |
| [InfiniCore](https://github.com/Yuuki-git/InfiniCore) | minimax-h3-8gpu-opt | 72f87009d6a6fdd970ee8365c99e7416bd59d4a0 |
| [InfiniLM](https://github.com/Yuuki-git/InfiniLM) | minimax-h3-8gpu-opt | 207f2fa95adec6104134cb95ce29acb8e5846178 |
| [NineToothed](https://github.com/Yuuki-git/ninetoothed-ss5) | main | 2e2d4da420d6edde1e0f3c17e05a43f111198d05 |
| [工程入口](https://github.com/Yuuki-git/minimax-h3-8gpu) | main | 固定标签 v3-ss5-minimal；完整 commit 见提交报告 |

前三个仓库的版本由 `source-lock.json` 固定。Core 基于官方 f32e2fbb2a9b2ca912b5711f4f3836c403196a74，仅改动 11 个文件；LM 基于官方 65deb91f840ebaef752d028fad2f3217ce0950f4，仅改动 Transformer 的 1 个文件。官方生成、比较脚本保持原样。九齿基于官方 0.26.0 wheel 完整源码，补丁版为 0.26.0+ss5.1。NTops SDPA 定义来源为 82cba6cf2200919b5d12c86adbea60cbfb516040，随源码保留 Apache-2.0 许可证。

## 基础环境

需要主办方 gfx936 镜像、8 张可见 GPU、Python 3.11、DTK 26.04、Git、C++ 编译器、make、tar 和 unzip。基础 Python 必须提供未修改的海光 Torch 2.9.0+das.opt1.dtk2604、Triton 3.5.1+das.opt1.dtk2604.torch290 和 FlashAttention 2.6.1+das.opt1.dtk2604.torch290。默认使用 `python3`；如厂商组件位于评审方自己的基础环境，可在准备前设置 `SS5_VENDOR_PYTHON=/该环境/bin/python`。

`prepare.sh` 在当前会话目录新建私有 venv，继承上述厂商组件，安装 requirements.txt 中的模型依赖，从公开固定源码准备 xmake 3.0.5、Core/LM/九齿及 C++ 依赖。镜像缺少 torchvision 0.24.0 时，使用官方源码配合现有 Torch 构建。九齿代码生成使用单独 venv 和 NumPy 1.26.4，模型运行使用 NumPy 1.25.0。所有新增内容都位于 `$SS5_OPT_WORK`，无需本队的个人 AFS 目录。

源码与依赖准备需要 GitHub / PyPI 网络。权重由主办方统一提供，默认 `/public/home/aclsylqidf/shared/MiniMax-H3/diffusers`；可在生成和比较命令中加入 `--model-path` 指定评审方已有权重。

## 全部复现命令

在新的八卡网页会话中执行。`IFAB_WORKSPACE_ROOT` 使用平台给出的值。构建并发为 8，三个算子和视频生成依次执行。

```bash
set -euo pipefail
cd -- "${IFAB_WORKSPACE_ROOT:?}"
export SS5_OPT_WORK="$PWD/ss5-optimized"
git clone --branch v3-ss5-minimal --depth 1 \
  https://github.com/Yuuki-git/minimax-h3-8gpu.git
cd minimax-h3-8gpu
bash scripts/prepare.sh
bash scripts/build.sh
bash scripts/run.sh reference \
  --output-dir "$SS5_OPT_WORK/results/reference"
bash scripts/run.sh native \
  --output-dir "$SS5_OPT_WORK/results/optimized"
bash scripts/run.sh compare \
  --reference-video "$SS5_OPT_WORK/results/reference/reference_fl2va.mp4" \
  --native-video "$SS5_OPT_WORK/results/optimized/infinilm_fl2va.mp4" \
  --output-dir "$SS5_OPT_WORK/results/comparison"
```

统一首帧位于 `assets/ss5_fl2va_template_first_frame.png`。生成固定使用 `A dog falls asleep`、seed 42、50 步、124 帧、768×928、24 FPS、GPU 0–7。`reference` 为官方 PyTorch / Diffusers 路径，`native` 为 InfiniLM / InfiniCore 路径，`compare` 调用官方比较脚本。输出视频与比较日志分别在上述三个 results 子目录。

默认 `SS5_NT_ATTN=1 SS5_NT_ADALN=1`。消融开关 00、01、10、11 保留，并须在启动进程前设置。00 为候选中的原算子回退路径；官方未修改基线另保留在 v1-baseline 标签。

注意力仅替换已适配的 BF16 `[1,27570,7,128]` 非因果、无 mask / KV cache 请求，packed V 的 token stride 为 2688。AdaLN / 门控使用宽度 5376 的调制布局并保留 BF16 逐步舍入。其他布局沿用原路径。构建时从九齿定义生成包含设备码的 HIP `.cpp` 和三个共享库；Core 的 `generated/*.h` 为接口。模型运行无需九齿 Python 包，厂商 Triton 仍用于 LDS 属性查询。

## 已有平台结果

2026-11-11，同一会话同配置的一组完整视频配对：

| 指标 | 原基线 | 双开优化 |
| --- | ---: | ---: |
| 生成流水线 | 441.909863 s | 287.327724 s |
| 官方脚本内部总时间，含加载和编码 | 481.573203 s | 329.299466 s |
| 平均视频 SSIM | 95.888506% | 96.309953% |

流水线加速 1.537999×、耗时降低 34.9805%；官方脚本内部总时间降低 31.6201%。官方比较未设最小 SSIM 门槛。以上数字对应此前已运行的同一数值候选；此精简公开入口的从零平台构建与运行尚未补验。

评审提交物为一份报告，其中包含四个公开仓库的链接、branch、完整 commit、环境前提和上述全部命令。实验日志、视频、运行压缩包及内部交接材料不作为源码附件提交。
