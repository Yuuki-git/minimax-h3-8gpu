# MiniMax H3 FL2VA 八卡九齿优化

在 InfiniFabric 的 8 张海光 `gfx936` 上，本队通过 NineToothed 生成注意力、AdaLN 和门控残差算子，并接入 InfiniCore 与 InfiniLM。S6 同会话一组完整 FL2VA 视频配对中，生成流水线从 **441.909863 s 降至 287.327724 s**，耗时降低 **34.9805%**，加速 **1.5380×**。官方视频比较的平均 SSIM 从 **95.888506% 提升至 96.309953%**。

以上为已完成的平台实测。此版本将同一候选的源码整理为公开构建入口；本地已核对源文件、接口摘要和脚本，**尚未在平台完成本公开入口的从零 clone、构建、冒烟补验**。这些数值不作为重新构建版本的实测值。

## 公开源码与固定提交

| 仓库 | 分支 | 提交 |
| --- | --- | --- |
| [InfiniCore](https://github.com/Yuuki-git/InfiniCore) | `minimax-h3-8gpu-opt` | `72f87009d6a6fdd970ee8365c99e7416bd59d4a0` |
| [InfiniLM](https://github.com/Yuuki-git/InfiniLM) | `minimax-h3-8gpu-opt` | `207f2fa95adec6104134cb95ce29acb8e5846178` |
| [NineToothed](https://github.com/Yuuki-git/ninetoothed-ss5) | `main` | `2e2d4da420d6edde1e0f3c17e05a43f111198d05` |
| [工程入口](https://github.com/Yuuki-git/minimax-h3-8gpu) | `main`，固定标签 `v2.1-ss5-optimized` | 以固定标签取得本版；其余三个提交及文件摘要见 [source-lock.json](source-lock.json) |

Core 基于官方 `f32e2fbb2a9b2ca912b5711f4f3836c403196a74`，仅修改 11 个候选文件；LM 基于官方 `65deb91f840ebaef752d028fad2f3217ce0950f4`，仅修改 Transformer 的 1 个文件。原基线保留在工程入口的 `v1-baseline` 标签（`0a56c656bae7c907a6d72d0382aa120ae5ec3f1e`）。原有基线脚本保持原样。

九齿仓库的首个提交从官方 PyPI `ninetoothed==0.26.0` wheel 完整导入模块，文件和 wheel 摘要见其 `UPSTREAM.json`；第二个提交包含 `generation.py` 的缓存路径补丁、`aot.py` 的厂商 HIP 输出适配及打包元数据。补丁版包版本为 `0.26.0+ss5.1`。未替换厂商 Torch、Triton 或 FlashAttention。

## 环境与只读输入

在具有网络访问能力的 **8 卡 gfx936 网页会话**运行下方命令。保留平台设置的 `IFAB_WORKSPACE_ROOT` 与 `IFAB_AFS_HOME`。`IFAB_AFS_HOME` 只提供已有依赖：

* `venvs/minimax-h3/bin/python`：Python 3.11.9，海光 Torch `2.9.0+das.opt1.dtk2604`、Triton `3.5.1+das.opt1.dtk2604.torch290`、FlashAttention `2.6.1+das.opt1.dtk2604.torch290`，以及 Diffusers 0.40.0、Transformers 5.14.1。
* `tools/bin/xmake`：`3.0.5+HEAD.f20295f`。
* `src-tars/xmake-repo-bd29d77f-src.tgz`：固定依赖仓库 `bd29d77f242e247589457638272d9936a24a3fe0`。
* `xmake-pkgs/boost-1.92.0.tar.gz` 和 `xmake-pkgs/pybind11-v3.1.0.zip`：摘要分别为 `f51707c27359a0df0cac1beada86de31bb5eed5e8285592dadec384df99c2984`、`affea1ada7b39fe1d835559fcb78800c7927d514bd480ed01b71b88205a5e536`。

镜像提供 DTK 26.04、C++ 编译器、Git、Bash、tar/unzip。默认只读模型目录为 `/public/home/aclsylqidf/shared/MiniMax-H3/diffusers`，可用 `--model-path` 指定已有 Diffusers 权重。仓库内含统一首帧，不含权重、视频或二进制库。

所有新源码、构建环境、缓存、日志与安装产物写入当前网页工作区的 `ss5-optimized/`，构建并发为 8，三个算子编译和 GPU 测试串行。构建专用 venv 从锁定九齿源码建立私有覆盖层，并校验、解包 NumPy 1.26.4 wheel；原运行环境保持不变。`FA_SO_PATH` 在导入 Torch 前指向现有厂商扩展。

## 从公开入口构建与运行

以下命令与报告逐行一致，完整脚本见 [docs/reproduce-optimized.sh](docs/reproduce-optimized.sh)。首次验证可执行到 `smoke` 为止；后三条命令生成 PyTorch 参考、原生候选与官方比较结果，需预留完整成片时间。目标源码和结果目录必须不存在；重试使用新的专用工作目录。

```bash
set -euo pipefail
cd -- "${IFAB_WORKSPACE_ROOT:?}"
export SS5_OPT_WORK="${IFAB_WORKSPACE_ROOT:?}/ss5-optimized"
git clone --branch v2.1-ss5-optimized --depth 1 \
  https://github.com/Yuuki-git/minimax-h3-8gpu.git
cd minimax-h3-8gpu
bash scripts/fetch_sources.sh
bash scripts/build_optimized.sh
bash scripts/run_optimized.sh smoke
bash scripts/run_optimized.sh reference \
  --output-dir "$SS5_OPT_WORK/results/reference"
bash scripts/run_optimized.sh native \
  --output-dir "$SS5_OPT_WORK/results/optimized"
bash scripts/compare.sh \
  --reference-video "$SS5_OPT_WORK/results/reference/reference_fl2va.mp4" \
  --native-video "$SS5_OPT_WORK/results/optimized/infinilm_fl2va.mp4" \
  --output-dir "$SS5_OPT_WORK/results/comparison"
```

生成参数固定为 prompt `A dog falls asleep`、seed 42、50 步、124 帧、768×928、GPU 0–7。`reference` 调用原版生成脚本的 `--legacy` 模式，`native` 调用原版 InfiniLM 入口并核验 50 次原生调用及八卡调用边界同步，`compare` 调用固定官方比较脚本。各进程单独启动，在进程初始化前读取开关。

默认 `SS5_NT_ATTN=1 SS5_NT_ADALN=1`，保留 `00/01/10/11` 消融开关：例如 `SS5_NT_ATTN=0 SS5_NT_ADALN=0 bash scripts/run_optimized.sh native --output-dir "$SS5_OPT_WORK/results/ablation00"`。`00` 表示候选中的原算子回退路径；官方未修改基线的构建入口为 `v1-baseline`。八卡新构建冒烟必须使用 `11`，在 Core 非默认流验证注意力及两种融合算子，并核对每项 8 次宿主提交。

构建记录保存在 `build-manifest.json`，包括公开源码锁定、运行包版本、原生库摘要。每个算子的 `codegen/*/build.json` 保留编译命令、厂商编译器来源、HIP/code object 摘要、LDS 和资源元数据。生成的 `.cpp` 内嵌由本次九齿/Triton 编译得到的 hsaco；Core 中四个 `generated/*.h` 仅为接口，构建时校验新头文件摘要。运行阶段不导入九齿 Python 包；厂商 Triton 仍用于每卡 LDS 属性查询。

## 已验收结果与范围

S6 为同一会话、同一配置的一组完整视频基线/候选配对，非多轮平均。候选双开 `11`，全流程使用已验收的同一套 AOT 与 Core/LM 库。

| 计时范围 | 原基线 s | 优化候选 s | 耗时降低 |
| --- | ---: | ---: | ---: |
| 生成流水线 `generation.pipeline_total` | 441.909863 | 287.327724 | 34.9805% |
| 49 次 H3 合计 | 410.405959 | 255.674176 | 37.7021% |
| 官方脚本内部总时间，含模型加载和编码 | 481.573203 | 329.299466 | 31.6201% |
| 完整工作进程壁钟，含导入、预检和媒体复核 | 563.549146 | 447.251250 | 20.6367% |

官方视频平均 SSIM：基线 95.88850583047505%，候选 96.30995271760216%；提高 0.421446887 个百分点。三路视频均可完整解码为 124 帧、24 FPS、5.1667 s；音频均为 32 kHz 双声道、165888 个有限解码采样点。官方比较脚本没有定义最小 SSIM 门槛，比较完成状态与实际指标分别记录。

S5 模型单步为同输入的两组基线/双开配对，几何平均加速 1.6216856×、耗时降低 38.3358%，计时排除权重加载、输入克隆和 CPU 比较。四开关消融中，`00/01` 与原基线逐值一致，`10/11` 彼此逐值一致。它与 S6 完整视频计时范围不同。

注意力专门覆盖 BF16 `[1,27570,7,128]`、非因果、无 mask/无 KV cache、packed V token stride 2688。AdaLN/门控覆盖宽度 5376 的实际调制布局，保留 BF16 逐步舍入。其他布局回退原算子。49 次 H3 的宿主提交计数为 attention 19600、AdaLN 39592、gate 39200；计数表示成功提交，不代表执行时长。

后续工作包括候选额外约 32.9 s 的生成前准备开销归因、进一步注意力调度优化、RMSNorm 融合和多输入回归。公开入口从零构建及冒烟的验收结果应单独补记，不回填为本表 S6 测量。
