# MiniMax-H3 (PyTorch)

<div align="center">

**原生 PyTorch 音视频端到端联合生成系统**  
零第三方框架依赖 · 原生变长 SDPA 与 3D RoPE · 32kHz 立体声声画协同

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.4%2B-ee4c2c.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-Apache_2.0-green.svg)](LICENSE)
[![Zero-Dependency](https://img.shields.io/badge/Diffusers%20%2F%20SGLang-0%20Dependency-brightgreen.svg)]()

[English](README.md) | [简体中文](README_zh.md)

</div>

---

## 📌 项目概述

**MiniMax-H3** 具备前沿的音视频端到端协同生成能力。目前官方与社区开源实现普遍深度依赖 `sglang` 或 `diffusers` 等重型推理框架，存在底层代码侵入严重、模块耦合高、跨平台环境编译繁琐等问题。

本项目提供了 **MiniMax-H3 的解耦原生 PyTorch 工程实现**（架构遵循 LTX-2.5 模块化管线标准）：
- **全链路原生化**：核心网络结构（DiT、Video VAE、Audio VAE、文本编码器）全部由原生 `torch.nn.Module` 构建，彻底剔除 `diffusers` 与 `sglang` 库依赖。
- **免扩展编译**：注意力机制基于 PyTorch 原生 `F.scaled_dot_product_attention` 与 `cu_seqlens` 切片，原生支持 3D RoPE 与变长多模态 Packed 序列计算，**无需编译安装 `flash-attn` 等 C++/CUDA 扩展**。
- **声画协同高保真生成**：
  - **Video VAE**：3D 因果卷积结构，内置升余弦汉宁窗（Hann Window）重叠分块平滑解码（Tiled Spatial Decode），彻底消除空间 Patch 高频网格伪影。
  - **Audio VAE**：32 通道声学潜在空间解码器，前置因果自注意力与高阶卷积网络，直接生成 32kHz 立体声高保真音频。
  - **调度算法**：支持非线性时间步 Shift 映射的 Flow Matching Euler ODE 求解器。
- **权重双向无损互转**：提供通用格式转换工具，支持 Native PyTorch $\leftrightarrow$ Diffusers $\leftrightarrow$ SGLang 三向权重等价互转（100% 键对齐与浮点 0 误差）。
- **高效微调基座**：内置基于速度预测目标（Flow Matching Velocity Loss）的轻量化 LoRA 微调管线。

---

## 🏗️ 架构与模块组织

```text
packages/
├── __init__.py                # packages 顶层导出
├── README.md                  # 英文主页文档 (English default)
├── README_zh.md               # 中文使用文档
├── requirements.txt           # 极简依赖列表
├── LICENSE                    # Apache-2.0 开源许可
├── .gitignore                 # 权重与大文件过滤
├── setup.py                   # pip 安装配置
├── pyproject.toml             # 标准 Python 构建配置
│
└── minimaxh3/                 # MiniMax-H3 核心模块包
    ├── __init__.py            # 对外导出主要类与函数
    ├── models/                # 核心神经网络组件
    │   ├── transformer.py     # 50层 DiT 主模型 (3D RoPE, 变长 SDPA, AdaLayerNormZero)
    │   ├── video_vae.py       # 3D 因果时空卷积 Video VAE (汉宁窗分块解码)
    │   ├── audio_vae.py       # 32kHz 立体声声学解码器
    │   └── text_encoder.py    # Qwen3-VL 50层多模态文本特征抽取器
    ├── pipelines/             # 采样管线
    │   └── pipeline_minimax_h3.py # MiniMaxH3Pipeline (端到端声画联合生成)
    ├── schedulers/            # 调度算法
    │   └── flow_match.py      # FlowMatchEuler 调度器 (Shifted ODE 求解)
    ├── utils/                 # 工具类
    │   ├── packed_sequence.py # Packed 变长序列切片与模态 Tag 映射
    │   └── media_io.py        # FFmpeg 原生管道流音视频封装与抽帧
    ├── scripts/               # 生产入口与工具脚本
    │   ├── inference.py       # 端到端音视频生成命令行脚本
    │   ├── train_sft.py       # SFT / LoRA 轻量化训练脚本
    │   ├── convert_weights.py # 权重格式双向转换工具
    │   └── test_no_dependencies.py # 零依赖 AST 静态与动态审计脚本
    └── run_inference.sh       # 快速运行脚本
```

---

## ⚙️ 运行环境要求

### 硬件要求
- **GPU**：NVIDIA GPU，算力 $\ge 8.0$（Ampere / Ada / Hopper 架构，如 RTX 3090 / 4090，A100，H100 / H200 等）。
- **显存推荐**：
  - 全 GPU 常驻极速模式（`--no_cpu_offload`）：显存建议 $\ge 40\text{GB}$。
  - 显存卸载模式（`--cpu_offload`）：最低 $16\text{GB} \sim 24\text{GB}$ 显存即可运行。

### 软件依赖
- **Python**：`>= 3.10`（测试通过：Python 3.12）
- **PyTorch**：`>= 2.4.0`（兼容 PyTorch 2.4 / 2.5 / 2.6 / 2.14）
- **torchvision & torchaudio**：与 PyTorch 版本保持一致
- **transformers**：`>= 4.45.0`（用于 Qwen3-VL 文本 Tokenizer 与特征抽取）
- **einops / safetensors / Pillow / numpy / tqdm**
- **ffmpeg**：系统多媒体处理工具（用于音视频合流封装与抽帧，Ubuntu 可直接 `sudo apt install ffmpeg`）

---

## 🚀 快速上手

### 1. 安装项目

```bash
# 克隆代码仓库
git clone https://github.com/Xuan292929/Minimax-H3-packages.git
cd Minimax-H3-packages

# 安装基础依赖
pip install -r requirements.txt

# (可选) 可编辑模式安装为 Python 包
pip install -e .
```

### 2. 权重存放目录

模型权重目录组织结构如下（兼容官方权重分片格式）：
```text
checkpoints/FL2VA/
├── transformer/      # DiT 权重分片 (safetensors)
├── video_vae/        # Video VAE 权重 (包含 source/ 目录)
├── audio_vae/        # Audio VAE 权重
├── text_encoder/     # Qwen3-VL 文本编码器权重
└── processor/        # Tokenizer 分词器配置与词表
```

### 3. 一键运行推理

#### 方式 A：使用 Shell 脚本快速启动
```bash
# 可选：通过环境变量自定义权重路径
export MODEL_PATH="./checkpoints/FL2VA"
bash minimaxh3/run_inference.sh
```

#### 方式 B：使用 Python CLI 运行（支持自定义参数）
```bash
python minimaxh3/scripts/inference.py \
  --model_path ./checkpoints/FL2VA \
  --prompt "A majestic lion standing proudly on a rocky outcrop at sunset, golden hour light, cinematic lighting, photorealistic 4k." \
  --height 480 \
  --width 832 \
  --num_frames 39 \
  --num_inference_steps 25 \
  --seed 42 \
  --device cuda:0 \
  --tile_size 256 \
  --tile_overlap 64 \
  --output_path ./outputs/lion_output.mp4 \
  --extract_frames
```

#### 方式 C：低显存运行（CPU Offload 模式）
针对 16GB / 24GB 显存设备，使用 `--cpu_offload` 参数：
```bash
python minimaxh3/scripts/inference.py \
  --model_path ./checkpoints/FL2VA \
  --prompt "A serene mountain lake at sunrise, crystal water, birds singing." \
  --cpu_offload \
  --output_path ./outputs/lake.mp4
```

### 4. Python API 代码调用

```python
import torch
from packages.minimaxh3 import MiniMaxH3Pipeline

# 初始化音视频联合生成管线
pipeline = MiniMaxH3Pipeline.from_pretrained(
    model_path="./checkpoints/FL2VA",
    device="cuda:0",
    torch_dtype=torch.bfloat16,
    cpu_offload=False,
)

# 生成音视频一体 MP4 文件
video, audio = pipeline(
    prompt="A futuristic city with flying vehicles at neon night, cyberpunk aesthetic, ambient synthwave audio.",
    height=480,
    width=832,
    num_frames=39,
    num_inference_steps=25,
    seed=42,
    tile_size=256,
    tile_overlap=64,
    output_path="./outputs/cyberpunk.mp4",
)
print("音视频生成完成，已保存至 ./outputs/cyberpunk.mp4")
```

---

## 🔧 进阶工具

### 1. 权重格式双向无损互转（Native $\leftrightarrow$ Diffusers $\leftrightarrow$ SGLang）

```bash
# 原生 PyTorch 权重转为 Diffusers 格式
python minimaxh3/scripts/convert_weights.py \
  --input_path ./checkpoints/native_weights.safetensors \
  --output_path ./checkpoints/diffusers_weights.safetensors \
  --mode to_diffusers

# Diffusers 格式转回原生 PyTorch 权重
python minimaxh3/scripts/convert_weights.py \
  --input_path ./checkpoints/diffusers_weights.safetensors \
  --output_path ./checkpoints/native_weights.safetensors \
  --mode from_diffusers

# 权重可逆性校验（比特级 0 误差验证）
python minimaxh3/scripts/convert_weights.py \
  --input_path ./checkpoints/weights.safetensors \
  --mode verify_roundtrip
```

### 2. SFT / LoRA 轻量化微调

基于 Flow Matching 速度预测目标对 DiT 块进行参数高效微调：
```bash
python minimaxh3/scripts/train_sft.py \
  --model_path ./checkpoints/FL2VA \
  --num_layers 4 \
  --lora_rank 16 \
  --lr 1e-4 \
  --steps 100 \
  --device cuda:0 \
  --output_dir ./outputs/sft_checkpoint
```

### 3. 架构零依赖审计

随时运行代码静态语法树与动态运行时审计，检验代码零第三方框架污染：
```bash
python minimaxh3/scripts/test_no_dependencies.py
```
预期输出：
```text
[Audit] Scanning python files in ... for forbidden imports...
PASS: No static imports of sglang or diffusers found in source files!
[Audit] Checking runtime sys.modules upon importing package...
PASS: Runtime sys.modules is completely clean of sglang and diffusers!
[Audit] ALL CHECKS PASSED: Criterion strictly satisfied!
```

---

## 📄 开源许可证

本项目遵循 [Apache 2.0 开源许可证](LICENSE)。
