# MiniMax-H3 (PyTorch)

<div align="center">

**Native PyTorch Video & Audio Joint Generation System**  
Zero Third-Party Dependencies · Native Variable-Length SDPA & 3D RoPE · 32kHz Stereo Co-Generation

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.4%2B-ee4c2c.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-Apache_2.0-green.svg)](LICENSE)
[![Zero-Dependency](https://img.shields.io/badge/Diffusers%20%2F%20SGLang-0%20Dependency-brightgreen.svg)]()

[English](README.md) | [简体中文](README_zh.md)

</div>

---

## 📌 Project Overview

**MiniMax-H3** features state-of-the-art end-to-end synchronized video and audio generation. Existing official and community implementations heavily couple the model with large inference engines such as `sglang` or `diffusers`, creating deep dependency bloat, high architectural coupling, and cross-platform compilation barriers.

This repository delivers a **clean, decoupled native PyTorch engineering implementation of MiniMax-H3** (designed following the modular pipeline standards of LTX-2.5):
- **100% Native PyTorch**: All neural network modules (DiT, Video VAE, Audio VAE, Text Encoder) are built purely with `torch.nn.Module`, completely eliminating third-party inference library dependencies (`diffusers`, `sglang`).
- **Zero Extension Compilation**: Attention computation is natively implemented via PyTorch `F.scaled_dot_product_attention` and `cu_seqlens` dynamic slicing. Native 3D RoPE and variable-length packed multimodal sequences work out-of-the-box **without compiling C++/CUDA extensions like `flash-attn`**.
- **Synchronized Audio-Video Co-Generation**:
  - **Video VAE**: 3D Causal Convolutional architecture featuring Hann-window overlap blending (`tiled_decode`), eliminating spatial patch boundary artifacts.
  - **Audio VAE**: 32-channel acoustic latent space decoder synthesizing rich 32kHz stereo audio directly aligned with video timing.
  - **Flow Matching Scheduler**: Shifted Euler ODE solver designed for joint multimodal trajectory sampling.
- **Bidirectional Lossless Weight Conversion**: Built-in format converter supporting 100% key alignment and 0.0 floating-point difference across Native PyTorch, Diffusers, and SGLang formats.
- **Efficient Fine-Tuning**: Out-of-the-box parameter-efficient LoRA training pipeline based on Flow Matching velocity prediction.

---

## 🏗️ Architecture & Directory Structure

```text
packages/
├── __init__.py                # Packages top-level entry point
├── README.md                  # English documentation (default homepage)
├── README_zh.md               # Chinese documentation (简体中文说明)
├── requirements.txt           # Minimal pip dependencies
├── LICENSE                    # Apache-2.0 open-source license
├── .gitignore                 # Binary weights & outputs filter
├── setup.py                   # Package setup configuration
├── pyproject.toml             # Standard PEP 517/621 specification
│
└── minimaxh3/                 # MiniMax-H3 core module
    ├── __init__.py            # Main classes and functions export
    ├── models/                # Neural network modules
    │   ├── transformer.py     # 50-layer DiT backbone (3D RoPE, Varlen SDPA, AdaLayerNormZero)
    │   ├── video_vae.py       # 3D Causal Video VAE (Hann-window tiled spatial decoding)
    │   ├── audio_vae.py       # 32kHz stereo acoustic latent space decoder
    │   └── text_encoder.py    # Qwen3-VL 50-layer multimodal feature extractor
    ├── pipelines/             # Sampling pipeline
    │   └── pipeline_minimax_h3.py # MiniMaxH3Pipeline (End-to-end joint video-audio sampling)
    ├── schedulers/            # Diffusion & flow matching schedulers
    │   └── flow_match.py      # FlowMatchEuler scheduler (Shifted ODE solver)
    ├── utils/                 # Utilities
    │   ├── packed_sequence.py # Packed multimodal sequence builder & coordinate mapping
    │   └── media_io.py        # FFmpeg rawvideo multiplexing & lossless frame extraction
    ├── scripts/               # Production entry points
    │   ├── inference.py       # End-to-end video & audio generation CLI
    │   ├── train_sft.py       # SFT / LoRA baseline fine-tuning script
    │   ├── convert_weights.py # Native <-> Diffusers <-> SGLang weight converter
    │   └── test_no_dependencies.py # Zero-dependency AST audit script
    └── run_inference.sh       # One-click shell launch script
```

---

## ⚙️ Environment & Requirements

### Hardware Requirements
- **GPU**: NVIDIA GPU with Compute Capability $\ge 8.0$ (Ampere / Ada / Hopper, e.g., RTX 3090 / 4090, A100, H100, H200).
- **VRAM Recommendation**:
  - Full GPU execution (`--no_cpu_offload`): $\ge 40\text{GB}$ VRAM recommended.
  - CPU Offloading mode (`--cpu_offload`): $16\text{GB} \sim 24\text{GB}$ VRAM.

### Software Dependencies
- **Python**: `>= 3.10` (tested on Python 3.12)
- **PyTorch**: `>= 2.4.0` (compatible with PyTorch 2.4 / 2.5 / 2.6 / 2.14)
- **torchvision & torchaudio**: Matching PyTorch version
- **transformers**: `>= 4.45.0` (for Qwen3-VL tokenizer & text feature extraction)
- **einops / safetensors / Pillow / numpy / tqdm**
- **ffmpeg**: System multimedia tool for audio/video muxing (`sudo apt install ffmpeg`).

---

## 🚀 Quickstart

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/Xuan292929/Minimax-H3-packages.git
cd Minimax-H3-packages

# Install dependencies
pip install -r requirements.txt

# (Optional) Editable installation
pip install -e .
```

### 2. Model Weights Structure

Ensure your model weights directory is organized as follows:
```text
checkpoints/FL2VA/
├── transformer/      # DiT sharded safetensors
├── video_vae/        # Video VAE weights (containing source/ folder)
├── audio_vae/        # Audio VAE weights
├── text_encoder/     # Qwen3-VL text encoder weights
└── processor/        # Tokenizer configs and files
```

### 3. Video & Audio Inference

#### Option A: Quick Shell Launch
```bash
# Optional: specify your weights path via environment variable
export MODEL_PATH="./checkpoints/FL2VA"
bash minimaxh3/run_inference.sh
```

#### Option B: Python Command Line
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
  --output_path ./outputs/lion.mp4 \
  --extract_frames
```

#### Option C: Low-VRAM Execution (CPU Offload)
For GPUs with 16GB - 24GB VRAM:
```bash
python minimaxh3/scripts/inference.py \
  --model_path ./checkpoints/FL2VA \
  --prompt "A serene mountain lake at sunrise, crystal clear water, natural ambient sound." \
  --cpu_offload \
  --output_path ./outputs/lake.mp4
```

### 4. Multimodal Conditioning Modes (首帧 / 尾帧 / 首尾帧 / 多参参考)

The pipeline natively supports all MiniMax-H3 conditioning workflows via both CLI and Python API:

#### Mode 1: 首帧生视频 (First Frame to Video / I2V)
```bash
python minimaxh3/scripts/inference.py \
  --model_path ./checkpoints/FL2VA \
  --first_frame ./assets/start_frame.png \
  --prompt "A magical glowing crystal cavern with flowing ethereal waters." \
  --output_path ./outputs/i2v_first_frame.mp4
```

#### Mode 2: 尾帧生视频 (Last Frame to Video)
```bash
python minimaxh3/scripts/inference.py \
  --model_path ./checkpoints/FL2VA \
  --last_frame ./assets/end_frame.png \
  --prompt "Camera panning toward a golden twilight sunset over mountains." \
  --output_path ./outputs/last_frame_gen.mp4
```

#### Mode 3: 首尾帧生视频 (Start & End Frame Transition / 首尾帧)
```bash
python minimaxh3/scripts/inference.py \
  --model_path ./checkpoints/FL2VA \
  --first_frame ./assets/start_frame.png \
  --last_frame ./assets/end_frame.png \
  --prompt "A seamless cinematic transition from dawn crystal cave to sunset mountains." \
  --output_path ./outputs/transition_first_last.mp4
```

#### Mode 4: 参考生视频 / 多参 (Reference & Multi-Reference Images)
```bash
python minimaxh3/scripts/inference.py \
  --model_path ./checkpoints/Ref2VA \
  --ref_images ./assets/ref_subject.png ./assets/ref_style.png \
  --prompt "A cybernetic futuristic flower blooming in neon lighting." \
  --output_path ./outputs/multi_ref_video.mp4
```

---

### 5. Python API Usage

```python
import torch
from PIL import Image
from minimaxh3.pipelines import MiniMaxH3Pipeline

# Initialize pipeline
pipeline = MiniMaxH3Pipeline.from_pretrained(
    model_path="./checkpoints/FL2VA",
    device="cuda:0",
    torch_dtype=torch.bfloat16,
    enable_cpu_offload=False,
)

# 1. Text-to-Video (T2V)
video, audio = pipeline(
    prompt="A futuristic city at neon night, ambient synthwave audio.",
    height=480,
    width=832,
    num_frames=39,
)

# 2. 首帧生视频 (I2V)
video, audio = pipeline(
    prompt="The crystal glows brighter with mystic sparks.",
    first_frame="start_frame.png",
)

# 3. 首尾帧过渡 (First & Last frame)
video, audio = pipeline(
    prompt="A cinematic transition connecting the start and end frames.",
    first_frame="start_frame.png",
    last_frame="end_frame.png",
)

# 4. 参考生视频 / 多参 (Multi-Reference)
video, audio = pipeline(
    prompt="Subject in style of references.",
    ref_images=["ref1.png", "ref2.png"],
)
```

---

## 🔧 Advanced Tools

### 1. Bidirectional Weight Conversion (Native $\leftrightarrow$ Diffusers $\leftrightarrow$ SGLang)

```bash
# Native PyTorch -> Diffusers format
python minimaxh3/scripts/convert_weights.py \
  --input_path ./checkpoints/native_weights.safetensors \
  --output_path ./checkpoints/diffusers_weights.safetensors \
  --mode to_diffusers

# Diffusers format -> Native PyTorch
python minimaxh3/scripts/convert_weights.py \
  --input_path ./checkpoints/diffusers_weights.safetensors \
  --output_path ./checkpoints/native_weights.safetensors \
  --mode from_diffusers

# Bit-level lossless verification (Roundtrip)
python minimaxh3/scripts/convert_weights.py \
  --input_path ./checkpoints/weights.safetensors \
  --mode verify_roundtrip
```

### 2. SFT / LoRA Fine-Tuning (All Conditioning Modes Supported)

The SFT training pipeline supports fine-tuning across all 5 conditioning paradigms:
- `t2v`: Pure Text-to-Video
- `first_frame`: 首帧生视频 (I2V)
- `last_frame`: 尾帧生视频
- `first_last`: 首尾帧生视频
- `multi_ref`: 参考生视频 / 多参

```bash
# Train across all modes in sequence
python minimaxh3/scripts/train_sft.py \
  --task_mode all \
  --steps 5 \
  --device cuda:0

# Or train a specific conditioning mode
python minimaxh3/scripts/train_sft.py \
  --task_mode first_frame \
  --steps 100 \
  --lora_rank 16 \
  --lr 1e-4 \
  --device cuda:0 \
  --output_dir ./outputs/sft_first_frame
```

### 3. Zero-Dependency Audit

Verify anytime that the codebase strictly satisfies zero third-party dependencies:
```bash
python minimaxh3/scripts/test_no_dependencies.py
```
Output:
```text
[Audit] Scanning python files in ... for forbidden imports...
PASS: No static imports of sglang or diffusers found in source files!
[Audit] Checking runtime sys.modules upon importing package...
PASS: Runtime sys.modules is completely clean of sglang and diffusers!
[Audit] ALL CHECKS PASSED: Criterion strictly satisfied!
```

---

## 📄 License

This project is licensed under the [Apache 2.0 License](LICENSE).
