#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Comprehensive End-to-End Validation Script for all 4 MiniMax-H3 Conditioning Modes:
1. 首帧生视频 (First Frame to Video / I2V)
2. 尾帧生视频 (Last Frame to Video)
3. 首尾帧生视频 (First & Last Frame to Video)
4. 参考生视频 / 多参 (Reference / Multi-Reference Images to Video)
"""

import os
import sys
import time
import torch
from PIL import Image

# Ensure repo root is on sys.path
repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from minimaxh3.pipelines.pipeline_minimax_h3 import MiniMaxH3Pipeline
from minimaxh3.utils.media_io import save_video_audio, extract_video_frames

def main():
    model_path = os.environ.get("MINIMAX_MODEL_PATH", "./checkpoints/FL2VA")
    device = "cuda:0"
    print(f"=== MiniMax-H3 All Conditioning Modes Validation ===")
    print(f"Loading pipeline from {model_path} on {device}...")

    pipe = MiniMaxH3Pipeline.from_pretrained(
        model_path=model_path,
        torch_dtype=torch.bfloat16,
        device=device,
        enable_cpu_offload=False,
    )

    test_first = "test_assets/first_frame.png"
    test_last = "test_assets/last_frame.png"
    test_ref1 = "test_assets/ref_image1.png"
    test_ref2 = "test_assets/ref_image2.png"

    os.makedirs("./outputs/test_modes", exist_ok=True)

    test_configs = [
        {
            "mode_name": "Mode 1: 首帧生视频 (First-frame I2V)",
            "kwargs": {
                "prompt": "A magical blue glowing crystal pulsing with cosmic energy in a deep cavern.",
                "first_frame": test_first,
            },
            "output_path": "./outputs/test_modes/mode1_first_frame.mp4",
        },
        {
            "mode_name": "Mode 2: 尾帧生视频 (Last-frame to video)",
            "kwargs": {
                "prompt": "A warm golden sunset dissolving into twilight over distant mountain silhouettes.",
                "last_frame": test_last,
            },
            "output_path": "./outputs/test_modes/mode2_last_frame.mp4",
        },
        {
            "mode_name": "Mode 3: 首尾帧生视频 (First & Last frame to video)",
            "kwargs": {
                "prompt": "A smooth seamless transformation from a blue crystal dawn to an ember sunset.",
                "first_frame": test_first,
                "last_frame": test_last,
            },
            "output_path": "./outputs/test_modes/mode3_first_last.mp4",
        },
        {
            "mode_name": "Mode 4: 参考生视频 / 多参 (Multi-Reference images to video)",
            "kwargs": {
                "prompt": "A futuristic vibrant cybernetic flower blooming in neon lighting with electronic ambient sound.",
                "ref_images": [test_ref1, test_ref2],
            },
            "output_path": "./outputs/test_modes/mode4_multi_ref.mp4",
        },
    ]

    for cfg in test_configs:
        print(f"\n=======================================================")
        print(f"Testing: {cfg['mode_name']}")
        print(f"=======================================================")
        t0 = time.time()
        frames, audio = pipe(
            height=480,
            width=832,
            num_frames=22,
            num_inference_steps=2,
            seed=42,
            **cfg["kwargs"],
        )
        dt = time.time() - t0
        num_f = frames.shape[2] if isinstance(frames, torch.Tensor) and frames.dim() == 5 else len(frames)
        print(f"Generation successful: {num_f} frames, audio shape={audio.shape} in {dt:.2f}s")
        save_video_audio(video_frames=frames, audio_waveform=audio, output_path=cfg["output_path"], fps=24)
        assert os.path.exists(cfg["output_path"]), f"Failed to save {cfg['output_path']}"
        print(f"Saved: {cfg['output_path']} (size={os.path.getsize(cfg['output_path'])} bytes)")

    print("\n=======================================================")
    print("ALL 4 CONDITIONING MODES TESTED AND PASSED SUCCESSFULLY!")
    print("=======================================================")

if __name__ == "__main__":
    main()
