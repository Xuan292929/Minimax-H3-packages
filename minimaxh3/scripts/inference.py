#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Native PyTorch Video & Audio Inference Script for MiniMax-H3.
Criterion 1: 视频推理成功，抽帧后图片清晰可见。
Criterion 2: 不依赖 sglang 代码 / diffusers 代码等第三方库。
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import torch
from PIL import Image

# Ensure repo root is on sys.path
repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

try:
    from minimaxh3.pipelines.pipeline_minimax_h3 import MiniMaxH3Pipeline
    from minimaxh3.utils.media_io import extract_video_frames, save_video_audio
except ImportError:
    try:
        from packages.minimaxh3.pipelines.pipeline_minimax_h3 import MiniMaxH3Pipeline
        from packages.minimaxh3.utils.media_io import extract_video_frames, save_video_audio
    except ImportError:
        from minimax_pytorch.pipelines.pipeline_minimax_h3 import MiniMaxH3Pipeline
        from minimax_pytorch.utils.media_io import extract_video_frames, save_video_audio


def main():
    parser = argparse.ArgumentParser(description="Standalone PyTorch MiniMax-H3 Inference.")
    parser.add_argument(
        "--model_path",
        type=str,
        default=os.environ.get("MINIMAX_MODEL_PATH", "./checkpoints/FL2VA"),
        help="Path to MiniMax-H3 weights directory (containing text_encoder, transformer, video_vae, audio_vae, processor).",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="A majestic lion standing proudly on a rocky outcrop at sunset, golden hour light, cinematic lighting, photorealistic 4k.",
        help="Text prompt for generation.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="./outputs/minimax_h3_output.mp4",
        help="Output mp4 filepath.",
    )
    parser.add_argument("--height", type=int, default=480, help="Video height (multiple of 32).")
    parser.add_argument("--width", type=int, default=832, help="Video width (multiple of 32).")
    parser.add_argument("--num_frames", type=int, default=124, help="Number of frames (17n + 5: e.g. 39, 56, 124).")
    parser.add_argument("--num_inference_steps", type=int, default=50, help="Denoising steps.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--flow_shift", type=float, default=6.0, help="Flow matching shift for video.")
    parser.add_argument("--audio_flow_shift", type=float, default=3.0, help="Flow matching shift for audio.")
    parser.add_argument("--device", type=str, default="cuda:0", help="CUDA device.")
    parser.add_argument("--cpu_offload", action="store_true", default=True, help="Enable sequential CPU offload to save VRAM.")
    parser.add_argument("--no_cpu_offload", dest="cpu_offload", action="store_false", help="Keep all models in GPU VRAM.")
    parser.add_argument("--first_frame", type=str, default=None, help="Optional image path for first frame conditioning.")
    parser.add_argument("--last_frame", type=str, default=None, help="Optional image path for last frame conditioning.")
    parser.add_argument("--ref_images", nargs="+", type=str, default=None, help="Optional reference image path(s) for reference/multi-reference conditioning.")
    parser.add_argument("--mode", type=str, default="auto", choices=["auto", "t2v", "fl2va", "ref2va"], help="Conditioning mode (auto-detected from inputs by default).")
    parser.add_argument("--extract_frames", action="store_true", default=True, help="Extract PNG frames from generated video.")
    parser.add_argument("--frames_dir", type=str, default=None, help="Directory to save extracted PNG frames.")
    parser.add_argument("--tile_size", type=int, default=256, help="VAE decode tile size (official default: 256).")
    parser.add_argument("--tile_overlap", type=int, default=64, help="VAE decode tile overlap (official default: 64).")
    parser.add_argument("--fp8", action="store_true", default=False, help="Run DiT in native 128x128 Block-FP8 mode (saves ~50% VRAM).")

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
    print(f"=== MiniMax-H3 PyTorch Inference ===", flush=True)
    print(f"Model Path: {args.model_path}", flush=True)
    print(f"Prompt: {args.prompt}", flush=True)
    print(f"Resolution: {args.width}x{args.height}, {args.num_frames} frames ({args.num_inference_steps} steps)", flush=True)
    print(f"Device: {args.device} (CPU offload: {args.cpu_offload}, FP8 native: {args.fp8})", flush=True)
    print(f"VAE Tiling: True (tile_size={args.tile_size}, tile_overlap={args.tile_overlap})", flush=True)
    if args.first_frame:
        print(f"First Frame: {args.first_frame}", flush=True)
    if args.last_frame:
        print(f"Last Frame: {args.last_frame}", flush=True)
    if args.ref_images:
        print(f"Reference Images ({len(args.ref_images)}): {args.ref_images}", flush=True)

    t0 = time.time()
    pipe = MiniMaxH3Pipeline.from_pretrained(
        model_path=args.model_path,
        torch_dtype=torch.bfloat16,
        device=args.device,
        enable_cpu_offload=args.cpu_offload,
        use_fp8=args.fp8,
    )

    t_load = time.time() - t0
    print(f"Model loading complete in {t_load:.1f}s.", flush=True)

    t1 = time.time()
    frames, audio = pipe(
        prompt=args.prompt,
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        num_inference_steps=args.num_inference_steps,
        seed=args.seed,
        flow_shift=args.flow_shift,
        audio_flow_shift=args.audio_flow_shift,
        first_frame=args.first_frame,
        last_frame=args.last_frame,
        ref_images=args.ref_images,
        mode=args.mode,
        tiled=True,
        tile_size=args.tile_size,
        tile_overlap=args.tile_overlap,
    )
    t_gen = time.time() - t1
    print(f"Inference generation complete in {t_gen:.1f}s.", flush=True)

    # Save output video and audio
    save_video_audio(
        video_frames=frames,
        audio_waveform=audio,
        output_path=args.output_path,
        fps=24,
        audio_sample_rate=32000,
    )
    print(f"Saved video to: {args.output_path}", flush=True)

    out_dir = os.path.dirname(os.path.abspath(args.output_path))
    # Extract audio directly from generated MP4 (Criterion 1)
    try:
        from minimaxh3.utils.media_io import extract_audio_track
    except ImportError:
        try:
            from packages.minimaxh3.utils.media_io import extract_audio_track
        except ImportError:
            from minimax_pytorch.utils.media_io import extract_audio_track
    extracted_audio_path = os.path.join(out_dir, "extracted_audio.wav")
    print(f"Extracting audio track from MP4 container to {extracted_audio_path}...", flush=True)
    extract_audio_track(args.output_path, extracted_audio_path)
    if os.path.exists(extracted_audio_path):
        import torchaudio
        wf, sr = torchaudio.load(extracted_audio_path)
        print(f"Extracted audio verified: channels={wf.shape[0]}, samples={wf.shape[1]}, sr={sr}Hz, "
              f"duration={wf.shape[1]/sr:.2f}s, peak={wf.abs().max().item():.3f}, "
              f"rms={(wf**2).mean().sqrt().item():.4f}", flush=True)

    # Frame extraction (Acceptance Criterion 1)
    if args.extract_frames:
        if args.frames_dir is None:
            args.frames_dir = os.path.join(out_dir, "extracted_frames")
        print(f"Extracting frames to {args.frames_dir}...", flush=True)
        extracted = extract_video_frames(
            video_path=args.output_path,
            output_dir=args.frames_dir,
            timestamps_sec=[0.0, 0.5, 1.0, 1.5],
        )
        print(f"Extracted {len(extracted)} frames successfully:", flush=True)
        for p in extracted:
            print(f"  - {p}", flush=True)

    print("=== Inference Completed Successfully ===", flush=True)


if __name__ == "__main__":
    main()
