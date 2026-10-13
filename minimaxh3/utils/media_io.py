# SPDX-License-Identifier: Apache-2.0
"""
Media I/O utilities: Video & Audio saving, multiplexing with ffmpeg, and frame extraction.
Pure PyTorch, torchvision, torchaudio, and standard system tools.
"""

from __future__ import annotations

import os
import subprocess
import tempfile
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import torchvision
from PIL import Image


def convert_to_stereo(waveform: torch.Tensor) -> torch.Tensor:
    """
    Ensure waveform is [2, T].
    """
    while waveform.dim() > 2:
        waveform = waveform.squeeze(0)
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)
    if waveform.shape[0] == 1:
        waveform = waveform.repeat(2, 1)
    elif waveform.shape[0] > 2:
        waveform = waveform[:2]
    return waveform


def resample_waveform(waveform: torch.Tensor, orig_sr: int, target_sr: int) -> torch.Tensor:
    if orig_sr == target_sr:
        return waveform
    try:
        import torchaudio.transforms as T
        resampler = T.Resample(orig_sr, target_sr).to(waveform.device)
        return resampler(waveform)
    except Exception:
        # Fallback linear interpolation
        length = waveform.shape[-1]
        target_len = int(round(length * target_sr / orig_sr))
        return torch.nn.functional.interpolate(waveform.unsqueeze(0), size=target_len, mode="linear", align_corners=False).squeeze(0)


def save_video_audio(
    video_frames: Union[List[Image.Image], torch.Tensor],
    output_path: Union[str, torch.Tensor],
    audio_waveform: Optional[Union[str, torch.Tensor]] = None,
    fps: int = 24,
    audio_sample_rate: int = 32000,
) -> str:
    """
    Save video frames and optional audio soundtrack into an MP4 file.
    Uses ffmpeg to combine H.264 video with AAC audio.
    """
    # Gracefully handle swapped argument order save_video_audio(frames, audio, path)
    if isinstance(output_path, torch.Tensor) and isinstance(audio_waveform, (str, os.PathLike)):
        audio_waveform, output_path = output_path, str(audio_waveform)
    output_path = str(output_path)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    # Convert frames to numpy array uint8 [T, H, W, 3]
    if isinstance(video_frames, torch.Tensor):
        if video_frames.dim() == 5:  # [B, C, T, H, W]
            video_frames = video_frames[0].permute(1, 2, 3, 0).contiguous()  # [T, H, W, C]
        elif video_frames.dim() == 4:  # [T, H, W, C] or [C, T, H, W]
            if video_frames.shape[0] == 3:
                video_frames = video_frames.permute(1, 2, 3, 0).contiguous()
        frames_np = (video_frames.float().clamp(0, 1) * 255).to(torch.uint8).cpu().numpy()
        frames_np = np.ascontiguousarray(frames_np)
    elif isinstance(video_frames, list):
        if isinstance(video_frames[0], Image.Image):
            frames_np = np.stack([np.array(f.convert("RGB")) for f in video_frames])
        else:
            frames_np = np.stack([np.array(f) for f in video_frames])
        frames_np = np.ascontiguousarray(frames_np)
    else:
        raise ValueError(f"Unsupported video_frames type: {type(video_frames)}")

    with tempfile.TemporaryDirectory() as tmp_dir:
        temp_video = os.path.join(tmp_dir, "temp_video.mp4")
        temp_audio = os.path.join(tmp_dir, "temp_audio.wav")

        # Encode video frames with ffmpeg rawvideo pipe (pure system tool, zero library dependency)
        h, w = frames_np.shape[1], frames_np.shape[2]
        cmd_v = [
            "ffmpeg",
            "-y",
            "-f", "rawvideo",
            "-vcodec", "rawvideo",
            "-s", f"{w}x{h}",
            "-pix_fmt", "rgb24",
            "-r", str(fps),
            "-i", "-",
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-crf", "18",
            temp_video,
        ]
        proc = subprocess.Popen(cmd_v, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        _, err = proc.communicate(input=frames_np.tobytes())
        if proc.returncode != 0:
            raise RuntimeError(f"ffmpeg video encoding failed: {err.decode('utf-8', errors='ignore')}")

        if audio_waveform is not None:
            import torchaudio
            audio_wf = convert_to_stereo(audio_waveform.float().cpu())
            # Peak normalize to -1.0 dBFS so that the sound is loud, clear, and audible
            peak = float(audio_wf.abs().max().item())
            if peak > 1e-4:
                audio_wf = (audio_wf / peak) * 0.90
            
            # Resample to 44.1kHz for universal device and browser compatibility
            if audio_sample_rate != 44100:
                audio_wf = resample_waveform(audio_wf, audio_sample_rate, 44100)
                mux_sr = 44100
            else:
                mux_sr = audio_sample_rate

            torchaudio.save(temp_audio, audio_wf, mux_sr)

            # Also save companion WAV alongside MP4 for direct playback/analysis
            companion_wav = os.path.splitext(output_path)[0] + ".wav"
            try:
                torchaudio.save(companion_wav, audio_wf, mux_sr)
            except Exception:
                pass

            # Multiplex video and audio with ffmpeg
            cmd = [
                "ffmpeg",
                "-y",
                "-i", temp_video,
                "-i", temp_audio,
                "-c:v", "copy",
                "-c:a", "aac",
                "-ar", str(mux_sr),
                "-b:a", "192k",
                "-shortest",
                output_path,
            ]
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, check=True)
        else:
            # Just move temp_video to output_path
            import shutil
            shutil.copy(temp_video, output_path)

    return output_path


def extract_audio_track(video_path: str, output_audio_path: str) -> str:
    """
    Extract audio track from MP4 container into a standard WAV file.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_audio_path)), exist_ok=True)
    cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", "44100",
        output_audio_path,
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, check=True)
    return output_audio_path


def extract_video_frames(
    video_path: str,
    output_dir: str,
    frame_indices: Optional[List[int]] = None,
    timestamps_sec: Optional[List[float]] = None,
) -> List[str]:
    """
    Extract key frames from video as PNG images for quality inspection.
    """
    os.makedirs(output_dir, exist_ok=True)
    extracted_paths = []

    if timestamps_sec is not None:
        for idx, t in enumerate(timestamps_sec):
            out_img = os.path.join(output_dir, f"frame_time_{t:.2f}s.png")
            cmd = [
                "ffmpeg", "-y",
                "-ss", str(t),
                "-i", video_path,
                "-vframes", "1",
                "-q:v", "2",
                out_img,
            ]
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, check=True)
            if os.path.exists(out_img):
                extracted_paths.append(out_img)
    elif frame_indices is not None:
        # Extract via select filter or ffmpeg vframes
        for frame_idx in frame_indices:
            out_img = os.path.join(output_dir, f"frame_{frame_idx:04d}.png")
            cmd = [
                "ffmpeg", "-y",
                "-i", video_path,
                "-vf", f"select=eq(n\\,{frame_idx})",
                "-vframes", "1",
                out_img,
            ]
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, check=True)
            if os.path.exists(out_img):
                extracted_paths.append(out_img)
    else:
        # Default: extract evenly spaced 5 frames across the video
        out_pattern = os.path.join(output_dir, "frame_%03d.png")
        cmd = [
            "ffmpeg", "-y",
            "-i", video_path,
            "-vf", "fps=1",
            out_pattern,
        ]
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, check=True)
        for f in sorted(os.listdir(output_dir)):
            if f.endswith(".png"):
                extracted_paths.append(os.path.join(output_dir, f))

    return extracted_paths


__all__ = [
    "convert_to_stereo",
    "resample_waveform",
    "save_video_audio",
    "extract_audio_track",
    "extract_video_frames",
]
