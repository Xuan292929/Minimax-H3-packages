# SPDX-License-Identifier: Apache-2.0
"""
MiniMax-H3 PyTorch: Standalone PyTorch Training & Inference Codebase
Reference design: LTX-Video / LTX-2.5 clean modular pipeline
Zero dependency on sglang or diffusers.
"""

from .models.transformer import MiniMaxH3DiTModel
from .models.video_vae import MiniMaxH3VideoVAE
from .models.audio_vae import MiniMaxH3AudioVAE
from .models.text_encoder import MiniMaxH3TextEncoder
from .schedulers.flow_match import MiniMaxH3FlowMatchScheduler
from .pipelines.pipeline_minimax_h3 import MiniMaxH3Pipeline
from .utils.media_io import save_video_audio, extract_video_frames

__all__ = [
    "MiniMaxH3DiTModel",
    "MiniMaxH3VideoVAE",
    "MiniMaxH3AudioVAE",
    "MiniMaxH3TextEncoder",
    "MiniMaxH3FlowMatchScheduler",
    "MiniMaxH3Pipeline",
    "save_video_audio",
    "extract_video_frames",
]
