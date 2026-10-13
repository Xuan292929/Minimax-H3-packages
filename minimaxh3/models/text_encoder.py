# SPDX-License-Identifier: Apache-2.0
"""
Native PyTorch implementation of MiniMax-H3 Qwen3-VL Text Encoder and prompt presentation builders.
Pure PyTorch and standard transformers model/tokenizer, zero dependency on sglang or diffusers.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from transformers import Qwen3VLConfig, Qwen3VLModel

VISION_START = "<|vision_start|>"
VISION_END = "<|vision_end|>"
IMAGE_PAD = "<|image_pad|>"
VIDEO_PAD = "<|video_pad|>"

PRESENTATION_TEXT_TAG = 1
PRESENTATION_VIDEO_TAG = 0

QWEN_VIDEO_SAMPLE_FPS = 2.0
QWEN_TEMPORAL_PATCH = 2
MINIMAX_SUPPORTED_FPS = 24


class MiniMaxH3TextEncoder(nn.Module):
    """
    Qwen3-VL 50-layer multimodal feature extractor for MiniMax-H3.
    """

    def __init__(self, num_retained_layers: int = 50) -> None:
        super().__init__()
        self.num_retained_layers = num_retained_layers
        config = Qwen3VLConfig(
            **{
                "architectures": ["Qwen3VLForConditionalGeneration"],
                "image_token_id": 151655,
                "model_type": "qwen3_vl",
                "text_config": {
                    "attention_bias": False,
                    "attention_dropout": 0.0,
                    "bos_token_id": 151643,
                    "dtype": "bfloat16",
                    "eos_token_id": 151645,
                    "head_dim": 128,
                    "hidden_act": "silu",
                    "hidden_size": 5120,
                    "initializer_range": 0.02,
                    "intermediate_size": 25600,
                    "max_position_embeddings": 262144,
                    "model_type": "qwen3_vl_text",
                    "num_attention_heads": 64,
                    "num_hidden_layers": num_retained_layers,
                    "num_key_value_heads": 8,
                    "rms_norm_eps": 1e-06,
                    "rope_scaling": {
                        "mrope_interleaved": True,
                        "mrope_section": [24, 20, 20],
                        "rope_type": "default",
                    },
                    "rope_theta": 5000000,
                    "use_cache": True,
                    "vocab_size": 151936,
                },
                "tie_word_embeddings": False,
                "transformers_version": "4.57.0.dev0",
                "video_token_id": 151656,
                "vision_config": {
                    "deepstack_visual_indexes": [8, 16, 24],
                    "depth": 27,
                    "hidden_act": "gelu_pytorch_tanh",
                    "hidden_size": 1152,
                    "in_channels": 3,
                    "initializer_range": 0.02,
                    "intermediate_size": 4304,
                    "model_type": "qwen3_vl",
                    "num_heads": 16,
                    "num_position_embeddings": 2304,
                    "out_hidden_size": 5120,
                    "patch_size": 16,
                    "spatial_merge_size": 2,
                    "temporal_patch_size": 2,
                },
                "vision_end_token_id": 151653,
                "vision_start_token_id": 151652,
            }
        )
        self.model = Qwen3VLModel(config)
        self.model.language_model.norm = nn.Identity()
        self.config = config
        self.image_token_id = config.image_token_id
        self.video_token_id = config.video_token_id

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.Tensor] = None,
        video_grid_thw: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        mm_token_type_ids = None
        if pixel_values is not None or pixel_values_videos is not None:
            mm_token_type_ids = torch.zeros_like(input_ids, dtype=torch.int32)
            mm_token_type_ids[input_ids == self.image_token_id] = 1
            mm_token_type_ids[input_ids == self.video_token_id] = 2
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            pixel_values_videos=pixel_values_videos,
            video_grid_thw=video_grid_thw,
            mm_token_type_ids=mm_token_type_ids,
            output_hidden_states=False,
            use_cache=False,
            return_dict=True,
            **kwargs,
        )
        return outputs.last_hidden_state[0]


def image_token_counts(processor: Any, images: List[Any]) -> Tuple[torch.Tensor, torch.Tensor, List[int]]:
    vision = processor.image_processor(images=images, return_tensors="pt")
    merge = int(processor.image_processor.merge_size) ** 2
    counts = [int(vision["image_grid_thw"][i].prod().item()) // merge for i in range(len(images))]
    return vision["pixel_values"], vision["image_grid_thw"], counts


def video_token_counts(
    processor: Any, videos: List[Any], timestamps_per_video: List[List[float]]
) -> Tuple[torch.Tensor, torch.Tensor, List[List[int]], List[List[float]]]:
    vout = processor.video_processor(videos=videos, do_sample_frames=False, return_tensors="pt")
    grid = vout["video_grid_thw"]
    merge = int(processor.image_processor.merge_size) ** 2
    block_counts, block_timestamps = [], []
    for index, timestamps in enumerate(timestamps_per_video):
        n_blocks = int(grid[index, 0])
        per_block = int(grid[index, 1]) * int(grid[index, 2]) // merge
        block_counts.append([per_block] * n_blocks)
        block_timestamps.append([float(t) for t in timestamps])
    return vout["pixel_values_videos"], grid, block_counts, block_timestamps


def _text_ids(tokenizer: Any, text: str) -> List[int]:
    return list(tokenizer(text, add_special_tokens=False)["input_ids"])


def _vision_block_ids(tokenizer: Any, pad_token: str, count: int) -> List[int]:
    return (
        [tokenizer.convert_tokens_to_ids(VISION_START)]
        + [tokenizer.convert_tokens_to_ids(pad_token)] * int(count)
        + [tokenizer.convert_tokens_to_ids(VISION_END)]
    )


class _Presentation:
    def __init__(self) -> None:
        self.ids: List[int] = []
        self.tags: List[int] = []

    def text(self, token_ids: List[int]) -> None:
        self.ids += token_ids
        self.tags += [PRESENTATION_TEXT_TAG] * len(token_ids)

    def vision(self, token_ids: List[int]) -> None:
        self.ids += token_ids
        self.tags += [PRESENTATION_VIDEO_TAG] * len(token_ids)

    def build(self) -> Tuple[torch.Tensor, torch.Tensor]:
        return (torch.tensor(self.ids, dtype=torch.long), torch.tensor(self.tags, dtype=torch.long))


def presentation_t2va(tokenizer: Any, prompt: str) -> Tuple[torch.Tensor, torch.Tensor]:
    presentation = _Presentation()
    presentation.text(_text_ids(tokenizer, prompt))
    return presentation.build()


def presentation_fl2va(tokenizer: Any, prompt: str, image_token_counts: List[int]) -> Tuple[torch.Tensor, torch.Tensor]:
    presentation = _Presentation()
    for index, count in enumerate(image_token_counts, start=1):
        presentation.text(_text_ids(tokenizer, f"<Picture {index}>: "))
        presentation.vision(_vision_block_ids(tokenizer, IMAGE_PAD, count))
    presentation.text(_text_ids(tokenizer, prompt))
    return presentation.build()


def presentation_ref2va(
    tokenizer: Any,
    prompt: str,
    condition_labels: List[Tuple[str, int]],
    image_token_counts: List[int],
    video_block_token_counts: List[List[int]],
    video_block_timestamps: List[List[float]],
) -> Tuple[torch.Tensor, torch.Tensor]:
    if not prompt:
        raise ValueError("prompt must be non-empty")
    presentation = _Presentation()
    image_seen = 0
    video_seen = 0
    for cond_type, ordinal in condition_labels:
        if cond_type == "image":
            image_seen += 1
            if image_seen > len(image_token_counts):
                raise ValueError("image_token_count required for an image reference")
            count = int(image_token_counts[image_seen - 1])
            if count <= 0:
                raise ValueError("image_token_count required for an image reference")
            presentation.text(_text_ids(tokenizer, f"<Picture {ordinal}>: "))
            presentation.vision(_vision_block_ids(tokenizer, IMAGE_PAD, count))
        elif cond_type == "audio":
            presentation.text(_text_ids(tokenizer, f"<Audio {ordinal}>: "))
        elif cond_type == "video":
            video_seen += 1
            if video_seen > len(video_block_token_counts):
                raise ValueError("video reference requires block token counts and timestamps")
            counts = video_block_token_counts[video_seen - 1]
            timestamps = video_block_timestamps[video_seen - 1]
            if not counts or len(counts) != len(timestamps):
                raise ValueError("video block token counts and timestamps must align")
            presentation.text(_text_ids(tokenizer, f"<Video {ordinal}>: "))
            for count, timestamp in zip(counts, timestamps):
                if int(count) <= 0:
                    raise ValueError("video block token count must be positive")
                presentation.text(_text_ids(tokenizer, f"<{timestamp:.1f} seconds>"))
                presentation.vision(_vision_block_ids(tokenizer, VIDEO_PAD, count))
        else:
            raise ValueError(f"unsupported ref2va condition type {cond_type!r}")
    if image_seen != len(image_token_counts):
        raise ValueError("unused image_token_count entries")
    if video_seen != len(video_block_token_counts):
        raise ValueError("unused video block token count entries")
    presentation.text(_text_ids(tokenizer, prompt))
    return presentation.build()


def sample_qwen_video_frames(frames: List[Any]) -> Tuple[List[Any], List[float]]:
    ratio = MINIMAX_SUPPORTED_FPS / QWEN_VIDEO_SAMPLE_FPS
    indices: List[int] = []
    cursor = 0.0
    while True:
        idx = int(round(cursor))
        if idx >= len(frames):
            break
        if not indices or idx > indices[-1]:
            indices.append(idx)
        cursor += ratio
    if not indices:
        raise ValueError("no frames sampled for the Qwen video presentation")
    ts = [i / QWEN_VIDEO_SAMPLE_FPS for i in range(len(indices))]
    pad = (-len(ts)) % QWEN_TEMPORAL_PATCH
    ts = ts + [ts[-1]] * pad
    block_timestamps = [(ts[i] + ts[i + QWEN_TEMPORAL_PATCH - 1]) / 2 for i in range(0, len(ts), QWEN_TEMPORAL_PATCH)]
    return [frames[i] for i in indices], block_timestamps


__all__ = [
    "MiniMaxH3TextEncoder",
    "presentation_t2va",
    "presentation_fl2va",
    "presentation_ref2va",
    "image_token_counts",
    "video_token_counts",
    "sample_qwen_video_frames",
]
