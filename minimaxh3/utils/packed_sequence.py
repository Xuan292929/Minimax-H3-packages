# SPDX-License-Identifier: Apache-2.0
"""
Sequence packing and 3D Rotary Position Grid generator for MiniMax-H3.
Pure PyTorch and numpy, zero external framework dependency.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

_INTERP = 32
_T_GROUP = 5
_FRAME_PER_TOKEN = (1, 4, 4, 4, 4)
_FRAME_RESCALE = 5.0 / 3.0
_SEQ_ALIGN = 64


def align_dimension(val: int, factor: int) -> int:
    return ((val + factor - 1) // factor) * factor


def align_video_frames(num_frames: int, factor: int = 17, remainder: int = 5) -> int:
    """
    Snap num_frames to nearest 17n + 5 (e.g., 5, 22, 39, 56, 73, 90, 107, 124).
    """
    if num_frames < remainder:
        return remainder
    n = max(0, math.ceil((num_frames - remainder) / factor))
    return n * factor + remainder


class PackedSequenceBuilder:
    def __init__(self) -> None:
        pass

    @staticmethod
    def _aligned_seq_len(used: int) -> int:
        return ((used + _SEQ_ALIGN - 1) // _SEQ_ALIGN) * _SEQ_ALIGN

    @staticmethod
    def _axis_from_sqrt_area(dim: int, patch: int, sqrt_area: float) -> torch.Tensor:
        ratio = dim / sqrt_area
        left = (1.0 - ratio) * 0.5
        right = left + ratio
        grid = np.linspace(left, right, dim // patch, endpoint=False) * _INTERP
        return torch.from_numpy(grid).to(torch.float64)

    @classmethod
    def _video_t_grid(cls, n: int, origin: float) -> torch.Tensor:
        spans = torch.tensor([_FRAME_RESCALE * _FRAME_PER_TOKEN[k % _T_GROUP] for k in range(n)], dtype=torch.float64)
        return origin + torch.cat([torch.zeros(1, dtype=torch.float64), spans[:-1].cumsum(0)])

    @classmethod
    def _temporal_position_span(cls, temporal_length: int) -> float:
        spans = np.ones(int(temporal_length), dtype=np.float64) * _FRAME_RESCALE
        for token_index in range(_T_GROUP):
            spans[token_index::_T_GROUP] *= _FRAME_PER_TOKEN[token_index]
        return float(spans.sum())

    @classmethod
    def _frame_grid(cls, latent_h: int, latent_w: int, sqrt_area: float) -> Tuple[torch.Tensor, torch.Tensor]:
        h_grid = cls._axis_from_sqrt_area(latent_h, 2, sqrt_area)
        w_grid = cls._axis_from_sqrt_area(latent_w, 2, sqrt_area)
        hh, ww = torch.meshgrid(h_grid, w_grid, indexing="ij")
        return torch.stack([hh.reshape(-1), ww.reshape(-1)], dim=-1), w_grid

    @classmethod
    def _video_grid(cls, latent_t: int, frame: torch.Tensor, origin: float) -> torch.Tensor:
        video_g = torch.empty(latent_t, frame.shape[0], 3, dtype=torch.float64)
        video_g[:, :, 0] = cls._video_t_grid(latent_t, origin)[:, None]
        video_g[:, :, 1:] = frame[None]
        return video_g.reshape(-1, 3)

    @staticmethod
    def _audio_w_axis(w_grid: torch.Tensor, audio_t: int, audio_rows: int) -> torch.Tensor:
        return torch.cat(
            [
                torch.full((audio_t,), float(w_grid[0]), dtype=torch.float64),
                torch.full((audio_rows - audio_t,), float(w_grid[-1]), dtype=torch.float64),
            ]
        )

    def build_packed_fl2va(
        self,
        text_len: int,
        latent_t: int,
        latent_h: int,
        latent_w: int,
        audio_t: int,
        keyframe_indices: List[int],
        audio_channel: int = 2,
    ) -> Dict[str, Any]:
        """
        fl2va sequence layout: [text | cond | audio | video | pad]
        """
        frame_rows = (latent_h // 2) * (latent_w // 2)
        video_rows = latent_t * frame_rows
        audio_rows = audio_t * audio_channel
        num_keyframes = len(keyframe_indices)
        cond_rows = num_keyframes * frame_rows
        used = text_len + cond_rows + audio_rows + video_rows
        seq_len = self._aligned_seq_len(used)

        text_sl = slice(0, text_len)
        cond_sl = slice(text_len, text_len + cond_rows)
        audio_sl = slice(cond_sl.stop, cond_sl.stop + audio_rows)
        video_sl = slice(audio_sl.stop, audio_sl.stop + video_rows)

        img_pos = torch.cat([torch.arange(cond_sl.start, cond_sl.stop), torch.arange(video_sl.start, video_sl.stop)])
        audio_pos = torch.arange(audio_sl.start, audio_sl.stop)

        g = torch.zeros(seq_len, 3, dtype=torch.float64)
        g[text_sl, 0] = torch.arange(text_len, dtype=torch.float64)

        sqrt_area = np.sqrt(latent_h * latent_w)
        frame, w_grid = self._frame_grid(latent_h, latent_w, sqrt_area)

        temporal_span = self._temporal_position_span(latent_t)
        for i, idx in enumerate(keyframe_indices):
            sl = slice(i * frame_rows, (i + 1) * frame_rows)
            if idx == 0:
                cond_t = float(text_len)
            else:
                cond_t = float(text_len) + temporal_span - _FRAME_RESCALE
            cond_g = torch.empty(frame_rows, 3, dtype=torch.float64)
            cond_g[:, 0] = cond_t
            cond_g[:, 1:] = frame
            g[cond_sl.start + sl.start : cond_sl.start + sl.stop] = cond_g

        g[video_sl] = self._video_grid(latent_t, frame, float(text_len))
        g[audio_sl, 0] = (float(text_len) + torch.arange(audio_t, dtype=torch.float64)).repeat(audio_channel)
        g[audio_sl, 2] = self._audio_w_axis(w_grid, audio_t, audio_rows)

        token_tags = torch.full((seq_len,), -1, dtype=torch.long)
        token_tags[text_sl] = 1
        token_tags[audio_sl] = 2
        token_tags[img_pos] = 0

        return {
            "img_pos": img_pos,
            "audio_pos": audio_pos,
            "text_pos": torch.arange(0, text_len),
            "img_position_ids": g[None],
            "token_tags": token_tags,
            "cu_seqlens": torch.tensor([0, used, seq_len], dtype=torch.int32),
            "seq_len": seq_len,
        }

    def build_packed_ref2va(
        self,
        text_len: int,
        latent_t: int,
        latent_h: int,
        latent_w: int,
        audio_t: int,
        ref_blocks: List[Dict[str, Any]],
        audio_channel: int = 2,
    ) -> Dict[str, Any]:
        """
        ref2va sequence layout: [text | ref_0 | ref_1 | ... | target_audio | target_video | pad]
        """
        ph, pw = latent_h // 2, latent_w // 2
        target_frame_rows = ph * pw
        target_video_rows = latent_t * target_frame_rows
        target_audio_rows = audio_t * audio_channel

        block_dims = []
        total_ref_visual_rows = 0
        total_ref_audio_rows = 0
        for b in ref_blocks:
            kind = b["kind"]
            info = {"kind": kind, "visual_rows": 0, "audio_rows": 0, "ref_audio_t": int(b.get("ref_audio_t", 0))}
            if kind in ("image", "video", "video_audio"):
                lt_r = int(b.get("latent_t", 1))
                lh_r, lw_r = int(b["latent_h"]), int(b["latent_w"])
                info.update(visual_rows=lt_r * (lh_r // 2) * (lw_r // 2), latent_t=lt_r, latent_h=lh_r, latent_w=lw_r)
                total_ref_visual_rows += info["visual_rows"]
                info["audio_rows"] = info["ref_audio_t"] * audio_channel
                total_ref_audio_rows += info["audio_rows"]
            elif kind == "audio":
                info["audio_rows"] = info["ref_audio_t"] * audio_channel
                total_ref_audio_rows += info["audio_rows"]
            else:
                raise ValueError(f"unknown ref kind: {kind}")
            block_dims.append(info)

        used = text_len + total_ref_visual_rows + total_ref_audio_rows + target_audio_rows + target_video_rows
        seq_len = self._aligned_seq_len(used)

        g = torch.zeros(seq_len, 3, dtype=torch.float64)
        g[0:text_len, 0] = torch.arange(text_len, dtype=torch.float64)
        token_tags = torch.full((seq_len,), -1, dtype=torch.long)
        token_tags[0:text_len] = 1

        cursor = text_len
        t_cursor = float(text_len)
        ref_visual_pos_parts = []
        ref_audio_pos_parts = []

        target_sqrt_area = float(np.sqrt(latent_h * latent_w))
        target_w_grid = self._axis_from_sqrt_area(latent_w, 2, target_sqrt_area)

        for info in block_dims:
            kind = info["kind"]
            if kind == "image":
                v_rows, lh_r, lw_r = info["visual_rows"], info["latent_h"], info["latent_w"]
                sqrt_area = float(np.sqrt(lh_r * lw_r))
                frame, _ = self._frame_grid(lh_r, lw_r, sqrt_area)
                sl = slice(cursor, cursor + v_rows)
                g[sl, 0] = t_cursor
                g[sl, 1:] = frame
                token_tags[sl] = 0
                ref_visual_pos_parts.append(torch.arange(sl.start, sl.stop))
                cursor += v_rows
                t_cursor += 1.0
            elif kind in ("video", "video_audio"):
                v_rows, lt_r, lh_r, lw_r = info["visual_rows"], info["latent_t"], info["latent_h"], info["latent_w"]
                sqrt_area = float(np.sqrt(lh_r * lw_r))
                frame, _ = self._frame_grid(lh_r, lw_r, sqrt_area)
                sl = slice(cursor, cursor + v_rows)
                g[sl] = self._video_grid(lt_r, frame, t_cursor)
                token_tags[sl] = 0
                ref_visual_pos_parts.append(torch.arange(sl.start, sl.stop))
                cursor += v_rows

                if info["audio_rows"] > 0:
                    a_rows = info["audio_rows"]
                    a_t = info["ref_audio_t"]
                    sl_a = slice(cursor, cursor + a_rows)
                    g[sl_a, 0] = (t_cursor + torch.arange(a_t, dtype=torch.float64)).repeat(audio_channel)
                    g[sl_a, 2] = self._audio_w_axis(target_w_grid, a_t, a_rows)
                    token_tags[sl_a] = 2
                    ref_audio_pos_parts.append(torch.arange(sl_a.start, sl_a.stop))
                    cursor += a_rows
                t_cursor += self._temporal_position_span(lt_r)
            elif kind == "audio":
                a_rows = info["audio_rows"]
                a_t = info["ref_audio_t"]
                sl_a = slice(cursor, cursor + a_rows)
                g[sl_a, 0] = (t_cursor + torch.arange(a_t, dtype=torch.float64)).repeat(audio_channel)
                g[sl_a, 2] = self._audio_w_axis(target_w_grid, a_t, a_rows)
                token_tags[sl_a] = 2
                ref_audio_pos_parts.append(torch.arange(sl_a.start, sl_a.stop))
                cursor += a_rows
                t_cursor += float(a_t)

        target_audio_sl = slice(cursor, cursor + target_audio_rows)
        target_video_sl = slice(target_audio_sl.stop, target_audio_sl.stop + target_video_rows)

        frame_t, w_grid_t = self._frame_grid(latent_h, latent_w, target_sqrt_area)
        g[target_video_sl] = self._video_grid(latent_t, frame_t, t_cursor)
        g[target_audio_sl, 0] = (t_cursor + torch.arange(audio_t, dtype=torch.float64)).repeat(audio_channel)
        g[target_audio_sl, 2] = self._audio_w_axis(w_grid_t, audio_t, target_audio_rows)

        target_video_pos = torch.arange(target_video_sl.start, target_video_sl.stop)
        target_audio_pos = torch.arange(target_audio_sl.start, target_audio_sl.stop)

        img_pos = torch.cat(ref_visual_pos_parts + [target_video_pos]) if ref_visual_pos_parts else target_video_pos
        audio_pos = torch.cat(ref_audio_pos_parts + [target_audio_pos]) if ref_audio_pos_parts else target_audio_pos

        token_tags[img_pos] = 0
        token_tags[audio_pos] = 2
        text_pos = torch.arange(0, text_len)
        cu = torch.tensor([0, used, seq_len], dtype=torch.int32)

        return {
            "img_pos": img_pos,
            "audio_pos": audio_pos,
            "text_pos": text_pos,
            "img_position_ids": g[None],
            "token_tags": token_tags,
            "cu_seqlens": cu,
            "seq_len": seq_len,
        }


__all__ = [
    "align_dimension",
    "align_video_frames",
    "PackedSequenceBuilder",
]
