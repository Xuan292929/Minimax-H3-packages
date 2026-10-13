# SPDX-License-Identifier: Apache-2.0
"""
Native PyTorch implementation of MiniMax-H3 DiT (Diffusion Transformer).
Style aligned with LTX-Video / LTX-2.5 clean modular pipeline.
Pure PyTorch implementation with zero dependency on sglang or diffusers.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

MINIMAX_H3_ADALN_MODALITY_NUM = 3
_PATCH_T, _PATCH_H, _PATCH_W = 1, 2, 2


def patchify_video(latent: torch.Tensor) -> torch.Tensor:
    """
    [1, 24, T, H, W] -> [T * (H/2) * (W/2), 96]
    """
    b, c, ft, fh, fw = (int(x) for x in latent.shape)
    t, h, w = ft // _PATCH_T, fh // _PATCH_H, fw // _PATCH_W
    packed = latent.reshape(b, c, t, _PATCH_T, h, _PATCH_H, w, _PATCH_W)
    packed = torch.einsum("nctrhpwq->nthwcrpq", packed)
    return packed.reshape(b * t * h * w, c * _PATCH_T * _PATCH_H * _PATCH_W).contiguous()


def unpatchify_video(rows: torch.Tensor, ft: int, fh: int, fw: int, channel: int = 24) -> torch.Tensor:
    """
    [T * (H/2) * (W/2), 96] -> [1, 24, T, H, W]
    """
    t, h, w = ft // _PATCH_T, fh // _PATCH_H, fw // _PATCH_W
    packed = rows.reshape(-1, t, h, w, channel, _PATCH_T, _PATCH_H, _PATCH_W)
    latent = torch.einsum("nthwcrpq->nctrhpwq", packed)
    return latent.reshape(-1, channel, ft, fh, fw).contiguous()


def pack_audio(latent: torch.Tensor) -> torch.Tensor:
    """
    [audio_channel, 32, T] -> [audio_channel * T, 32]
    """
    ac, ld, steps = (int(x) for x in latent.shape)
    return latent.permute(0, 2, 1).reshape(ac * steps, ld).contiguous()


def unpack_audio(rows: torch.Tensor, audio_channel: int, steps: int, latent_dim: int = 32) -> torch.Tensor:
    """
    [audio_channel * T, 32] -> [audio_channel, 32, T]
    """
    return rows.reshape(audio_channel, steps, latent_dim).permute(0, 2, 1).contiguous()


def _norm(size: int, *, eps: float) -> nn.RMSNorm:
    return nn.RMSNorm(size, eps=eps)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = torch.chunk(x, 2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def _apply_rope(x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    rot_dim = freqs.shape[-1]
    x_rot, x_pass = x[..., :rot_dim], x[..., rot_dim:]
    cos = torch.cos(freqs).to(x.dtype).unsqueeze(1)
    sin = torch.sin(freqs).to(x.dtype).unsqueeze(1)
    x_rot = (x_rot * cos) + (_rotate_half(x_rot) * sin)
    return torch.cat((x_rot, x_pass), dim=-1)


def _modulate_scale_shift(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    return (x * (1.0 + scale.index_select(0, indices)) + shift.index_select(0, indices)).to(x.dtype)


def _modulate_gate(x: torch.Tensor, gate: torch.Tensor, other: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    return (x + gate.index_select(0, indices) * other).to(x.dtype)


def _sdpa_varlen_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens: torch.Tensor,
    softmax_scale: float,
) -> torch.Tensor:
    out = torch.empty_like(q)
    bounds = cu_seqlens.tolist()
    for start, stop in zip(bounds[:-1], bounds[1:]):
        if stop == start:
            continue
        seg_q = q[start:stop].transpose(0, 1).unsqueeze(0)  # [1, H, S, D]
        seg_k = k[start:stop].transpose(0, 1).unsqueeze(0)  # [1, H, S, D]
        seg_v = v[start:stop].transpose(0, 1).unsqueeze(0)  # [1, H, S, D]
        seg_out = F.scaled_dot_product_attention(seg_q, seg_k, seg_v, scale=softmax_scale)
        out[start:stop] = seg_out.squeeze(0).transpose(0, 1)
    return out


class MiniMaxH3Rope(nn.Module):
    def __init__(self, inv_freq_len: int) -> None:
        super().__init__()
        self.inv_freq_len = inv_freq_len
        self.inv_freq = nn.Parameter(self._build_inv_freq())

    def _build_inv_freq(self, device: torch.device = None) -> torch.Tensor:
        steps = torch.arange(0, self.inv_freq_len, dtype=torch.float32, device=device)
        return 1.0 / (10000.0 ** (steps / self.inv_freq_len))

    def forward(self, img_position_ids: torch.Tensor) -> torch.Tensor:
        if img_position_ids.dim() != 3 or img_position_ids.shape[0] != 1:
            raise ValueError(f"img_position_ids must be [1, S, 3], got {list(img_position_ids.shape)}")
        pos = img_position_ids[0].to(torch.float32)
        inv_freq = self._build_inv_freq(pos.device)
        per_axis = pos.unsqueeze(-1) * inv_freq.view(1, 1, -1)
        t_f, h_f, w_f = per_axis.unbind(dim=1)
        half = torch.cat((t_f, h_f, w_f), dim=-1)
        return torch.cat((half, half), dim=-1)


class MiniMaxH3TimeEmbedder(nn.Module):
    def __init__(self, timestep_input_dim: int, time_embed_hidden_size: int, time_embed_dim: int) -> None:
        super().__init__()
        self.frequency_embedding_size = timestep_input_dim
        self.proj_in = nn.Linear(timestep_input_dim, time_embed_hidden_size, bias=True)
        self.proj_out = nn.Linear(time_embed_hidden_size, time_embed_dim, bias=True)

    def forward(self, t: torch.Tensor, *, dtype: torch.dtype) -> torch.Tensor:
        half = self.frequency_embedding_size // 2
        freqs = torch.exp(
            -math.log(10000.0) * torch.arange(half, dtype=torch.float32, device=t.device) / half
        )
        args = t.to(torch.float32)[:, None] * freqs[None]
        t_freq = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        hidden = self.proj_in(t_freq.to(dtype))
        hidden = F.silu(hidden)
        return self.proj_out(hidden)


class MiniMaxH3Attention(nn.Module):
    def __init__(self, hidden_size: int, num_attention_heads: int, attention_head_dim: int, qk_norm_eps: float) -> None:
        super().__init__()
        self.num_heads = num_attention_heads
        self.head_dim = attention_head_dim
        inner_dim = self.num_heads * self.head_dim
        self.softmax_scale = self.head_dim ** -0.5
        self.qkv_proj = nn.Linear(hidden_size, inner_dim * 3, bias=False)
        self.q_norm = _norm(attention_head_dim, eps=qk_norm_eps)
        self.k_norm = _norm(attention_head_dim, eps=qk_norm_eps)
        self.out_proj = nn.Linear(inner_dim, hidden_size, bias=False)

    def forward(self, x: torch.Tensor, *, rope_freqs: Optional[torch.Tensor], cu_seqlens: torch.Tensor, max_seqlen: Optional[int] = None) -> torch.Tensor:
        total = x.shape[0]
        qkv = self.qkv_proj(x)
        qkv = qkv.view(total, self.num_heads, 3, self.head_dim)
        q = qkv[:, :, 0, :]
        k = qkv[:, :, 1, :]
        v = qkv[:, :, 2, :]
        q = self.q_norm(q)
        k = self.k_norm(k)
        if rope_freqs is not None:
            q = _apply_rope(q, rope_freqs)
            k = _apply_rope(k, rope_freqs)
        out = _sdpa_varlen_attention(q, k, v, cu_seqlens=cu_seqlens, softmax_scale=self.softmax_scale)
        out = out.reshape(total, self.num_heads * self.head_dim)
        return self.out_proj(out)


class MiniMaxH3MLP(nn.Module):
    def __init__(self, hidden_size: int, ffn_hidden_size: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, ffn_hidden_size * 2, bias=False)
        self.fc2 = nn.Linear(ffn_hidden_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = self.fc1(x)
        gate, up = hidden.chunk(2, dim=-1)
        hidden = F.silu(gate) * up
        return self.fc2(hidden)


class MiniMaxH3AdalnProj(nn.Module):
    def __init__(self, hidden_size: int, time_embed_dim: int, out_features: int, *, expand_ratio: int, modality_num: int) -> None:
        super().__init__()
        if out_features != expand_ratio * hidden_size * modality_num:
            raise ValueError(
                f"adaln out_features mismatch: {out_features} != {expand_ratio}*{hidden_size}*{modality_num}"
            )
        self.expand_ratio = expand_ratio
        self.modality_num = modality_num
        self.hidden_size = hidden_size
        self.linear = nn.Linear(time_embed_dim, out_features, bias=True)

    def forward(self, t_emb: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        x = F.silu(t_emb)
        x = self.linear(x)
        m = x.shape[0]
        x = x.view(m * self.modality_num, self.expand_ratio * self.hidden_size)
        return tuple(x.chunk(self.expand_ratio, dim=-1))


class MiniMaxH3TokenRefinerBlock(nn.Module):
    def __init__(self, hidden_size: int, num_attention_heads: int, attention_head_dim: int, ffn_hidden_size: int, norm_eps: float, qk_norm_eps: float) -> None:
        super().__init__()
        self.norm1 = _norm(hidden_size, eps=norm_eps)
        self.norm2 = _norm(hidden_size, eps=norm_eps)
        self.attn = MiniMaxH3Attention(hidden_size, num_attention_heads, attention_head_dim, qk_norm_eps)
        self.mlp = MiniMaxH3MLP(hidden_size, ffn_hidden_size)

    def forward(self, x: torch.Tensor, *, cu_seqlens: torch.Tensor, max_seqlen: int) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), rope_freqs=None, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen)
        x = x + self.mlp(self.norm2(x))
        return x


class MiniMaxH3TokenRefiner(nn.Module):
    def __init__(
        self,
        num_layers: int,
        hidden_size: int,
        num_attention_heads: int,
        attention_head_dim: int,
        ffn_hidden_size: int,
        norm_eps: float,
        qk_norm_eps: float,
        final_norm_eps: float,
    ) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            [MiniMaxH3TokenRefinerBlock(hidden_size, num_attention_heads, attention_head_dim, ffn_hidden_size, norm_eps, qk_norm_eps) for _ in range(num_layers)]
        )
        self.final_norm = _norm(hidden_size, eps=final_norm_eps)

    def forward(self, x: torch.Tensor, *, cu_seqlens: torch.Tensor, max_seqlen: int) -> torch.Tensor:
        for block in self.blocks:
            x = block(x, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen)
        return self.final_norm(x)


class MiniMaxH3DiTBlock(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        attention_head_dim: int,
        ffn_hidden_size: int,
        time_embed_dim: int,
        adaln_out_features: int,
        norm_eps: float,
        qk_norm_eps: float,
    ) -> None:
        super().__init__()
        self.norm1 = _norm(hidden_size, eps=norm_eps)
        self.norm2 = _norm(hidden_size, eps=norm_eps)
        self.attn = MiniMaxH3Attention(hidden_size, num_attention_heads, attention_head_dim, qk_norm_eps)
        self.mlp = MiniMaxH3MLP(hidden_size, ffn_hidden_size)
        self.adaln_proj = MiniMaxH3AdalnProj(hidden_size, time_embed_dim, adaln_out_features, expand_ratio=6, modality_num=MINIMAX_H3_ADALN_MODALITY_NUM)

    def forward(
        self,
        x: torch.Tensor,
        *,
        t_emb: torch.Tensor,
        combined_indices: torch.Tensor,
        rope_freqs: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: Optional[int] = None,
    ) -> torch.Tensor:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaln_proj(t_emb)
        residual = x
        h = self.norm1(x)
        h = _modulate_scale_shift(h, shift_msa, scale_msa, combined_indices)
        h = self.attn(h, rope_freqs=rope_freqs, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen)
        x = _modulate_gate(residual, gate_msa, h, combined_indices)
        residual = x
        h = self.norm2(x)
        h = _modulate_scale_shift(h, shift_mlp, scale_mlp, combined_indices)
        h = self.mlp(h)
        return _modulate_gate(residual, gate_mlp, h, combined_indices)


class MiniMaxH3FinalLayer(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        time_embed_dim: int,
        final_adaln_out_features: int,
        latents_dim: int,
        audio_latents_dim: int,
        patch_size: Tuple[int, int, int],
        final_norm_eps: float,
    ) -> None:
        super().__init__()
        video_patch_dim = latents_dim * patch_size[0] * patch_size[1] * patch_size[2]
        self.norm = _norm(hidden_size, eps=final_norm_eps)
        self.adaln_proj = MiniMaxH3AdalnProj(hidden_size, time_embed_dim, final_adaln_out_features, expand_ratio=2, modality_num=1)
        self.video_out = nn.Linear(hidden_size, video_patch_dim, bias=True)
        self.audio_out = nn.Linear(hidden_size, audio_latents_dim, bias=True)

    def forward(self, x: torch.Tensor, *, t_emb: torch.Tensor, inverse_indices: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        shift, scale = self.adaln_proj(t_emb)
        h = self.norm(x)
        h = _modulate_scale_shift(h, shift, scale, inverse_indices)
        video = self.video_out(h)
        audio = self.audio_out(h)
        return video, audio


class MiniMaxH3DiTModel(nn.Module):
    """
    Pure PyTorch implementation of MiniMax-H3 DiT Model.
    Supports packed sequence video + audio joint generation.
    """

    def __init__(
        self,
        num_layers: int = 50,
        token_refiner_num_layers: int = 2,
        hidden_size: int = 5376,
        num_attention_heads: int = 56,
        attention_head_dim: int = 128,
        ffn_hidden_size: int = 14336,
        latents_dim: int = 24,
        audio_latents_dim: int = 32,
        patch_size: Tuple[int, int, int] = (1, 2, 2),
        text_dim: int = 5120,
        timestep_input_dim: int = 256,
        time_embed_hidden_size: int = 5376,
        time_embed_dim: int = 2688,
        adaln_out_features: int = 96768,
        final_adaln_out_features: int = 10752,
        rope_inv_freq_len: int = 16,
        norm_eps: float = 1e-5,
        qk_norm_eps: float = 1e-5,
        final_norm_eps: float = 1e-5,
        **kwargs,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.num_channels_latents = latents_dim
        self.audio_latents_dim = audio_latents_dim
        video_patch_dim = latents_dim * patch_size[0] * patch_size[1] * patch_size[2]

        self.video_patch_proj = nn.Linear(video_patch_dim, hidden_size, bias=True)
        self.audio_patch_proj = nn.Linear(audio_latents_dim, hidden_size, bias=True)
        self.condition_proj = nn.Linear(text_dim, hidden_size, bias=True)
        self.time_embedder = MiniMaxH3TimeEmbedder(timestep_input_dim, time_embed_hidden_size, time_embed_dim)
        self.rope = MiniMaxH3Rope(rope_inv_freq_len)
        self.token_refiner = MiniMaxH3TokenRefiner(
            token_refiner_num_layers,
            hidden_size,
            num_attention_heads,
            attention_head_dim,
            ffn_hidden_size,
            norm_eps,
            qk_norm_eps,
            final_norm_eps,
        )
        self.blocks = nn.ModuleList(
            [
                MiniMaxH3DiTBlock(
                    hidden_size,
                    num_attention_heads,
                    attention_head_dim,
                    ffn_hidden_size,
                    time_embed_dim,
                    adaln_out_features,
                    norm_eps,
                    qk_norm_eps,
                )
                for _ in range(num_layers)
            ]
        )
        self.final_layer = MiniMaxH3FinalLayer(
            hidden_size,
            time_embed_dim,
            final_adaln_out_features,
            latents_dim,
            audio_latents_dim,
            patch_size,
            final_norm_eps,
        )

    def _embed(
        self,
        *,
        x: torch.Tensor,
        audio_x: torch.Tensor,
        text_embeddings_selected: torch.Tensor,
        unique_timesteps: torch.Tensor,
        img_pos: torch.Tensor,
        audio_pos: torch.Tensor,
        text_pos: torch.Tensor,
        refiner_cu_seqlens: torch.Tensor,
        refiner_max_seqlen: int,
        seq_len: int,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        dtype = text_embeddings_selected.dtype
        x_rows = x.view(-1, x.shape[-1]).index_select(0, img_pos).to(dtype)
        video_embed = self.video_patch_proj(x_rows)
        audio_rows = audio_x.view(-1, audio_x.shape[-1]).index_select(0, audio_pos).to(dtype)
        audio_embed = self.audio_patch_proj(audio_rows)
        text_rows = text_embeddings_selected.to(device=device)
        text_embed = self.condition_proj(text_rows)
        text_embed = self.token_refiner(text_embed, cu_seqlens=refiner_cu_seqlens, max_seqlen=refiner_max_seqlen)

        embeddings = torch.zeros((seq_len, self.hidden_size), device=device, dtype=dtype)
        embeddings.index_add_(0, text_pos, text_embed.to(dtype)[: text_pos.shape[0]])
        embeddings.index_add_(0, img_pos, video_embed.to(dtype)[: img_pos.shape[0]])
        embeddings.index_add_(0, audio_pos, audio_embed.to(dtype)[: audio_pos.shape[0]])

        t_emb = self.time_embedder(unique_timesteps, dtype=dtype)
        return embeddings, t_emb

    def forward(
        self,
        x: torch.Tensor,
        audio_x: torch.Tensor,
        img_position_ids: torch.Tensor,
        unique_timesteps: torch.Tensor,
        inverse_indices: torch.Tensor,
        token_tags: torch.Tensor,
        prompt_embeds: torch.Tensor,
        img_pos_info: Dict[str, torch.Tensor],
        audio_pos_info: Dict[str, torch.Tensor],
        text_pos_info: Dict[str, torch.Tensor],
        img_pos_for_infer_output_info: Dict[str, torch.Tensor],
        packed_seq_params: Dict[str, Any],
        refiner_packed_seq_params: Dict[str, Any],
        use_gradient_checkpointing: bool = False,
        update_mask: Optional[torch.Tensor] = None,
        update_audio_mask: Optional[torch.Tensor] = None,
        skip_mask_out_condition: bool = False,
        control_hints: Optional[Dict[int, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        inverse_indices = inverse_indices.view(-1).to(torch.long)
        token_tags = token_tags.view(-1).to(torch.long)
        text_selected = prompt_embeds

        img_pos = img_pos_info["position_ids"].view(-1).to(torch.long)
        audio_pos = audio_pos_info["position_ids"].view(-1).to(torch.long)
        text_pos = text_pos_info["position_ids"].view(-1).to(torch.long)
        infer_out_pos = img_pos_for_infer_output_info["position_ids"].view(-1).to(torch.long)

        cu_seqlens = packed_seq_params["cu_seqlens_q"].to(torch.int32)
        max_seqlen = int(packed_seq_params["max_seqlen_q"])
        refiner_cu = refiner_packed_seq_params["cu_seqlens_q"].to(torch.int32)
        refiner_max = int(refiner_packed_seq_params["max_seqlen_q"])

        if x.dim() != 3 or x.shape[0] != 1:
            raise ValueError(f"x must be [1, S, C], got {list(x.shape)}")
        seq_len = int(x.shape[1])
        device = x.device

        rope_freqs = self.rope(img_position_ids).to(device)

        decoder_input, t_emb = self._embed(
            x=x,
            audio_x=audio_x,
            text_embeddings_selected=text_selected,
            unique_timesteps=unique_timesteps.view(-1).to(device),
            img_pos=img_pos.to(device),
            audio_pos=audio_pos.to(device),
            text_pos=text_pos.to(device),
            refiner_cu_seqlens=refiner_cu.to(device),
            refiner_max_seqlen=refiner_max,
            seq_len=seq_len,
            device=device,
        )

        combined_indices = (inverse_indices * MINIMAX_H3_ADALN_MODALITY_NUM + token_tags.clamp(min=0)).to(device)
        inverse_indices = inverse_indices.to(device)

        hidden = decoder_input
        cu_seqlens = cu_seqlens.to(device)

        for block_id, block in enumerate(self.blocks):
            if use_gradient_checkpointing and self.training:
                hidden = torch.utils.checkpoint.checkpoint(
                    block,
                    hidden,
                    t_emb=t_emb,
                    combined_indices=combined_indices,
                    rope_freqs=rope_freqs,
                    cu_seqlens=cu_seqlens,
                    max_seqlen=max_seqlen,
                    use_reentrant=False,
                )
            else:
                hidden = block(
                    hidden,
                    t_emb=t_emb,
                    combined_indices=combined_indices,
                    rope_freqs=rope_freqs,
                    cu_seqlens=cu_seqlens,
                    max_seqlen=max_seqlen,
                )
            if control_hints is not None and block_id in control_hints:
                hidden = hidden + control_hints[block_id].to(hidden.device, hidden.dtype)

        video_logits, audio_logits = self.final_layer(hidden, t_emb=t_emb, inverse_indices=inverse_indices)

        video_logits = video_logits.index_select(0, infer_out_pos.to(device))
        audio_logits = audio_logits.index_select(0, audio_pos.to(device))

        if not skip_mask_out_condition:
            if update_mask is not None:
                update_mask = update_mask.view(-1).to(device)
                if update_mask.shape[0] != video_logits.shape[0]:
                    raise ValueError(f"update_mask length mismatch: {update_mask.shape[0]} != {video_logits.shape[0]}")
                video_logits = video_logits * update_mask.unsqueeze(-1)
            if update_audio_mask is not None:
                audio_logits = audio_logits * update_audio_mask.view(-1).unsqueeze(-1)

        return video_logits, audio_logits


class BlockFP8Linear(nn.Module):
    """
    128x128 Block-scaled FP8 Linear layer for MiniMax-H3 DiT.
    Native PyTorch module without third-party dependencies.
    Holds weight in torch.float8_e4m3fn and weight_scale_inv in torch.float32.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        device: Optional[Union[str, torch.device]] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.register_buffer("weight", torch.empty((out_features, in_features), dtype=torch.float8_e4m3fn, device=device))
        s_rows = (out_features + 127) // 128
        s_cols = (in_features + 127) // 128
        self.register_buffer("weight_scale_inv", torch.empty((s_rows, s_cols), dtype=torch.float32, device=device))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, dtype=torch.bfloat16, device=device))
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out_dim, in_dim = self.weight.shape
        scale_exp = self.weight_scale_inv.repeat_interleave(128, dim=0)[:out_dim, :].repeat_interleave(128, dim=1)[:, :in_dim]
        w = (self.weight.to(torch.float32) * scale_exp.to(torch.float32)).to(x.dtype)
        return F.linear(x, w, self.bias)


def convert_linear_to_block_fp8(model: nn.Module, target_submodules: Iterable[str]) -> None:
    """
    Convert named linear submodules in `model` to BlockFP8Linear.
    `target_submodules` contains paths like 'blocks.0.attn.qkv_proj'.
    """
    for full_name in target_submodules:
        parts = full_name.split(".")
        parent = model
        for p in parts[:-1]:
            if hasattr(parent, p):
                parent = getattr(parent, p)
            elif p.isdigit() and isinstance(parent, (nn.ModuleList, list)):
                parent = parent[int(p)]
            else:
                parent = None
                break
        if parent is None:
            continue
        attr = parts[-1]
        if hasattr(parent, attr):
            old_layer = getattr(parent, attr)
            if isinstance(old_layer, nn.Linear):
                has_bias = old_layer.bias is not None
                new_layer = BlockFP8Linear(old_layer.in_features, old_layer.out_features, bias=has_bias)
                setattr(parent, attr, new_layer)


__all__ = [
    "MiniMaxH3DiTModel",
    "BlockFP8Linear",
    "convert_linear_to_block_fp8",
    "patchify_video",
    "unpatchify_video",
    "pack_audio",
    "unpack_audio",
]

