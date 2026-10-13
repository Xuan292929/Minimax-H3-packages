# SPDX-License-Identifier: Apache-2.0
"""
Native PyTorch generation pipeline for MiniMax-H3.
Design reference: LTX-Video / LTX-2.5 clean modular pipeline.
Pure PyTorch implementation with zero dependency on sglang or diffusers.
"""

from __future__ import annotations

import gc
import json
import os
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from safetensors.torch import load_file
from tqdm import tqdm
from transformers import AutoProcessor, AutoTokenizer

from ..models.audio_vae import MiniMaxH3AudioVAE
from ..models.text_encoder import (
    MiniMaxH3TextEncoder,
    image_token_counts,
    presentation_fl2va,
    presentation_ref2va,
    presentation_t2va,
    video_token_counts,
)
from ..models.transformer import (
    MiniMaxH3DiTModel,
    pack_audio,
    patchify_video,
    unpack_audio,
    unpatchify_video,
)
from ..models.video_vae import MiniMaxH3VideoVAE
from ..schedulers.flow_match import MiniMaxH3FlowMatchScheduler
from ..utils.media_io import convert_to_stereo, resample_waveform, save_video_audio
from ..utils.packed_sequence import PackedSequenceBuilder, align_dimension, align_video_frames


def _load_sharded_safetensors(
    model_dir: str,
    device: str = "cpu",
    dtype: torch.dtype = torch.bfloat16,
    key_filter: Optional[Callable[[str], bool]] = None,
) -> Dict[str, torch.Tensor]:
    index_file = os.path.join(model_dir, "model.safetensors.index.json")
    state_dict = {}
    if os.path.exists(index_file):
        with open(index_file, "r") as f:
            idx = json.load(f)
        weight_map = idx.get("weight_map", {})
        if key_filter is not None:
            shards_needed = set()
            for k, shard in weight_map.items():
                if key_filter(k):
                    shards_needed.add(shard)
            shard_files = sorted(list(shards_needed))
        else:
            shard_files = sorted(list(set(weight_map.values())))

        for shard in shard_files:
            shard_path = os.path.join(model_dir, shard)
            shard_dict = load_file(shard_path, device=device)
            for k, v in shard_dict.items():
                if key_filter is None or key_filter(k):
                    state_dict[k] = v.to(device=device, dtype=dtype)
    else:
        single_file = os.path.join(model_dir, "model.safetensors")
        if os.path.exists(single_file):
            shard_dict = load_file(single_file, device=device)
            for k, v in shard_dict.items():
                if key_filter is None or key_filter(k):
                    state_dict[k] = v.to(device=device, dtype=dtype)
        else:
            st_files = sorted([f for f in os.listdir(model_dir) if f.endswith(".safetensors")])
            for st in st_files:
                shard_dict = load_file(os.path.join(model_dir, st), device=device)
                for k, v in shard_dict.items():
                    if key_filter is None or key_filter(k):
                        state_dict[k] = v.to(device=device, dtype=dtype)
    return state_dict


class MiniMaxH3Pipeline:
    """
    Modular PyTorch audio-video generation pipeline for MiniMax-H3.
    """

    def __init__(
        self,
        text_encoder: Optional[MiniMaxH3TextEncoder] = None,
        dit: Optional[MiniMaxH3DiTModel] = None,
        video_vae: Optional[MiniMaxH3VideoVAE] = None,
        audio_vae: Optional[MiniMaxH3AudioVAE] = None,
        tokenizer: Optional[Any] = None,
        processor: Optional[Any] = None,
        scheduler: Optional[MiniMaxH3FlowMatchScheduler] = None,
        scheduler_audio: Optional[MiniMaxH3FlowMatchScheduler] = None,
        device: Union[str, torch.device] = "cuda",
        torch_dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        self.device = torch.device(device)
        self.torch_dtype = torch_dtype
        self.text_encoder = text_encoder
        self.dit = dit
        self.video_vae = video_vae
        self.audio_vae = audio_vae
        self.tokenizer = tokenizer
        self.processor = processor
        self.scheduler = scheduler or MiniMaxH3FlowMatchScheduler()
        self.scheduler_audio = scheduler_audio or MiniMaxH3FlowMatchScheduler()
        self.packed_builder = PackedSequenceBuilder()

        # Sequential GPU offload state
        self._cpu_offload_enabled = False

    def enable_sequential_cpu_offload(self) -> None:
        """
        Enables sequential loading of models onto GPU one at a time,
        offloading inactive models to CPU. Reduces peak VRAM to ~65GB.
        """
        self._cpu_offload_enabled = True
        for m in (self.text_encoder, self.dit, self.video_vae, self.audio_vae):
            if m is not None:
                m.to("cpu")
        torch.cuda.empty_cache()

    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        torch_dtype: torch.dtype = torch.bfloat16,
        device: Union[str, torch.device] = "cuda",
        enable_cpu_offload: bool = True,
    ) -> "MiniMaxH3Pipeline":
        """
        Load MiniMax-H3 pipeline from pretrained checkpoint directory.
        `model_path` should contain subfolders: text_encoder, transformer, video_vae, audio_vae, processor.
        """
        print(f"[MiniMaxH3Pipeline] Loading models from {model_path} with dtype {torch_dtype}...")

        # 1. Processor & Tokenizer
        processor_path = os.path.join(model_path, "processor")
        if not os.path.exists(processor_path):
            processor_path = os.path.join(model_path, "text_encoder")
        processor = AutoProcessor.from_pretrained(processor_path)
        tokenizer = processor.tokenizer

        # 2. Text Encoder (Qwen3-VL 50 layers)
        print("  - Initializing Text Encoder...")
        with torch.device("meta"):
            text_encoder = MiniMaxH3TextEncoder(num_retained_layers=50)

        # Re-register rotary embeddings on CPU to eliminate non-persistent meta buffers
        from transformers.models.qwen3_vl.modeling_qwen3_vl import (
            Qwen3VLTextRotaryEmbedding,
            Qwen3VLVisionRotaryEmbedding,
        )
        load_device = "cpu" if enable_cpu_offload else device

        # Re-register rotary embeddings on load_device to eliminate non-persistent meta buffers
        from transformers.models.qwen3_vl.modeling_qwen3_vl import (
            Qwen3VLTextRotaryEmbedding,
            Qwen3VLVisionRotaryEmbedding,
        )
        text_encoder.model.language_model.rotary_emb = Qwen3VLTextRotaryEmbedding(config=text_encoder.config.text_config, device=load_device)
        text_encoder.model.visual.rotary_pos_emb = Qwen3VLVisionRotaryEmbedding(config=text_encoder.config.vision_config)

        def _te_filter(k: str) -> bool:
            if k == "lm_head.weight" or k.startswith("model.language_model.norm."):
                return False
            if k.startswith("model.language_model.layers."):
                l_idx = int(k[len("model.language_model.layers."):].split(".", 1)[0])
                if l_idx >= 50:
                    return False
            return True

        te_path = os.path.join(model_path, "text_encoder")
        clean_te_sd = _load_sharded_safetensors(te_path, device=load_device, dtype=torch_dtype, key_filter=_te_filter)
        missing, unexpected = text_encoder.load_state_dict(clean_te_sd, strict=False, assign=True)
        print(f"    Text Encoder loaded. (missing: {len(missing)}, unexpected: {len(unexpected)})")
        del clean_te_sd
        gc.collect()

        # 3. DiT Transformer
        print("  - Initializing DiT Transformer...")
        with torch.device("meta"):
            dit = MiniMaxH3DiTModel()
        tr_path = os.path.join(model_path, "transformer")
        tr_sd = _load_sharded_safetensors(tr_path, device=load_device, dtype=torch_dtype)
        clean_tr_sd = {}
        for k, v in tr_sd.items():
            k_clean = k.replace("model.diffusion_model.", "").replace("transformer.", "")
            clean_tr_sd[k_clean] = v
        missing, unexpected = dit.load_state_dict(clean_tr_sd, strict=False, assign=True)
        print(f"    DiT Transformer loaded. (missing: {len(missing)}, unexpected: {len(unexpected)})")
        del tr_sd, clean_tr_sd
        gc.collect()

        # 4. Video VAE
        print("  - Initializing Video VAE...")
        with torch.device("meta"):
            video_vae = MiniMaxH3VideoVAE()
        vvae_path = os.path.join(model_path, "video_vae", "source")
        if not os.path.exists(vvae_path):
            vvae_path = os.path.join(model_path, "video_vae")
        vvae_sd = _load_sharded_safetensors(vvae_path, device=load_device, dtype=torch_dtype)
        clean_vvae_sd = {k.replace("video_vae.", ""): v for k, v in vvae_sd.items()}
        missing, unexpected = video_vae.load_state_dict(clean_vvae_sd, strict=False, assign=True)
        print(f"    Video VAE loaded. (missing: {len(missing)}, unexpected: {len(unexpected)})")
        del vvae_sd, clean_vvae_sd
        gc.collect()

        # 5. Audio VAE
        print("  - Initializing Audio VAE...")
        with torch.device("meta"):
            audio_vae = MiniMaxH3AudioVAE()
        avae_path = os.path.join(model_path, "audio_vae")
        avae_sd = _load_sharded_safetensors(avae_path, device=load_device, dtype=torch_dtype)
        clean_avae_sd = {k.replace("audio_vae.", ""): v for k, v in avae_sd.items()}
        missing, unexpected = audio_vae.load_state_dict(clean_avae_sd, strict=False, assign=True)
        print(f"    Audio VAE loaded. (missing: {len(missing)}, unexpected: {len(unexpected)})")
        del avae_sd, clean_avae_sd
        gc.collect()

        pipeline = cls(
            text_encoder=text_encoder,
            dit=dit,
            video_vae=video_vae,
            audio_vae=audio_vae,
            tokenizer=tokenizer,
            processor=processor,
            device=device,
            torch_dtype=torch_dtype,
        )

        if enable_cpu_offload:
            pipeline.enable_sequential_cpu_offload()
        else:
            for m in (pipeline.text_encoder, pipeline.dit, pipeline.video_vae, pipeline.audio_vae):
                if m is not None:
                    m.to(device=device, dtype=torch_dtype)

        print("[MiniMaxH3Pipeline] Pipeline initialization complete.")
        return pipeline

    @torch.no_grad()
    def encode_prompt(
        self,
        prompt: str,
        keyframes: Optional[List[Image.Image]] = None,
        height: int = 480,
        width: int = 832,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode prompt into text embeddings and token tags.
        """
        if self._cpu_offload_enabled:
            self.text_encoder.to(self.device, dtype=self.torch_dtype)

        pixel_values = image_grid_thw = None
        if keyframes:
            resized_kf = [img.convert("RGB").resize((width, height), Image.LANCZOS) for img in keyframes]
            pixel_values, image_grid_thw, image_counts = image_token_counts(self.processor, resized_kf)
            input_ids, text_token_tags = presentation_fl2va(self.tokenizer, prompt, image_counts)
        else:
            input_ids, text_token_tags = presentation_t2va(self.tokenizer, prompt)

        ids = input_ids.unsqueeze(0).to(self.device)
        kwargs = {"input_ids": ids, "attention_mask": torch.ones_like(ids)}
        if pixel_values is not None:
            kwargs["pixel_values"] = pixel_values.to(self.device, self.torch_dtype)
            kwargs["image_grid_thw"] = image_grid_thw.to(self.device, torch.long)

        hidden = self.text_encoder(**kwargs)
        prompt_embeds = hidden.to(self.device, self.torch_dtype)
        text_token_tags = text_token_tags.view(-1).to(self.device, torch.long)

        if self._cpu_offload_enabled:
            self.text_encoder.to("cpu")
            torch.cuda.empty_cache()

        return prompt_embeds, text_token_tags

    def generate_noise(
        self,
        height: int,
        width: int,
        num_frames: int,
        seed: int = 42,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        generator = torch.Generator(device="cpu").manual_seed(seed)
        video_latent_t = ((num_frames - 5) // 17) * 5 + 2
        latent_h = height // 16
        latent_w = width // 16
        video_latents = torch.randn(
            (1, 24, video_latent_t, latent_h, latent_w),
            generator=generator,
            dtype=self.torch_dtype,
        ).to(self.device)

        audio_latent_t = round(num_frames / 24.0 * 40.0)
        audio_latents = torch.randn(
            (2, 32, audio_latent_t),
            generator=generator,
            dtype=self.torch_dtype,
        ).to(self.device)

        return video_latents, audio_latents

    @torch.no_grad()
    def denoise_loop(
        self,
        video_latents: torch.Tensor,
        audio_latents: torch.Tensor,
        prompt_embeds: torch.Tensor,
        text_token_tags: torch.Tensor,
        num_inference_steps: int = 50,
        flow_shift: float = 6.0,
        audio_flow_shift: float = 3.0,
        keyframe_cond_anchor: Optional[torch.Tensor] = None,
        keyframe_indices: Optional[List[int]] = None,
        progress_bar_cmd: Callable = tqdm,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self._cpu_offload_enabled:
            self.dit.to(self.device, dtype=self.torch_dtype)

        self.scheduler.set_timesteps(num_inference_steps, shift=flow_shift)
        self.scheduler_audio.set_timesteps(num_inference_steps, shift=audio_flow_shift)

        # Build packed sequence structures
        text_len = prompt_embeds.shape[0]
        video_latent_t, latent_h, latent_w = video_latents.shape[2:]
        audio_latent_t = audio_latents.shape[-1]
        kf_idx = keyframe_indices if keyframe_cond_anchor is not None else []
        packed = self.packed_builder.build_packed_fl2va(
            text_len, video_latent_t, latent_h, latent_w, audio_latent_t, kf_idx
        )
        packed["token_tags"][packed["text_pos"]] = text_token_tags.cpu()
        packed = {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in packed.items()}

        dtype, device = video_latents.dtype, video_latents.device
        f, h, w = video_latents.shape[2:]
        audio_channel = audio_latents.shape[0]

        img_pos, audio_pos, text_pos = packed["img_pos"], packed["audio_pos"], packed["text_pos"]
        cu, seq_len = packed["cu_seqlens"], packed["seq_len"]

        cond_anchor = keyframe_cond_anchor
        cond_rows_count = 0 if cond_anchor is None else cond_anchor.shape[0]
        refiner_cu = torch.tensor([0, text_len, text_len], dtype=torch.int32, device=device)

        for step_idx, timestep_video_val in enumerate(progress_bar_cmd(self.scheduler.timesteps, desc="MiniMax-H3 Denoising")):
            t_video = 1.0 - float(timestep_video_val) / 1000.0
            timestep_audio_val = self.scheduler_audio.timesteps[step_idx]
            t_audio = 1.0 - float(timestep_audio_val) / 1000.0

            video_rows = patchify_video(video_latents)
            audio_rows = pack_audio(audio_latents)

            x = torch.zeros(1, seq_len, 96, dtype=dtype, device=device)
            x[0].index_copy_(0, img_pos[cond_rows_count:], video_rows)
            if cond_anchor is not None:
                x[0].index_copy_(0, img_pos[:cond_rows_count], cond_anchor)

            audio_x = torch.zeros(1, seq_len, 32, dtype=dtype, device=device)
            audio_x[0].index_copy_(0, audio_pos, audio_rows)

            timesteps = torch.full((seq_len,), t_video, dtype=torch.float32, device=device)
            timesteps[audio_pos] = t_audio
            if cond_anchor is not None and cond_rows_count > 0:
                timesteps[img_pos[:cond_rows_count]] = max(t_video, 0.999)

            unique_timesteps, inverse_indices = torch.unique(timesteps, sorted=True, return_inverse=True)

            v_video_rows, v_audio_rows = self.dit(
                x=x,
                audio_x=audio_x,
                img_position_ids=packed["img_position_ids"],
                unique_timesteps=unique_timesteps,
                inverse_indices=inverse_indices,
                token_tags=packed["token_tags"],
                prompt_embeds=prompt_embeds,
                img_pos_info={"position_ids": img_pos},
                audio_pos_info={"position_ids": audio_pos},
                text_pos_info={"position_ids": text_pos},
                img_pos_for_infer_output_info={"position_ids": img_pos},
                packed_seq_params={"cu_seqlens_q": cu, "max_seqlen_q": int(cu[1])},
                refiner_packed_seq_params={"cu_seqlens_q": refiner_cu, "max_seqlen_q": text_len},
                skip_mask_out_condition=True,
            )

            v_video_rows = v_video_rows[cond_rows_count:]
            v_video = unpatchify_video(v_video_rows, f, h, w)
            v_audio = unpack_audio(v_audio_rows, audio_channel, audio_latent_t)

            # MiniMax-H3 Euler-eta0 ODE step:
            # denoised = x + sigma_curr * v
            # x_next = ratio * x + (1.0 - ratio) * denoised
            sigma_v_curr = self.scheduler.sigmas[step_idx].to(device=device, dtype=dtype)
            denoised_v = video_latents + sigma_v_curr * v_video
            if step_idx + 1 < len(self.scheduler.sigmas):
                sigma_v_next = self.scheduler.sigmas[step_idx + 1].to(device=device, dtype=dtype)
                ratio_v = sigma_v_next / (sigma_v_curr + 1e-8)
                video_latents = ratio_v * video_latents + (1.0 - ratio_v) * denoised_v
            else:
                video_latents = denoised_v

            sigma_a_curr = self.scheduler_audio.sigmas[step_idx].to(device=device, dtype=dtype)
            denoised_a = audio_latents + sigma_a_curr * v_audio
            if step_idx + 1 < len(self.scheduler_audio.sigmas):
                sigma_a_next = self.scheduler_audio.sigmas[step_idx + 1].to(device=device, dtype=dtype)
                ratio_a = sigma_a_next / (sigma_a_curr + 1e-8)
                audio_latents = ratio_a * audio_latents + (1.0 - ratio_a) * denoised_a
            else:
                audio_latents = denoised_a

        if self._cpu_offload_enabled:
            self.dit.to("cpu")
            torch.cuda.empty_cache()

        return video_latents, audio_latents

    @torch.no_grad()
    def decode_latents(
        self,
        video_latents: torch.Tensor,
        audio_latents: torch.Tensor,
        tiled: bool = True,
        tile_size: int = 256,
        tile_overlap: int = 64,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # 1. Decode Video
        if self._cpu_offload_enabled:
            self.video_vae.to(self.device, dtype=self.torch_dtype)
        frames = self.video_vae.decode_video(
            video_latents,
            dtype=self.torch_dtype,
            tiled=tiled,
            tile_size=tile_size,
            tile_overlap=tile_overlap,
        )
        if self._cpu_offload_enabled:
            self.video_vae.to("cpu")
            torch.cuda.empty_cache()

        # 2. Decode Audio
        orig_dtype = next(self.audio_vae.parameters()).dtype
        orig_device = next(self.audio_vae.parameters()).device
        self.audio_vae.to(self.device, dtype=torch.float32)
        waveform = self.audio_vae.decode_audio(audio_latents, dtype=torch.float32)
        waveform = convert_to_stereo(waveform.float().cpu())
        if self._cpu_offload_enabled:
            self.audio_vae.to("cpu")
            torch.cuda.empty_cache()
        else:
            self.audio_vae.to(orig_device, dtype=orig_dtype)

        return frames, waveform

    @torch.no_grad()
    def __call__(
        self,
        prompt: str,
        height: int = 480,
        width: int = 832,
        num_frames: int = 124,
        num_inference_steps: int = 50,
        seed: int = 42,
        flow_shift: float = 6.0,
        audio_flow_shift: float = 3.0,
        keyframes: Optional[List[Image.Image]] = None,
        keyframe_indices: Optional[List[int]] = None,
        tiled: bool = True,
        tile_size: int = 256,
        tile_overlap: int = 64,
        progress_bar_cmd: Callable = tqdm,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate synchronized video and audio from text prompt (and optional keyframes).
        """
        # Align spatial & temporal dimensions
        height = align_dimension(height, 32)
        width = align_dimension(width, 32)
        num_frames = align_video_frames(num_frames, 17, 5)

        print(f"[MiniMaxH3Pipeline] Generating {width}x{height}, {num_frames} frames @ 24fps ({num_inference_steps} steps, seed={seed})...")

        # 1. Text & Keyframe Encoding
        prompt_embeds, text_token_tags = self.encode_prompt(
            prompt=prompt,
            keyframes=keyframes,
            height=height,
            width=width,
        )

        # 2. Keyframe Latent Anchor
        keyframe_cond_anchor = None
        if keyframes is not None and keyframe_indices is not None:
            if self._cpu_offload_enabled:
                self.video_vae.to(self.device, dtype=self.torch_dtype)
            all_cond_rows = []
            for img in keyframes:
                img_tensor = img.convert("RGB").resize((width, height), Image.LANCZOS)
                img_t = torch.from_numpy(np.array(img_tensor)).permute(2, 0, 1).float().unsqueeze(0) / 255.0
                z_norm = self.video_vae.encode_video(img_t.to(self.device), dtype=self.torch_dtype, process_image=True)
                rows = patchify_video(z_norm)
                all_cond_rows.append(rows)
            clean_cond_rows = torch.cat(all_cond_rows, dim=0).to(device=self.device, dtype=self.torch_dtype)
            keyframe_cond_anchor = clean_cond_rows
            if self._cpu_offload_enabled:
                self.video_vae.to("cpu")
                torch.cuda.empty_cache()

        # 3. Latent Noise Initialization
        video_latents, audio_latents = self.generate_noise(
            height=height,
            width=width,
            num_frames=num_frames,
            seed=seed,
        )

        # 4. Denoise Loop
        video_latents, audio_latents = self.denoise_loop(
            video_latents=video_latents,
            audio_latents=audio_latents,
            prompt_embeds=prompt_embeds,
            text_token_tags=text_token_tags,
            num_inference_steps=num_inference_steps,
            flow_shift=flow_shift,
            audio_flow_shift=audio_flow_shift,
            keyframe_cond_anchor=keyframe_cond_anchor,
            keyframe_indices=keyframe_indices,
            progress_bar_cmd=progress_bar_cmd,
        )

        # 5. Decode Latents
        frames, waveform = self.decode_latents(
            video_latents=video_latents,
            audio_latents=audio_latents,
            tiled=tiled,
            tile_size=tile_size,
            tile_overlap=tile_overlap,
        )

        return frames, waveform


__all__ = ["MiniMaxH3Pipeline"]
