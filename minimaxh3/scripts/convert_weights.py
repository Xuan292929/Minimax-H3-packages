#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Bidirectional Weight Conversion Tool for MiniMax-H3.
Criterion 3: 格式固定，同时提供权重格式转化（向 sglang / diffuser 转化）
Pure PyTorch & safetensors implementation without sglang or diffusers dependency.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, Tuple

import torch
from safetensors.torch import load_file, save_file

PREFIX_DIFFUSERS = "transformer."
PREFIX_SGLANG = "model.diffusion_model."


def pytorch_to_diffusers(state_dict: Dict[str, torch.Tensor], add_prefix: bool = True) -> Dict[str, torch.Tensor]:
    """
    Convert native PyTorch state_dict to Diffusers format.
    """
    out_dict = {}
    for k, v in state_dict.items():
        # Remove any existing prefixes first
        clean_k = k
        if clean_k.startswith(PREFIX_SGLANG):
            clean_k = clean_k[len(PREFIX_SGLANG):]
        if clean_k.startswith(PREFIX_DIFFUSERS):
            clean_k = clean_k[len(PREFIX_DIFFUSERS):]

        if add_prefix:
            out_dict[f"{PREFIX_DIFFUSERS}{clean_k}"] = v
        else:
            out_dict[clean_k] = v
    return out_dict


def diffusers_to_pytorch(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    Convert Diffusers format state_dict to native PyTorch format.
    """
    out_dict = {}
    for k, v in state_dict.items():
        clean_k = k
        if clean_k.startswith(PREFIX_DIFFUSERS):
            clean_k = clean_k[len(PREFIX_DIFFUSERS):]
        out_dict[clean_k] = v
    return out_dict


def pytorch_to_sglang(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    Convert native PyTorch state_dict to SGLang format.
    """
    out_dict = {}
    for k, v in state_dict.items():
        clean_k = k
        if clean_k.startswith(PREFIX_DIFFUSERS):
            clean_k = clean_k[len(PREFIX_DIFFUSERS):]
        if clean_k.startswith(PREFIX_SGLANG):
            clean_k = clean_k[len(PREFIX_SGLANG):]
        out_dict[f"{PREFIX_SGLANG}{clean_k}"] = v
    return out_dict


def sglang_to_pytorch(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    Convert SGLang format state_dict to native PyTorch format.
    """
    out_dict = {}
    for k, v in state_dict.items():
        clean_k = k
        if clean_k.startswith(PREFIX_SGLANG):
            clean_k = clean_k[len(PREFIX_SGLANG):]
        out_dict[clean_k] = v
    return out_dict


def load_input_weights(input_path: str) -> Dict[str, torch.Tensor]:
    if os.path.isfile(input_path):
        if input_path.endswith(".safetensors"):
            return load_file(input_path, device="cpu")
        elif input_path.endswith(".pt") or input_path.endswith(".bin"):
            return torch.load(input_path, map_location="cpu", weights_only=True)
        else:
            raise ValueError(f"Unsupported file format: {input_path}")
    elif os.path.isdir(input_path):
        # Look for index file or safetensors
        index_file = os.path.join(input_path, "model.safetensors.index.json")
        state_dict = {}
        if os.path.exists(index_file):
            with open(index_file, "r") as f:
                idx = json.load(f)
            shards = sorted(list(set(idx["weight_map"].values())))
            for shard in shards:
                p = os.path.join(input_path, shard)
                sd = load_file(p, device="cpu")
                state_dict.update(sd)
        else:
            files = sorted([f for f in os.listdir(input_path) if f.endswith(".safetensors")])
            if not files:
                raise FileNotFoundError(f"No safetensors found in {input_path}")
            for f in files:
                sd = load_file(os.path.join(input_path, f), device="cpu")
                state_dict.update(sd)
        return state_dict
    else:
        raise FileNotFoundError(f"Input path does not exist: {input_path}")


def save_output_weights(state_dict: Dict[str, torch.Tensor], output_path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    if output_path.endswith(".safetensors"):
        save_file(state_dict, output_path)
    elif output_path.endswith(".pt"):
        torch.save(state_dict, output_path)
    else:
        # Default to safetensors
        if not output_path.endswith(".safetensors"):
            output_path += ".safetensors"
        save_file(state_dict, output_path)
    print(f"[convert_weights] Successfully saved {len(state_dict)} tensors to {output_path}")


def verify_roundtrip(state_dict: Dict[str, torch.Tensor]) -> bool:
    print("[convert_weights] Verifying round-trip conversions...")
    # 1. Native -> Diffusers -> Native
    diffusers_sd = pytorch_to_diffusers(state_dict, add_prefix=True)
    recovered_from_diffusers = diffusers_to_pytorch(diffusers_sd)
    assert set(state_dict.keys()) == set(recovered_from_diffusers.keys()), "Diffusers roundtrip key mismatch!"
    for k in state_dict:
        assert torch.equal(state_dict[k], recovered_from_diffusers[k]), f"Tensor mismatch for key {k} in Diffusers roundtrip"
    print("  ✓ Native <-> Diffusers round-trip PASSED (100% bit-exact).")

    # 2. Native -> SGLang -> Native
    sglang_sd = pytorch_to_sglang(state_dict)
    recovered_from_sglang = sglang_to_pytorch(sglang_sd)
    assert set(state_dict.keys()) == set(recovered_from_sglang.keys()), "SGLang roundtrip key mismatch!"
    for k in state_dict:
        assert torch.equal(state_dict[k], recovered_from_sglang[k]), f"Tensor mismatch for key {k} in SGLang roundtrip"
    print("  ✓ Native <-> SGLang round-trip PASSED (100% bit-exact).")

    return True


def main():
    parser = argparse.ArgumentParser(description="Bidirectional weight format conversion for MiniMax-H3.")
    parser.add_argument("--input_path", type=str, help="Input model file or directory.")
    parser.add_argument("--output_path", type=str, help="Output destination file.")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["to_diffusers", "from_diffusers", "to_sglang", "from_sglang", "verify_roundtrip"],
        required=True,
        help="Conversion direction or verification mode.",
    )
    args = parser.parse_args()

    if args.mode == "verify_roundtrip":
        # Create mock tensor state dict or load from input_path
        if args.input_path:
            sd = load_input_weights(args.input_path)
        else:
            print("[convert_weights] Creating mock state dict for testing...")
            sd = {
                "video_patch_proj.weight": torch.randn(5376, 96, dtype=torch.bfloat16),
                "video_patch_proj.bias": torch.zeros(5376, dtype=torch.bfloat16),
                "blocks.0.attn.qkv_proj.weight": torch.randn(21504, 5376, dtype=torch.bfloat16),
                "blocks.0.mlp.fc1.weight": torch.randn(28672, 5376, dtype=torch.bfloat16),
                "final_layer.video_out.weight": torch.randn(96, 5376, dtype=torch.bfloat16),
            }
        ok = verify_roundtrip(sd)
        if ok:
            print("[convert_weights] All roundtrip verifications passed successfully!")
            return

    if not args.input_path or not args.output_path:
        print("Error: --input_path and --output_path are required for conversion modes.")
        sys.exit(1)

    print(f"[convert_weights] Loading input weights from {args.input_path}...")
    sd = load_input_weights(args.input_path)
    print(f"[convert_weights] Loaded {len(sd)} keys.")

    if args.mode == "to_diffusers":
        out_sd = pytorch_to_diffusers(sd)
    elif args.mode == "from_diffusers":
        out_sd = diffusers_to_pytorch(sd)
    elif args.mode == "to_sglang":
        out_sd = pytorch_to_sglang(sd)
    elif args.mode == "from_sglang":
        out_sd = sglang_to_pytorch(sd)
    else:
        raise ValueError(f"Unknown mode: {args.mode}")

    save_output_weights(out_sd, args.output_path)


if __name__ == "__main__":
    main()
