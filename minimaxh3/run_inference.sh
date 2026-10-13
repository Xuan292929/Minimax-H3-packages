#!/bin/bash
set -e
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONUNBUFFERED=1

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"

MODEL_PATH="${MODEL_PATH:-./checkpoints/FL2VA}"
OUTPUT_PATH="${OUTPUT_PATH:-./outputs/minimax_h3_output.mp4}"
FRAMES_DIR="${FRAMES_DIR:-./outputs/extracted_frames}"
PROMPT="${PROMPT:-A majestic lion standing proudly on a rocky outcrop at sunset, golden hour light, cinematic lighting, photorealistic 4k.}"

mkdir -p "$(dirname "$OUTPUT_PATH")"
mkdir -p "$FRAMES_DIR"

echo "=== Starting MiniMax-H3 Inference ==="
echo "Model Path : $MODEL_PATH"
echo "Output Path: $OUTPUT_PATH"
echo "Prompt     : $PROMPT"

"$PYTHON_BIN" "$DIR/scripts/inference.py" \
  --model_path "$MODEL_PATH" \
  --prompt "$PROMPT" \
  --height 480 \
  --width 832 \
  --num_frames 39 \
  --num_inference_steps 25 \
  --seed 42 \
  --device cuda:0 \
  --no_cpu_offload \
  --tile_size 256 \
  --output_path "$OUTPUT_PATH" \
  --frames_dir "$FRAMES_DIR" \
  --extract_frames
echo "=== Done MiniMax-H3 Inference ==="
