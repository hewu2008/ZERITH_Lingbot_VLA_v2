#!/bin/bash

set -e

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)

CHECKPOINT_DIR="${PROJECT_ROOT}/output/zerith_lora/checkpoints/global_step_500"
ROBOT_NORM_PATH="${PROJECT_ROOT}/assets/norm_stats/zerith.json"
PORT=55555

export QWEN3VL_PATH="/home/jszn/hewu/model_zoo/Qwen3-VL-4B-Instruct/"

echo "======================================"
echo "Zerith WebSocket Inference Server"
echo "(Direct LoRA loading mode)"
echo "======================================"
echo "Project root: $PROJECT_ROOT"
echo "Checkpoint dir: $CHECKPOINT_DIR"
echo "Robot norm path: $ROBOT_NORM_PATH"
echo ""

if [ ! -d "$CHECKPOINT_DIR" ]; then
    echo "[ERROR] Checkpoint directory not found: $CHECKPOINT_DIR"
    exit 1
fi

if [ ! -f "$ROBOT_NORM_PATH" ]; then
    echo "[ERROR] Robot norm stats file not found: $ROBOT_NORM_PATH"
    exit 1
fi

echo ""
echo "[INFO] Starting WebSocket inference server with LoRA..."
echo "[INFO] Model: $CHECKPOINT_DIR"
echo "[INFO] Port: $PORT"
echo "[INFO] Press Ctrl+C to stop"
echo ""

cd "$PROJECT_ROOT"
python robot_infer/websocket_inference_server.py \
    --model_path "$CHECKPOINT_DIR" \
    --robot_norm_path "$ROBOT_NORM_PATH" \
    --port "$PORT" \
    --use_bf16 true \
    --use_compile true