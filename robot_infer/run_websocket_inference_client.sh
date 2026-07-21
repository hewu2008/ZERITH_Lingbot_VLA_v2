#!/bin/bash

set -e

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)

ROBO_NAME="zerith"
HOST="172.31.200.250"
PORT=55555
PROMPT="clear the bin box"
NUM_STEPS=5
WARMUP_STEPS=2
CAMERA_NAMES="cam_high cam_left_wrist cam_right_wrist"
SKIP_PAUSE=""

echo "======================================"
echo "Zerith WebSocket Inference Client"
echo "(Run this after starting the server)"
echo "======================================"
echo "Project root: $PROJECT_ROOT"
echo ""

echo "[INFO] Starting WebSocket inference client..."
echo "[INFO] Server: $HOST:$PORT"
echo "[INFO] Prompt: $PROMPT"
echo "[INFO] Number of steps: $NUM_STEPS"
echo "[INFO] Warmup steps: $WARMUP_STEPS"
echo "[INFO] Camera names: $CAMERA_NAMES"
echo "[INFO] Press Ctrl+C to stop"
echo ""

cd "$PROJECT_ROOT"
sudo -E /home/robot/miniconda3/envs/lingbot-va/bin/python robot_infer/websocket_inference_client.py \
    --host "$HOST" \
    --port "$PORT" \
    --robo_name "$ROBO_NAME" \
    --prompt "$PROMPT" \
    --num_steps "$NUM_STEPS" \
    --warmup_steps "$WARMUP_STEPS" \
    --camera_names $CAMERA_NAMES \
    --init_hdf5 /home/robot/hewu/alg-product/ZERITH_Lingbot_VLA_V2/assets/episode.hdf5 \
    --init_frame_idx 0 \
    $SKIP_PAUSE