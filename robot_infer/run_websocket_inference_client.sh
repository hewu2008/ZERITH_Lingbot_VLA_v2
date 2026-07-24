#!/bin/bash

set -e

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)

ROBO_NAME="zerith"
HOST="172.31.200.250"
PORT=55555
PROMPT="clear the bin box"
NUM_STEPS=1200
WARMUP_STEPS=2
CAMERA_NAMES="cam_high cam_left_wrist cam_right_wrist"
SKIP_PAUSE=""

OPEN_LOOP_EVAL=""
DATA_PATH="/home/jszn/hewu/dataset/hewu2008/1_clear_the_bin_box_20260721"
TRAJ_IDS="0"
CHUNK_SIZE=50
MAX_INFER_TIME=40
SAVE_PLOT_PATH="./open_loop_test_cs/"

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

if [ -n "$OPEN_LOOP_EVAL" ]; then
    echo "[INFO] Open-loop evaluation mode enabled"
    echo "[INFO] Data path: $DATA_PATH"
    echo "[INFO] Trajectory IDs: $TRAJ_IDS"
    echo "[INFO] Chunk size: $CHUNK_SIZE"
    echo "[INFO] Max infer time: $MAX_INFER_TIME"
    echo "[INFO] Save plot path: $SAVE_PLOT_PATH"
    echo ""

    cd "$PROJECT_ROOT"
    python robot_infer/websocket_inference_client.py \
        --host "$HOST" \
        --port "$PORT" \
        --robo_name "$ROBO_NAME" \
        --open_loop_eval \
        --data_path "$DATA_PATH" \
        --traj_ids $TRAJ_IDS \
        --chunk_size "$CHUNK_SIZE" \
        --max_infer_time "$MAX_INFER_TIME" \
        --save_plot_path "$SAVE_PLOT_PATH"
else
    cd "$PROJECT_ROOT"
    sudo -E /home/robot/miniconda3/envs/lingbot-va/bin/python robot_infer/websocket_inference_client.py \
        --host "$HOST" \
        --port "$PORT" \
        --robo_name "$ROBO_NAME" \
        --prompt "$PROMPT" \
        --num_steps "$NUM_STEPS" \
        --warmup_steps "$WARMUP_STEPS" \
        --camera_names $CAMERA_NAMES \
        --init_hdf5 assets/episode.hdf5 \
        --init_frame_idx 0 \
        $SKIP_PAUSE
fi