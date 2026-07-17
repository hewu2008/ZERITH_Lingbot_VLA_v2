#!/bin/bash

set -e

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)

CHECKPOINT_DIR="${PROJECT_ROOT}/output/zerith_lora/checkpoints/global_step_200"

echo "======================================"
echo "Checking LoRA weights at step 200"
echo "======================================"
echo "Project root: $PROJECT_ROOT"
echo "Checkpoint dir: $CHECKPOINT_DIR"
echo ""

if [ ! -d "$CHECKPOINT_DIR" ]; then
    echo "[ERROR] Checkpoint directory not found: $CHECKPOINT_DIR"
    exit 1
fi

echo "Running LoRA weight checker..."
echo ""

python "$PROJECT_ROOT/scripts/check_lora_weights.py" \
    --checkpoint_dir "$CHECKPOINT_DIR"

echo ""
echo "======================================"
echo "Done!"
echo "======================================"