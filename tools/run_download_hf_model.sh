#!/bin/bash

export HF_ENDPOINT=https://hf-mirror.com

python scripts/download_ms_model.py \
    --repo_id robbyant/lingbot-vla-v2-6b \
    --local_dir /mnt/d/wsl/model_zoo

python scripts/download_hf_model.py \
    --repo_id Qwen/Qwen3-VL-4B-Instruct \
    --local_dir /mnt/d/wsl/model_zoo

python scripts/download_hf_model.py \
    --repo_id Ruicheng/moge-2-vitb-normal \
    --local_dir /mnt/d/wsl/model_zoo