#!/bin/bash

export HF_ENDPOINT=https://hf-mirror.com

python scripts/download_ms_model.py \
    --repo_id robbyant/lingbot-vla-v2-6b \
    --local_dir /data/4T-1/hewu/model_zoo

python scripts/download_ms_model.py \
    --repo_id Qwen/Qwen3-VL-4B-Instruct \
    --local_dir /data/4T-1/hewu/model_zoo

python scripts/download_ms_model.py \
    --repo_id Ruicheng/moge-2-vitb-normal \
    --local_dir /data/4T-1/hewu/model_zoo
