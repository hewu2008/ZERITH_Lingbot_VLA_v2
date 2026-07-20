#!/bin/bash

DATASET_PATH="/home/jszn/hewu/dataset/hewu2008/pick_and_place_lingbot_vla_v2"

torchrun --nproc_per_node=1 scripts/compute_norm_stats.py ./configs/vla/zerith/zerith.yaml \
  --data.data_name zerith \
  --data.num_workers 8 \
  --data.train_path "$DATASET_PATH" \
  --data.robot_config_root ./configs/robot_configs \
  --data.norm_path assets/norm_stats/zerith.json \
  --data.data_ratio_for_norm_compute 0.25