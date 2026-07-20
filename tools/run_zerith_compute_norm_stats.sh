#!/bin/bash

DATASET_PATH="/home/jszn/hewu/dataset/hewu2008/1_clear_the_bin_box_20260720"

torchrun --nproc_per_node=1 scripts/compute_norm_stats.py ./configs/vla/zerith/zerith.yaml \
  --data.data_name zerith \
  --data.num_workers 8 \
  --data.train_path "$DATASET_PATH" \
  --data.robot_config_root ./configs/robot_configs \
  --data.norm_path assets/norm_stats/zerith.json \
  --data.data_ratio_for_norm_compute 1.0