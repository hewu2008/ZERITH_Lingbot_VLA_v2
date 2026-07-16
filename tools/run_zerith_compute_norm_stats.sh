#!/bin/bash

DATASET_PATH="/home/jszn/hewu/dataset/hewu2008/pick_and_place_lingbot_vla_v2"

CUDA_VISIBLE_DEVICES=0 bash train.sh scripts/compute_norm_stats.py ./configs/vla/zerith/zerith.yaml \
  --data.data_name zerith \
  --data.train_path "$DATASET_PATH" \
  --data.robot_config_root ./configs/robot_configs \
  --data.norm_path assets/norm_stats/zerith.json \
  --data.data_ratio_for_norm_compute 1