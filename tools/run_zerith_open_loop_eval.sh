#!/bin/bash

export QWEN3_PATH=/home/jszn/hewu/model_zoo/Qwen3-VL-4B-Instruct
python scripts/open_loop_eval.py \
  --model_path /home/jszn/hewu/alg-product/ZERITH_Lingbot_VLA_V2/output/zerith_lora/checkpoints \
  --robo_name zerith \
  --data_path /home/jszn/hewu/dataset/hewu2008/1_clear_the_bin_box_20260721 \
  --use_length 50