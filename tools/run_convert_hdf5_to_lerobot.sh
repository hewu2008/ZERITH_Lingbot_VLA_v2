#!/bin/bash

export HF_LEROBOT_HOME=/home/jszn/hewu/dataset/

python scripts/convert_hdf5_to_lerobot.py \
  --raw_dir /home/jszn/hewu/dataset/hdf5/1_clear_the_bin_box_20260720 \
  --repo_id hewu2008/1_clear_the_bin_box_20260720 \
  --task 'clear the bin box'
