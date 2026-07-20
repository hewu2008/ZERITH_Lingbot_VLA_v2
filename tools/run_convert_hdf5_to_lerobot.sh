#!/bin/bash

export HF_LEROBOT_HOME=/home/jszn/hewu/dataset/

python scripts/convert_hdf5_to_lerobot.py \
  --raw_dir /home/jszn/hewu/dataset/clear_the_bin_box \
  --repo_id hewu2008/clear_the_bin_box_lingbot_vla_v2 \
  --task 'clear the bin box'
