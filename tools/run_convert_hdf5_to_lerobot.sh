#!/bin/bash

export HF_LEROBOT_HOME=/home/jszn/hewu/dataset/

python scripts/convert_hdf5_to_lerobot.py \
  --raw_dir /home/jszn/hewu/dataset/1_put_the_part_in_the_box \
  --repo_id hewu2008/pick_and_place_lingbot_vla_v2
