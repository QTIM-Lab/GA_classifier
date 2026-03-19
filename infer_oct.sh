#!/bin/bash

python infer_oct.py \
    --ckpt_path  /path_to_checkpoint.ckpt \
    --volume_dir /path_to_volume_directory  \
    --output_csv /directory/output.csv \
    --threshold  0.5 \
    --num_frames 49 \
    --batch_size 12 \
    --gpu        1

# This script is for predicting GA presence in OCT volumes
# The image directory should be structured as follows: 
# /volume_directory/volume_identifier/image_1.png
# /volume_directory/volume_identifier/image_2.png
# /volume_directory/volume_identifier/image_3.png
# etc.
# The images only need to be named in a way that forces them to be organized consecutively within the folder