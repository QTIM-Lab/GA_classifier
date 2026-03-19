#!/bin/bash
python infer_2d.py \
    --ckpt_path  /path_to_checkpoint.ckpt \
    --image_dir  /path_to_png_directory \
    --output_csv /directory/output.csv \
    --threshold  0.5 \
    --batch_size 16 \
    --gpu        0

# This script is for predicting GA in IR and FAF images