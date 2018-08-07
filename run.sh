#!/bin/bash

python train.py --cuda \
    --image_size=64 \
    --data_dir="data/resized" \
    --out_dir="systemout" \
    --sample_every=500 \
    2> log.txt
