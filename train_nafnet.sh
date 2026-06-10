#!/bin/bash
CONFIG=${1:-options/train/BCI/NAFNet-BCI-512-100k-part1.yml}
export PYTHONPATH=/code/NAFNet:/usr/local/lib64/python3.9/site-packages:$PYTHONPATH
cd /code/NAFNet
python3 basicsr/train.py \
    -opt $CONFIG \
    2>&1 | tee /output/train_log.txt
