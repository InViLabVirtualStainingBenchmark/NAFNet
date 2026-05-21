#!/bin/bash
CONFIG=${1:-options/test/BCI/NAFNet-width64.yml}
export PYTHONPATH=/code/NAFNet:/usr/local/lib64/python3.9/site-packages:$PYTHONPATH
cd /code/NAFNet
python3 basicsr/test.py \
    -opt $CONFIG \
    --launcher none \
    2>&1 | tee /output/test_log.txt
