#!/bin/bash

BLOCKS_PER_TASK=5

for ((i=0; i<=56; i+=BLOCKS_PER_TASK))
do
    echo "Compressing blocks ${i} to $((i+BLOCKS_PER_TASK))"
    taskset -c $((i/BLOCKS_PER_TASK)) python compress_flux.py \
        --model_name_or_path black-forest-labs/FLUX.1-dev \
        --save_path ./FLUX.1-dev-DF11 \
        --check_correctness \
        --block_range ${i} $((i+BLOCKS_PER_TASK)) &
done

wait
