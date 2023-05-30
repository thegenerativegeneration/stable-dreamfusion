#! /bin/bash

set -e

ENV_CUDA_VISIBLE_DEVICES=0


CUDA_VISIBLE_DEVICES=$ENV_CUDA_VISIBLE_DEVICES python main.py -O --text "a baby bunny sitting on top of a stack of pancakes" --workspace trial_if_rabbit_pancake --iters 5000 --IF --vram_O --fp16 --gui
