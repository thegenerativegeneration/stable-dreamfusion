#! /bin/bash

set -e

CUDA_VISIBLE_DEVICES=0 python main.py -O --text "a DSLR photo of a delicious hamburger" --workspace trial_hamburger --iters 5000
CUDA_VISIBLE_DEVICES=1 python main.py -O --text "a DSLR photo of a delicious hamburger" --workspace trial2_hamburger --dmtet --iters 5000 --init_with trial_hamburger/checkpoints/df.pth

CUDA_VISIBLE_DEVICES=1 python main.py -O --text "a highly detailed stone bust of Theodoros Kolokotronis" --workspace trial_stonehead --iters 5000
CUDA_VISIBLE_DEVICES=1 python main.py -O --text "a highly detailed stone bust of Theodoros Kolokotronis" --workspace trial2_stonehead --dmtet --iters 5000 --init_with trial_stonehead/checkpoints/df.pth

CUDA_VISIBLE_DEVICES=1 python main.py -O --text "an astronaut, full body" --workspace trial_astronaut --iters 5000
CUDA_VISIBLE_DEVICES=1 python main.py -O --text "an astronaut, full body" --workspace trial2_astronaut --dmtet --iters 5000 --init_with trial_astronaut/checkpoints/df.pth

CUDA_VISIBLE_DEVICES=1 python main.py -O --text "a DSLR photo of a squirrel-octopus hybrid" --workspace trial_squrrel_octopus --iters 5000
CUDA_VISIBLE_DEVICES=1 python main.py -O --text "a DSLR photo of a squirrel-octopus hybrid" --workspace trial2_squrrel_octopus --dmtet --iters 5000 --init_with trial_squrrel_octopus/checkpoints/df.pth

CUDA_VISIBLE_DEVICES=1 python main.py -O --text "a baby bunny sitting on top of a stack of pancakes" --workspace trial_rabbit_pancake --iters 5000
CUDA_VISIBLE_DEVICES=1 python main.py -O --text "a metal bunny sitting on top of a stack of chocolate cookies" --workspace trial2_rabbit_pancake --dmtet --iters 5000 --init_with trial_rabbit_pancake/checkpoints/df.pth