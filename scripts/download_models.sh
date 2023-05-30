#!/bin/bash

set -e

wget https://huggingface.co/cvlab/zero123-weights/resolve/main/105000.ckpt -O pretrained/zero123/105000.ckpt
wget https://huggingface.co/thegenerativegeneration/omnidata/resolve/main/omnidata_dpt_depth_v2.ckpt -O pretrained/omnidata/omnidata_dpt_depth_v2.ckpt
wget https://huggingface.co/thegenerativegeneration/omnidata/resolve/main/omnidata_dpt_normal_v2.ckpt -O pretrained/omnidata/omnidata_dpt_normal_v2.ckpt