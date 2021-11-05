#!/usr/bin/env bash
set -ex

CUDA_VISIBLE_DEVICES='0,1'
echo "$(date "+%Y.%m.%d-%H.%M.%S")"
python gtad_train.py  
python gtad_inference.py 
python gtad_postprocess.py 
echo "$(date "+%Y.%m.%d-%H.%M.%S")"
