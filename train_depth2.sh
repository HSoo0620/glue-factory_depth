#!/bin/bash

EXPERIMENT="Depth_only_sp2"
CONF="gluefactory/configs/3Dlightglue_custom2.yaml"
GPU_ID=${1:-0}  # 첫 번째 인자로 GPU 번호 지정, 기본값 0

CUDA_VISIBLE_DEVICES=$GPU_ID python3 -m gluefactory.train_depth_custom "$EXPERIMENT" \
    --conf "$CONF" \
    --mixed_precision float16 \
    2>&1 | tee "outputs/training/${EXPERIMENT}/train.log"
