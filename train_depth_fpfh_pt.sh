#!/bin/bash

EXPERIMENT="Depth_fpfh_pt"
CONF="gluefactory/configs/3Dlightglue_fpfh.yaml"
GPU_ID=${1:-0}

CUDA_VISIBLE_DEVICES=$GPU_ID python3 -m gluefactory.train_depth_fpfh "$EXPERIMENT" \
    --conf "$CONF" \
    --mixed_precision float16 \
    train.load_experiment=Depth_fpfh \
    2>&1 | tee "outputs/training/${EXPERIMENT}/train.log"
