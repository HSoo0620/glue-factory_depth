#!/bin/bash

GPU_ID=${1:-0}
EXPERIMENT=${2:-"Depth_fpfh"}
FPFH_RADIUS=${3:-1.5}       # FPFH 계산용 radius (precompute 단계)
GT_RADIUS=${4:-3}            # GT matching radius
CONF="gluefactory/configs/3Dlightglue_fpfh.yaml"

# 1) FPFH precompute (캐시가 없을 때만 실행)
CACHE_DIR="gluefactory/datasets/mitsubishi/fpfh_cache_r${FPFH_RADIUS}"
if [ ! -d "$CACHE_DIR" ]; then
    echo "=== Precompute FPFH (radius=${FPFH_RADIUS}) ==="
    python3 precompute_fpfh.py --fpfh_radius "$FPFH_RADIUS"
else
    echo "=== Cache exists: ${CACHE_DIR}, skip precompute ==="
fi

# 2) 학습
echo "=== Training (fpfh_radius=${FPFH_RADIUS}, gt_radius=${GT_RADIUS}) ==="
CUDA_VISIBLE_DEVICES=$GPU_ID python3 -m gluefactory.train_depth_fpfh "$EXPERIMENT" \
    --conf "$CONF" \
    --mixed_precision float16 \
    data.fpfh_radius="$FPFH_RADIUS" \
    model.ground_truth.gt_radius="$GT_RADIUS" \
    2>&1 | tee "outputs/training/${EXPERIMENT}/train.log"
