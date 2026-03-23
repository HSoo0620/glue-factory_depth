#!/bin/bash
# 모델3: SP(frozen) + FPFH + LG 학습 (resample 데이터)

GPU_ID=${1:-0}
EXPERIMENT=${2:-"resample_sp_fpfh_lg"}
FPFH_RADIUS=${3:-1.5}
IMAGE_SIZE=${4:-2880}
GT_RADIUS=${5:-3}
CONF="gluefactory/configs/resample_sp_fpfh_lg.yaml"

# 1) FPFH precompute (캐시가 없을 때만)
CACHE_DIR="gluefactory/datasets/mitsubishi/fpfh_resample_cache_r${FPFH_RADIUS}"
if [ ! -d "$CACHE_DIR" ]; then
    echo "=== Precompute FPFH for resample (radius=${FPFH_RADIUS}, image_size=${IMAGE_SIZE}) ==="
    python3 precompute_fpfh_resample.py \
        --fpfh_radius "$FPFH_RADIUS" \
        --image_size "$IMAGE_SIZE"
else
    echo "=== Cache exists: ${CACHE_DIR}, skip precompute ==="
fi

# 2) 학습
echo "=== Training SP+FPFH+LG on resample data ==="
CUDA_VISIBLE_DEVICES=$GPU_ID python3 -m gluefactory.train_resample_fpfh "$EXPERIMENT" \
    --conf "$CONF" \
    --mixed_precision float16 \
    data.fpfh_radius="$FPFH_RADIUS" \
    data.image_size="$IMAGE_SIZE" \
    model.ground_truth.gt_radius="$GT_RADIUS" \
    2>&1 | tee "outputs/training/${EXPERIMENT}/train.log"
