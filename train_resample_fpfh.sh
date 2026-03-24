#!/bin/bash
# 모델3: SP(frozen) + FPFH + LG 학습 (resample 데이터)

GPU_ID=${1:-0}
EXPERIMENT=${2:-"resample_sp_fpfh_lg"}
FPFH_RADIUS=${3:-1.5}
IMAGE_SIZE=${4:-2880}
GT_RADIUS=${5:-11}
RESTORE=${6:-""}
CONF="gluefactory/configs/resample_sp_fpfh_lg.yaml"

RESTORE_FLAG=""
if [ "$RESTORE" = "--restore" ]; then
    RESTORE_FLAG="--restore"
    echo "=== RESUME training from last checkpoint ==="
fi

# 1) FPFH precompute (캐시 파일 수가 부족할 때만)
CACHE_DIR="gluefactory/datasets/mitsubishi/fpfh_resample_cache_r${FPFH_RADIUS}"
TOTAL_IMAGES=$(ls gluefactory/datasets/mitsubishi/dataset_resample/depth_raw_*.png 2>/dev/null | wc -l)
CACHED_COUNT=$(ls "$CACHE_DIR"/depth_raw_*.npz 2>/dev/null | wc -l)
if [ "$CACHED_COUNT" -lt "$TOTAL_IMAGES" ]; then
    echo "=== Precompute FPFH for resample (${CACHED_COUNT}/${TOTAL_IMAGES} done, radius=${FPFH_RADIUS}) ==="
    CUDA_VISIBLE_DEVICES="$GPU_ID" python3 precompute_fpfh_resample.py \
        --fpfh_radius "$FPFH_RADIUS" \
        --image_size "$IMAGE_SIZE"
else
    echo "=== FPFH cache complete: ${CACHED_COUNT} files, skip precompute ==="
fi

# 2) 학습
echo "=== Training SP+FPFH+LG on resample data ==="
CUDA_VISIBLE_DEVICES=$GPU_ID python3 -m gluefactory.train_resample_fpfh "$EXPERIMENT" \
    --conf "$CONF" \
    $RESTORE_FLAG \
    --mixed_precision float16 \
    data.fpfh_radius="$FPFH_RADIUS" \
    data.image_size="$IMAGE_SIZE" \
    model.ground_truth.gt_radius="$GT_RADIUS" \
    2>&1 | tee "outputs/training/${EXPERIMENT}/train.log"
