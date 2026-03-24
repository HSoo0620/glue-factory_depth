#!/bin/bash
# 모델4: SP(frozen) + Hybrid(SP+FPFH) + LG 학습 (resample 데이터)

GPU_ID=${1:-0}
EXPERIMENT=${2:-"resample_sp_hybrid_lg"}
FPFH_RADIUS=${3:-0.5}
IMAGE_SIZE=${4:-2880}
GT_RADIUS=${5:-11}
RESTORE=${6:-""}
CONF="gluefactory/configs/resample_sp_hybrid_lg.yaml"

RESTORE_FLAG=""
if [ "$RESTORE" = "--restore" ]; then
    RESTORE_FLAG="--restore"
    echo "=== RESUME training from last checkpoint ==="
fi

# 1) FPFH 캐시 확인 (hybrid 캐시의 전제조건)
FPFH_CACHE_DIR="gluefactory/datasets/mitsubishi/fpfh_resample_cache_r${FPFH_RADIUS}"
if [ ! -d "$FPFH_CACHE_DIR" ]; then
    echo "=== FPFH cache not found, creating first ==="
    python3 precompute_fpfh_resample.py \
        --fpfh_radius "$FPFH_RADIUS" \
        --image_size "$IMAGE_SIZE"
fi

# 2) Hybrid cache precompute (기존 FPFH 캐시 + SP descriptor)
CACHE_DIR="gluefactory/datasets/mitsubishi/hybrid_resample_cache_r${FPFH_RADIUS}"
FPFH_COUNT=$(ls "$FPFH_CACHE_DIR"/depth_raw_*.npz 2>/dev/null | wc -l)
HYBRID_COUNT=$(ls "$CACHE_DIR"/depth_raw_*.npz 2>/dev/null | wc -l)
if [ "$HYBRID_COUNT" -lt "$FPFH_COUNT" ]; then
    echo "=== Precompute Hybrid cache (${HYBRID_COUNT}/${FPFH_COUNT} done, radius=${FPFH_RADIUS}) ==="
    python3 precompute_hybrid_resample.py \
        --fpfh_radius "$FPFH_RADIUS" \
        --image_size "$IMAGE_SIZE"
else
    echo "=== Hybrid cache complete: ${HYBRID_COUNT} files, skip precompute ==="
fi

# 3) 학습
echo "=== Training SP+Hybrid(SP+FPFH)+LG on resample data ==="
CUDA_VISIBLE_DEVICES=$GPU_ID python3 -m gluefactory.train_resample_hybrid "$EXPERIMENT" \
    --conf "$CONF" \
    $RESTORE_FLAG \
    --mixed_precision float16 \
    data.fpfh_radius="$FPFH_RADIUS" \
    data.image_size="$IMAGE_SIZE" \
    model.ground_truth.gt_radius="$GT_RADIUS" \
    2>&1 | tee "outputs/training/${EXPERIMENT}/train.log"
