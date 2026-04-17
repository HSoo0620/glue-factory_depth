#!/bin/bash
# FPFH radius ablation: 여러 radius로 짧은 학습 후 경향성 비교
# Usage: bash ablation_fpfh_radius.sh [GPU_ID] [EPOCHS]

GPU_ID=${1:-4}
EPOCHS=${2:-3}
IMAGE_SIZE=2880
GT_RADIUS=11
CONF="gluefactory/configs/resample_sp_fpfh_lg.yaml"
RADII=(0.3 0.5 1.0 1.5 2.0 3.0)

TOTAL_IMAGES=$(ls gluefactory/datasets/mitsubishi/dataset_resample/depth_raw_*.png 2>/dev/null | wc -l)

echo "============================================"
echo "  FPFH Radius Ablation Study"
echo "  GPU: ${GPU_ID}, Epochs: ${EPOCHS}"
echo "  Radii: ${RADII[*]}"
echo "============================================"

# 1) 모든 radius에 대해 캐시 먼저 생성
for R in "${RADII[@]}"; do
    CACHE_DIR="gluefactory/datasets/mitsubishi/fpfh_resample_cache_r${R}"
    CACHED_COUNT=$(ls "$CACHE_DIR"/depth_raw_*.npz 2>/dev/null | wc -l)
    if [ "$CACHED_COUNT" -lt "$TOTAL_IMAGES" ]; then
        echo ""
        echo "=== [Precompute] radius=${R} (${CACHED_COUNT}/${TOTAL_IMAGES}) ==="
        CUDA_VISIBLE_DEVICES="$GPU_ID" python3 precompute_fpfh_resample.py \
            --fpfh_radius "$R" \
            --image_size "$IMAGE_SIZE"
    else
        echo "=== [Precompute] radius=${R} — cache complete (${CACHED_COUNT}), skip ==="
    fi
done

# 2) 각 radius에 대해 짧은 학습
for R in "${RADII[@]}"; do
    EXP="ablation_fpfh_r${R}"
    mkdir -p "outputs/training/${EXP}"

    echo ""
    echo "============================================"
    echo "  Training radius=${R} (${EPOCHS} epochs)"
    echo "============================================"

    CUDA_VISIBLE_DEVICES=$GPU_ID python3 -m gluefactory.train_resample_fpfh "$EXP" \
        --conf "$CONF" \
        --mixed_precision float16 \
        data.fpfh_radius="$R" \
        data.image_size="$IMAGE_SIZE" \
        model.ground_truth.gt_radius="$GT_RADIUS" \
        train.epochs="$EPOCHS" \
        2>&1 | tee "outputs/training/${EXP}/train.log"

    echo "=== Done: radius=${R} ==="
done

echo ""
echo "============================================"
echo "  Ablation Complete! Results:"
echo "============================================"
for R in "${RADII[@]}"; do
    EXP="ablation_fpfh_r${R}"
    LOG="outputs/training/${EXP}/train.log"
    if [ -f "$LOG" ]; then
        BEST=$(grep -oP 'recall[^\d]*\K[\d.]+' "$LOG" 2>/dev/null | sort -n | tail -1)
        echo "  r=${R}: best recall = ${BEST:-N/A}"
    fi
done
