#!/bin/bash
# 모델2: SP(frozen) + LG 학습 (resample 데이터)

GPU_ID=${1:-0}
EXPERIMENT=${2:-"resample_sp_lg"}
IMAGE_SIZE=${3:-2880}
GT_RADIUS=${4:-3}
MAX_KEYPOINTS=${5:-512}
BATCH_SIZE=${6:-4}
RESTORE=${7:-""}
CONF="gluefactory/configs/resample_sp_lg.yaml"

RESTORE_FLAG=""
if [ "$RESTORE" = "--restore" ]; then
    RESTORE_FLAG="--restore"
    echo "=== RESUME training from last checkpoint ==="
fi

echo "=== Training SP+LG on resample data (image_size=${IMAGE_SIZE}, max_kp=${MAX_KEYPOINTS}, batch=${BATCH_SIZE}) ==="
CUDA_VISIBLE_DEVICES=$GPU_ID python3 -m gluefactory.train_resample "$EXPERIMENT" \
    --conf "$CONF" \
    $RESTORE_FLAG \
    --mixed_precision float16 \
    data.image_size="$IMAGE_SIZE" \
    data.batch_size="$BATCH_SIZE" \
    model.ground_truth.gt_radius="$GT_RADIUS" \
    model.extractor.max_num_keypoints="$MAX_KEYPOINTS" \
    2>&1 | tee "outputs/training/${EXPERIMENT}/train.log"