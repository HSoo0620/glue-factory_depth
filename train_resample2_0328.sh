#!/bin/bash
# SP(frozen) + LG 학습 (resample_2 데이터, 고정 crop 3502 -> 리사이즈 1751)

GPU_ID=${1:-0}
EXPERIMENT=${2:-"0328_resample2_sp_lg"}
IMAGE_SIZE=${3:-1751}
GT_RADIUS=${4:-6}
MAX_KEYPOINTS=${5:-512}
BATCH_SIZE=${6:-4}
RESTORE=${7:-""}
CONF="gluefactory/configs/0328_resample2_sp_lg.yaml"

RESTORE_FLAG=""
if [ "$RESTORE" = "--restore" ]; then
    RESTORE_FLAG="--restore"
    echo "=== RESUME training from last checkpoint ==="
fi

# 출력 디렉토리 생성
mkdir -p "outputs/training/${EXPERIMENT}"

echo "=== Training SP+LG on resample_2 data (crop=3502, image_size=${IMAGE_SIZE}, gt_radius=${GT_RADIUS}, max_kp=${MAX_KEYPOINTS}, batch=${BATCH_SIZE}) ==="
CUDA_VISIBLE_DEVICES=$GPU_ID python3 -m gluefactory.train_resample2 "$EXPERIMENT" \
    --conf "$CONF" \
    $RESTORE_FLAG \
    --mixed_precision float16 \
    data.image_size="$IMAGE_SIZE" \
    data.batch_size="$BATCH_SIZE" \
    model.ground_truth.gt_radius="$GT_RADIUS" \
    model.extractor.max_num_keypoints="$MAX_KEYPOINTS" \
    2>&1 | tee "outputs/training/${EXPERIMENT}/train.log"
