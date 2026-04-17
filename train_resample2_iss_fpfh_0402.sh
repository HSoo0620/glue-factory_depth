#!/bin/bash
# ISS(detector) + FPFH(descriptor) + LG 학습 (resample_2 데이터)
# ISS+FPFH 캐시 없으면 자동 생성

GPU_ID=${1:-3}
EXPERIMENT=${2:-"0406_resample2_iss_fpfh_xyz_lg"}
FPFH_RADIUS=${3:-5.0}
IMAGE_SIZE=${4:-1751}
GT_RADIUS=${5:-20}
BATCH_SIZE=${6:-32}
RESTORE=${7:-""}
CONF="gluefactory/configs/0402_resample2_iss_fpfh_lg.yaml"

RESTORE_FLAG=""
if [ "$RESTORE" = "--restore" ]; then
    RESTORE_FLAG="--restore"
    echo "=== RESUME training from last checkpoint ==="
fi

# ISS+FPFH 캐시 확인 및 생성
CACHE_DIR="gluefactory/datasets/mitsubishi/iss_fpfh_resample2_cache_r${FPFH_RADIUS}_xyz"
if [ ! -d "$CACHE_DIR" ] || [ -z "$(ls -A $CACHE_DIR/*.npz 2>/dev/null)" ]; then
    echo "=== ISS+FPFH cache not found. Precomputing... ==="
    python3 precompute_iss_fpfh_resample2.py \
        --fpfh_radius "$FPFH_RADIUS" \
        --image_size "$IMAGE_SIZE"
fi

# 출력 디렉토리 생성
mkdir -p "outputs/training/${EXPERIMENT}"

echo "=== Training ISS(det)+FPFH(desc)+LG on resample_2 (fpfh_r=${FPFH_RADIUS}, image_size=${IMAGE_SIZE}, gt_radius=${GT_RADIUS}, batch=${BATCH_SIZE}) ==="
CUDA_VISIBLE_DEVICES=$GPU_ID python3 -m gluefactory.train_resample2_iss_fpfh "$EXPERIMENT" \
    --conf "$CONF" \
    $RESTORE_FLAG \
    --mixed_precision float16 \
    data.fpfh_radius="$FPFH_RADIUS" \
    data.image_size="$IMAGE_SIZE" \
    data.batch_size="$BATCH_SIZE" \
    model.ground_truth.gt_radius="$GT_RADIUS" \
    2>&1 | tee "outputs/training/${EXPERIMENT}/train.log"
