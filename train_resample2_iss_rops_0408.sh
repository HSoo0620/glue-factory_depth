#!/bin/bash
# ISS(detector) + C++ RoPS135(descriptor) + LG 학습 (resample_2 데이터)
# C++ RoPS135: voxelSize=1.0, ropsRadius=10.0, 135D, L2-norm

GPU_ID=${1:-3}
EXPERIMENT=${2:-"0408_resample2_iss_rops135_lg"}
IMAGE_SIZE=${3:-1751}
GT_RADIUS=${4:-20}
BATCH_SIZE=${5:-16}
RESTORE=${6:-""}
CONF="gluefactory/configs/0408_resample2_iss_rops_lg.yaml"
CACHE_DIR="gluefactory/datasets/mitsubishi/iss_rops135_resample2_cache"

RESTORE_FLAG=""
if [ "$RESTORE" = "--restore" ]; then
    RESTORE_FLAG="--restore"
    echo "=== RESUME training from last checkpoint ==="
fi

# ISS+RoPS135 캐시 확인 및 생성
NPZ_COUNT=$(ls "${CACHE_DIR}"/*.npz 2>/dev/null | wc -l)
TOTAL_IMAGES=$(ls gluefactory/datasets/mitsubishi/dataset_resample_2/depth_raw_*.png 2>/dev/null | wc -l)

if [ "$NPZ_COUNT" -lt "$TOTAL_IMAGES" ]; then
    echo "=== ISS+RoPS135 cache incomplete (${NPZ_COUNT}/${TOTAL_IMAGES}). Precomputing... ==="
    python3 precompute_iss_rops_resample2.py --image_size "$IMAGE_SIZE"
else
    echo "=== Cache complete (${NPZ_COUNT}/${TOTAL_IMAGES} frames). Skipping precompute. ==="
fi

# 출력 디렉토리 생성
mkdir -p "outputs/training/${EXPERIMENT}"

echo "=== Training ISS(det)+RoPS135(desc)+LG on resample_2 ==="
echo "    GPU=${GPU_ID}, image_size=${IMAGE_SIZE}, gt_radius=${GT_RADIUS}, batch=${BATCH_SIZE}"
CUDA_VISIBLE_DEVICES=$GPU_ID python3 -m gluefactory.train_resample2_iss_rops "$EXPERIMENT" \
    --conf "$CONF" \
    $RESTORE_FLAG \
    --mixed_precision float16 \
    data.image_size="$IMAGE_SIZE" \
    data.batch_size="$BATCH_SIZE" \
    model.ground_truth.gt_radius="$GT_RADIUS" \
    2>&1 | tee "outputs/training/${EXPERIMENT}/train.log"
