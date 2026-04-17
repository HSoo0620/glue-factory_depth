#!/bin/bash
# ISS(detector) + FPFH(descriptor) + LG training on new dataset.
# Auto-builds FPFH cache if missing.

GPU_ID=${1:-0}
EXPERIMENT=${2:-"0413_new_iss_fpfh_lg"}
FPFH_RADIUS=${3:-50.0}
VOXEL_SIZE=${4:-5.0}
BATCH_SIZE=${5:-32}
GT_RADIUS=${6:-20}
RESTORE=${7:-""}
CONF="gluefactory/configs/0413_new_iss_fpfh_lg.yaml"

RESTORE_FLAG=""
if [ "$RESTORE" = "--restore" ]; then
    RESTORE_FLAG="--restore"
    echo "=== RESUME training from last checkpoint ==="
fi

CACHE_DIR="gluefactory/datasets/new_dataset_cache/cache_new_iss_fpfh_v${VOXEL_SIZE}_r${FPFH_RADIUS}"
if [ ! -d "$CACHE_DIR" ] || [ -z "$(ls -A $CACHE_DIR/*.npz 2>/dev/null)" ]; then
    echo "=== FPFH cache not found. Precomputing... ==="
    python3 precompute_new_iss_fpfh.py \
        --voxel_size "$VOXEL_SIZE" --fpfh_radius "$FPFH_RADIUS"
fi

mkdir -p "outputs/training/${EXPERIMENT}"

echo "=== Training new_dataset ISS+FPFH+LG (voxel=${VOXEL_SIZE}, fpfh_r=${FPFH_RADIUS}, batch=${BATCH_SIZE}, gt_radius=${GT_RADIUS}) ==="
CUDA_VISIBLE_DEVICES=$GPU_ID python3 -m gluefactory.train_new_iss_desc "$EXPERIMENT" \
    --conf "$CONF" \
    $RESTORE_FLAG \
    --mixed_precision float16 \
    data.voxel_size="$VOXEL_SIZE" \
    data.fpfh_radius="$FPFH_RADIUS" \
    data.batch_size="$BATCH_SIZE" \
    model.ground_truth.gt_radius="$GT_RADIUS" \
    2>&1 | tee "outputs/training/${EXPERIMENT}/train.log"
