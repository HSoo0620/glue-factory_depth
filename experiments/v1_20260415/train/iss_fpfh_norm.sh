#!/bin/bash
# v1 (2026-04-15) ISS+FPFH(L2-norm)+LG 학습 entry.
# 재학습 (2026-04-17): collate_fn padding_value -1.0 → 0.0 수정 후 재학습.
#   기존 체크포인트(iss_fpfh_v1_20260415_norm)는 padding GT row 가 valid 로 통과되어 오염된 그래디언트로 학습됨.
# 실행: bash experiments/v1_20260415/train/iss_fpfh_norm.sh  (repo root 기준)
set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

GPU=${GPU:-0}
EXP=${EXP:-iss_fpfh_v1_20260415_norm_0417}
BATCH=${BATCH:-4}
RESTORE=${RESTORE:-}

CONFIG=gluefactory/configs/iss_fpfh_v1_20260415_norm_lg.yaml
CACHE_DIR=gluefactory/datasets/mitsubishi/iss_fpfh_v1_20260415_cache_vox1_nr20_fr20_norm

if [ ! -d "$CACHE_DIR" ] || [ -z "$(ls -A $CACHE_DIR 2>/dev/null)" ]; then
    echo "[v1] FPFH(norm) cache missing -> run precompute first:"
    echo "  python experiments/v1_20260415/precompute/iss_fpfh_norm.py"
    exit 1
fi

CMD="CUDA_VISIBLE_DEVICES=$GPU python -m gluefactory.train_iss_v1_20260415 \
    $EXP --conf $CONFIG data.batch_size=$BATCH"
if [ -n "$RESTORE" ]; then
    CMD="$CMD --restore $RESTORE"
fi
echo "$CMD"
eval "$CMD"
