# ISS+SHOT FLANN Baseline 평가 스크립트 설계

## 목적

LightGlue 학습 없이, 캐시된 ISS keypoints + SHOT descriptors를 OpenCV FLANN으로 매칭하여 Registration baseline 성능을 확인한다.
기존 `eval_registration_iss_shot.py` (LightGlue 기반)와 동일한 RANSAC+SVD+RMSE 파이프라인을 사용하여 공정한 비교가 가능하도록 한다.

## 배경

- SIFT/ORB는 OpenCV FLANN 내장 함수로 바로 매칭 가능 → `eval_registration_rule_based.py`에 이미 구현됨
- 3D descriptor(SHOT/FPFH/RoPS)도 float 벡터이므로 동일한 FLANN KDTree로 매칭 가능
- 학습 기반 matcher(LightGlue) 대비 FLANN baseline 성능을 먼저 확인하여, matcher 학습의 가치를 판단

## 파일

- **새 스크립트**: `eval_registration_iss_shot_flann.py`
- **기존 참조**: `eval_registration_rule_based.py` (SIFT/ORB FLANN), `eval_registration_iss_shot.py` (ISS+SHOT+LG)

## 캐시 구조

```
iss_shot352_resample2_cache/depth_raw_XXXX.npz
  ├─ keypoints: (512, 2) float32       # crop 기준 (1751px)
  ├─ keypoint_scores: (512,) float32
  ├─ shot_descriptors: (512, 352) float32
  ├─ n_valid: int                       # 유효 keypoint 수 (패딩 제외)
  └─ n_iss: int                         # ISS 검출 전체 수
```

## 데이터 흐름

```
npz 캐시 로드
→ n_valid으로 슬라이싱: kp[:n_valid], desc[:n_valid]
→ cv2.FlannBasedMatcher(KDTree).knnMatch(desc0, desc1, k=2)
→ Lowe's ratio test (기본 0.75)
→ matched keypoints → crop_to_orig (1751px → 5761px)
→ pixel_to_grid3d (원본 depth에서 Grid 3D 좌표)
→ ransac_rigid (1000 iter, inlier_th=5.0)
→ compute_transform_rmse (T_est vs T_gt, 30K pts)
→ 양방향 (forward + reverse) 평균 RMSE
```

## 좌표 변환

캐시의 keypoint는 1751px (crop+resize) 기준. 원본 5761px로 변환:

```python
scale = CROP_SIZE / image_size  # 3502 / 1751 = 2.0
kp_orig = kp_crop * scale
kp_orig[:, 0] += CROP_X0  # 1129
kp_orig[:, 1] += CROP_Y0  # 1081
```

`eval_registration_iss_shot.py`의 `kpts_to_orig`과 동일.

## FLANN 설정

SHOT 352D float32 → KDTree (SIFT와 동일한 알고리즘):

```python
index_params = dict(algorithm=1, trees=5)   # FLANN_INDEX_KDTREE
search_params = dict(checks=50)
```

## 매칭 메트릭

| 메트릭 | 설명 |
|---|---|
| `n_kp0`, `n_kp1` | 유효 keypoint 수 (n_valid) |
| `n_matches` | ratio test 통과한 매칭 수 |
| `n_3d` | depth > 0인 유효 3D 매칭 수 |
| `n_inliers` | RANSAC inlier 수 |
| `inlier_ratio` | n_inliers / n_3d |

## Registration 평가

기존 eval 스크립트와 완전 동일:
- **좌표계**: Grid `(u*GRID_DX, v*GRID_DY, depth_real)`
- **RANSAC**: Custom SVD RANSAC (1000 iter, inlier_th=5.0)
- **T_gt**: GT CSV 비-occluded 대응점 → Grid 좌표 → SVD
- **RMSE**: T_est vs T_gt, 30K 샘플 포인트, 양방향 평균

## Pair 인덱싱

`eval_registration_rule_based.py`와 동일한 방식:
- `gluefactory/datasets/mitsubishi/outputs_txt/combination.csv` 로드
- Split: train / val(마지막 200) / test(마지막 100)
- `--indices`로 split 내 인덱스 지정, 없으면 `--num_samples`개 랜덤 선택

캐시 파일명은 combination.csv의 master/input 파일명에서 인덱스 추출 (예: `depth_raw_0000.npz`).

## CLI

```bash
python eval_registration_iss_shot_flann.py \
    --cache_dir iss_shot352_resample2_cache \
    --ratio_th 0.75 \
    --indices 0 10 50 90 \
    --inlier_th 5.0 \
    --ransac_iter 1000
```

## 출력

- 콘솔: pair별 매칭/registration 결과 + 전체 summary (mean RMSE)
- `results/registration/iss_shot_flann_eval/`: overlay 이미지 + `summary.csv`

## 기존 스크립트와의 차이점

| | `eval_registration_rule_based.py` | 새 스크립트 |
|---|---|---|
| Keypoint 검출 | SIFT/ORB (이미지에서 실시간) | 캐시 로드 (ISS) |
| Descriptor | SIFT 128D / ORB 32D binary | SHOT 352D float |
| 좌표 변환 | crop 좌표 → +offset → orig | crop(1751px) → ×scale+offset → orig |
| FLANN | KDTree(SIFT) / LSH(ORB) | KDTree (float) |
| 나머지 파이프라인 | 동일 | 동일 |

## 향후 확장

`--descriptor` 인자로 shot/fpfh/rops 선택 가능하게 확장 가능. 캐시 포맷이 동일하므로 descriptor 키 이름과 캐시 경로만 변경하면 됨.
