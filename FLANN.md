# ISS + FLANN Baseline (학습 없는 매칭)

LightGlue 학습 없이, 캐시된 ISS keypoints + descriptors를 OpenCV FLANN으로 직접 매칭.
학습 기반 matcher의 필요성을 판단하기 위한 baseline.

---

## 파이프라인

```
캐시 npz 로드 (keypoints + descriptors, n_valid으로 슬라이싱)
→ cv2.FlannBasedMatcher(KDTree, trees=5, checks=50)
→ knnMatch(desc0, desc1, k=2)
→ Lowe's ratio test (ratio_th=0.75)
→ matched keypoints → crop_to_orig (1751px → 5761px)
→ pixel_to_grid3d → RANSAC+SVD → RMSE
```

---

## FLANN 매칭 원리

- **입력**: descriptor 벡터만 (keypoint 좌표는 사용하지 않음)
- **KDTree**: D차원 공간을 분할하여 L2 최근접 이웃을 O(log N)에 검색
- **kNN(k=2)**: 각 descriptor에 대해 best/second-best match를 찾음
- **Ratio test**: `best.distance < ratio_th × second.distance` → ambiguous match 거부
- 어떤 차원의 float descriptor든 동일하게 적용 가능 (SHOT 352D, FPFH 33D, RoPS 135D)

### FLANN 파라미터

| 파라미터 | 값 | 의미 |
|---|---|---|
| `algorithm` | 1 (FLANN_INDEX_KDTREE) | float 벡터용 KDTree |
| `trees` | 5 | 랜덤 KDTree 5개 구축 (정확도/속도 트레이드오프) |
| `checks` | 50 | 검색 시 최대 50개 노드 탐색 |
| `ratio_th` | 0.75 | Lowe's ratio test 임계값 (낮을수록 엄격) |

### Ratio Test 동작

```
좋은 매칭:  best=10, second=100  →  10/100 = 0.1  < 0.75  ✓ 통과
나쁜 매칭:  best=50, second=60   →  50/60  = 0.83 > 0.75  ✗ 거부
```

| ratio_th | 효과 |
|---|---|
| 0.5 (엄격) | 매칭 수 적지만 정밀도 높음 |
| 0.75 (기본) | Lowe 논문 추천 밸런스 |
| 0.9 (관대) | 매칭 수 많지만 오매칭 증가 |

---

## LightGlue와의 차이

| | FLANN | LightGlue |
|---|---|---|
| 입력 | descriptor만 | descriptor + keypoint 위치 |
| 방식 | L2 거리 + ratio test | Transformer cross-attention |
| 학습 | 불필요 | GT correspondence로 학습 |
| GPU | 불필요 | 필요 |
| 반복 구조 구별 | 불가 (descriptor만 봄) | 가능 (위치 관계 학습) |
| 일대일 매칭 | 보장 안됨 (many-to-one 가능) | mutual matching + dustbin으로 보장 |

---

## 파일

| 파일 | 역할 |
|---|---|
| `eval_registration_iss_shot_flann.py` | Registration RMSE 평가 (`--descriptor shot/fpfh/rops`) |
| `test_resample2_iss_flann.py` | 매칭 시각화 — 4색 라인 (skyblue/purple/limegreen/red) + recall |
| `visualize_registration_flann.py` | 포인트 클라우드 정합 시각화 (3x3 Before/Estimated/GT + overlay) |

---

## 캐시

| Descriptor | 캐시 디렉토리 | 차원 |
|---|---|---|
| SHOT | `iss_shot352_resample2_cache/` | 352D |
| FPFH | `iss_fpfh_resample2_cache_r5.0_xyz/` | 33D |
| RoPS | `iss_rops135_resample2_cache/` | 135D |

캐시 포맷 (모두 동일):
```
depth_raw_XXXX.npz
  ├─ keypoints: (512, 2) float32       # crop 기준 (1751px)
  ├─ keypoint_scores: (512,) float32
  ├─ {shot/fpfh/rops}_descriptors: (512, D) float32
  ├─ n_valid: int                       # 유효 keypoint 수
  └─ n_iss: int                         # ISS 검출 전체 수
```

---

## 사용법

```bash
# Registration 평가
python eval_registration_iss_shot_flann.py --descriptor shot --indices 0 10 50 90
python eval_registration_iss_shot_flann.py --descriptor fpfh --ratio_th 0.8
python eval_registration_iss_shot_flann.py --descriptor rops --num_samples 20

# 매칭 시각화
python test_resample2_iss_flann.py --descriptor shot --indices 0 90
python test_resample2_iss_flann.py --descriptor fpfh --gt_radius 20

# Registration 시각화
python visualize_registration_flann.py --descriptor shot --indices 0 90
python visualize_registration_flann.py --descriptor rops --ratio_th 0.8
```

---

## Baseline 결과 (Pair 0, ratio_th=0.75)

| Descriptor | 매칭 수 (fwd) | Inliers | RMSE (bi) |
|---|---|---|---|
| SHOT (352D) | 53 | 5-6 | ~87 |
| FPFH (33D) | 83 | 5 | ~83 |
| RoPS (135D) | 42 | 5 | ~83 |

- Inlier 비율 ~10% → 대부분의 pair에서 registration 실패
- FLANN은 descriptor만 보고 매칭하므로, 반복 구조에서 오매칭이 많음
- 유사한 뷰의 pair (예: Pair 90)에서는 RMSE ~0.5로 성공
- LightGlue 학습이 이 gap을 줄일 수 있는지가 핵심 질문
