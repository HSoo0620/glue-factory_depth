# Registration RMSE Evaluation — Transform-Based

**Date**: 2026-04-08
**Status**: Draft
**Methods**: ISS+SHOT+LG, ISS+FPFH+LG

---

## 목적

방법론 간 Registration 성능을 공정하게 비교하기 위한 Transform 기반 RMSE 평가 스크립트 작성.

### 현재 방식의 문제

- RANSAC inlier pair의 잔차(residual)로 RMSE 산출
- 매칭 pair 수/분포에 따라 RMSE 달라짐 → 방법론 간 공정 비교 불가

### 새 방식

- RANSAC으로 T_est(4x4) 추정 후, GT correspondence SVD로 유도한 T_gt와 비교
- 포인트 클라우드 샘플링 기반 RMSE → 매칭 pair 수와 무관

---

## 파이프라인

```
[모델 추론] → [매칭 pairs] → [Grid 3D 변환] → [RANSAC → T_est]
                                                     |
[GT CSV] → [비-occluded 대응점] → [Grid 3D 변환] → [SVD → T_gt]
                                                     |
                                          [포인트 클라우드 30K 샘플링]
                                                     |
                                          [T_est vs T_gt RMSE]
                                                     |
                                    [reverse pair로 반복 → 양방향 평균]
```

---

## 좌표계: Grid 3D (Calibration 스케일)

Calibration grid 스케일 적용. RANSAC, T_gt, RMSE 모두 동일 좌표계.

```python
depth_real = clip_start + (raw_uint16 / 65535.0) * (clip_end - clip_start)
X = u * grid_dx          # u * 0.05 mm
Y = v * grid_dy          # v * 0.05 mm
Z = depth_real * grid_dz  # depth_real * 0.02 mm
```

- grid_dx = grid_dy = 0.05 mm/pixel, grid_dz = 0.02
- u, v는 원본 해상도(5761) 기준

---

## GT 변환 행렬 (T_gt)

GT CSV의 비-occluded 대응점으로부터 SVD로 rigid transform 유도.

```python
# 1. GT CSV에서 비-occluded 대응점 로드
#    (master_x, master_y) → depth lookup → Grid 3D
#    (input_x, input_y)   → depth lookup → Grid 3D

# 2. SVD로 rigid transform 추정 (input → master)
#    centroid 제거 → H = P_input_centered.T @ P_master_centered
#    U, S, Vt = svd(H) → R_gt = Vt.T @ U.T, t_gt = centroid_master - R_gt @ centroid_input
```

T_gt는 Grid 3D 좌표계에서 input→master 변환.

---

## RANSAC

- Open3D `registration_ransac_based_on_correspondence` 사용 (기존과 동일)
- 입력 좌표: Grid 3D (u*dx, v*dy, depth_real*dz)
- 매칭된 keypoint 쌍을 Grid 3D로 변환 → correspondence로 전달
- 출력: T_est (4x4)

---

## RMSE 평가

### 포인트 클라우드 샘플링
- 소스(input/master) depth map에서 유효(depth > 0) 픽셀 중 30,000점 균일 샘플링
- Grid 3D로 변환 → P_src (30000, 3)

### RMSE 계산
```python
# source 이미지의 샘플 포인트에 T_est, T_gt 각각 적용 후 차이 측정
P_est = (R_est @ P_src.T).T + t_est
P_gt  = (R_gt  @ P_src.T).T + t_gt
RMSE  = sqrt(mean(||P_est - P_gt||^2))
```

T_est와 T_gt를 동일한 소스 포인트에 적용하고, 변환 결과 간 거리로 RMSE 산출.
매칭 pair 수와 완전히 독립.

### 양방향 평가
1. Forward: (master, input) 추론 → T_est_fwd (input→master) → input 포인트 30K 샘플링 → RMSE_fwd
2. Reverse: (input, master) 추론 → T_est_rev (master→input) → master 포인트 30K 샘플링 → RMSE_rev
3. **Bidirectional RMSE = (RMSE_fwd + RMSE_rev) / 2**

---

## 출력

### 터미널
- Pair별: forward RMSE, reverse RMSE, bidirectional RMSE (mm)

### CSV
- 전체 test set 결과 테이블: pair_idx, master, input, rmse_fwd, rmse_rev, rmse_bi, n_matches_fwd, n_matches_rev, n_inliers_fwd, n_inliers_rev

### 시각화 (간소화)
- Overlay 1장: master(cyan) + warped input(red)
- Forward 방향만 시각화

---

## 파일 구조

| 파일 | 역할 |
|---|---|
| `eval_registration_iss_shot.py` | ISS+SHOT+LG 평가 |
| `eval_registration_iss_fpfh.py` | ISS+FPFH+LG 평가 |

두 파일은 dataset/collate 임포트만 다르고 평가 로직은 동일.

---

## CLI 인터페이스

```bash
python eval_registration_iss_shot.py \
    --experiment 0407_resample2_iss_shot352_lg \
    --split test \
    --indices 0 10 50 90 \
    --n_sample_pts 30000 \
    --inlier_th 5.0 \
    --ransac_iter 100000
```

---

## 상수 (기존 유지)

- CLIP_START = 0.1, CLIP_END = 1000.0
- ORIG_SIZE = 5761
- CROP_X0 = 1129, CROP_Y0 = 1081, CROP_SIZE = 3502
- GRID_DX = GRID_DY = 0.05, GRID_DZ = 0.02
