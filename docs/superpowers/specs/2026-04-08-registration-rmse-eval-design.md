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

- RANSAC으로 T_est(4x4) 추정 후, INI의 GT 변환 T_gt와 비교
- 포인트 클라우드 샘플링 기반 RMSE → 매칭 pair 수와 무관

---

## 파이프라인

```
[모델 추론] → [매칭 pairs] → [3D back-projection (K)] → [RANSAC → T_est]
                                                              |
[INI 파싱] → [T_gt 계산] ──────────────────────────→ [포인트 클라우드 샘플링]
                                                              |
                                                    [T_est vs T_gt RMSE]
                                                              |
                                              [reverse pair로 반복 → 양방향 평균]
```

---

## 좌표계: 카메라 3D (Back-projection)

기존 `(u*dx, v*dy, depth_real)` 대신, K matrix로 진짜 카메라 3D 좌표 사용.

```python
Z = clip_start + (raw_uint16 / 65535.0) * (clip_end - clip_start)
X = (u - cx) * Z / fx
Y = (v - cy) * Z / fy
```

- K matrix: fx=fy=8001.39, cx=cy=2880.5 (원본 5761 기준)
- INI의 R/t와 동일 좌표계 → T_est와 T_gt 직접 비교 가능

---

## GT 변환 행렬 (T_gt)

각 이미지의 INI에서 `r_matrix`, `t_vector` 파싱.

```python
# input camera → master camera
R_gt = R_master @ R_input.T
t_gt = t_master - R_gt @ t_input
```

T_gt는 4x4 homogeneous matrix로 구성.

---

## RANSAC

- Open3D `registration_ransac_based_on_correspondence` 사용 (기존과 동일)
- 입력 좌표만 카메라 3D (X, Y, Z)로 변경
- 매칭된 keypoint 쌍을 back-project → correspondence로 전달
- 출력: T_est (4x4)

---

## RMSE 평가

### 포인트 클라우드 샘플링
- Master depth map에서 유효(depth > 0) 픽셀 중 30,000점 균일 샘플링
- 카메라 3D로 back-project → P_master (30000, 3)

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
- fx = fy = 8001.388671875, cx = cy = 2880.5
