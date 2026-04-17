# 프로젝트 개요

2D Depth 이미지 간 Feature Matching + 3D Registration 연구.
SuperPoint keypoint + LightGlue matcher. FPFH descriptor 실험 포함.

- **conda 환경**: `LightGlue`
- **작업 경로**: `/home/jhs/work/Registration/glue-factory_depth/`

---

# 데이터셋

## Resample_2 (5761×5761)
- 경로: `gluefactory/datasets/mitsubishi/dataset_resample_2/`
- Calib: fx=fy=8001.39, cx=cy=2880.5, clip_start=0.1, clip_end=1000.0, grid_dx=0.05, grid_dy=0.05, grid_dz=0.02
- 고정 crop: (x0=1129, y0=1081, size=3502×3502) → 학습 시 1751px로 리사이즈 (INTER_NEAREST)
- crop 후 intrinsics: cx_crop=1751.5, cy_crop=1799.5, fx=fy=8001.39
- 641개 이미지, combination.csv 기반 페어링
- depth=0 영역 keypoint는 `filter_zero_depth: true`로 마스킹

### Depth 변환
```python
depth_real = clip_start + (raw_uint16 / 65535.0) * (clip_end - clip_start)
depth_real[raw == 0] = 0.0
```
- **단위**: calib INI 파일에 단위 명시 없음. Blender scene unit 상속 (추정: mm, 근거: t_vector Z=400, clip_end=1000 → 산업용 근접 촬영에 합리적)
- **좌표계**: Grid `(u*grid_dx, v*grid_dy, depth_real)` 사용. XYZ `((u-cx)*Z/fx, ...)` 대비 RANSAC 성능이 높음 (매칭 노이즈에 robust)

---

# 파이프라인

## 1. SP + LG (2D descriptor)
SP(frozen) 256D descriptor → LightGlue 매칭

## 2. SP + FPFH + LG (3D descriptor)
SP keypoint 위치에서 Dense FPFH(33D) lookup → LightGlue 매칭.
Sparse FPFH는 descriptor=0 문제로 실패 → **Dense 방식 필수**.

```
Dense point cloud → estimate_normals(r=2*fpfh_r) → compute_fpfh(r=fpfh_r) → KDTree lookup → L2 norm
```

FPFH는 precompute 캐시 사용 (SP frozen이라 매번 같은 keypoint).

## 3. ISS + FPFH(XYZ) + LG (3D detector + view-invariant descriptor)
ISS keypoint 검출(u,v,depth 공간) + 진짜 3D FPFH 매칭(camera intrinsics 사용).

- ISS 검출: (u,v,depth_scaled) 공간 → 2D 이미지 좌표 추출
- FPFH 계산: (u,v,depth) → (X,Y,Z) 변환 후 진짜 3D PCD에서 FPFH 계산
- 이유: FPFH는 rigid transform invariance가 (X,Y,Z) 공간에서만 성립.
  (u,v,depth) 공간에서는 positive pair sim ≈ random pair sim → 학습 불가

```
ISS 검출용  PCD: (u, v, depth_scaled)            → ISS keypoints (u,v) 위치 추출
FPFH 계산용 PCD: (X, Y, Z) using intrinsics      → estimate_normals → compute_fpfh
kp_crop (u,v) → kp_xyz (X,Y,Z) via depth lookup → KDTree lookup → L2 norm
```

캐시: `iss_fpfh_resample2_cache_r{radius}_xyz/`

---

# 주요 파일

## Resample 실험
| 파일 | 역할 |
|---|---|
| `train_resample.sh` | SP+LG 학습 (GPU, EXP, IMAGE_SIZE, GT_RADIUS, MAX_KP, BATCH, --restore) |
| `train_resample_fpfh.sh` | SP+FPFH+LG 학습 (캐시 자동 생성, --restore) |
| `precompute_fpfh_resample.py` | Dense FPFH 캐시 생성 (2880px) |
| `test_resample.py` | 매칭 시각화 (4색: skyblue/purple/limegreen/red) |
| `test_registration_resample.py` | 3D Registration (RANSAC+SVD) |
| `gluefactory/configs/resample_sp_lg.yaml` | SP+LG config (gt_radius=11) |
| `gluefactory/configs/resample_sp_fpfh_lg.yaml` | FPFH config (input_dim=33→36) |

## Resample_2 실험 (현재 활성)
| 파일 | 역할 |
|---|---|
| `train_resample2_iss_fpfh_0402.sh` | ISS+FPFH(XYZ)+LG 학습 (GPU, EXP, FPFH_RADIUS, IMAGE_SIZE, GT_RADIUS, BATCH, --restore) |
| `precompute_iss_fpfh_resample2.py` | ISS keypoint + XYZ FPFH 캐시 생성 (1751px, intrinsics 사용) |
| `test_resample2_iss_fpfh_0402.py` | 매칭 시각화 |
| `test_registration_resample2_iss_fpfh_0406.py` | 3D Registration (RANSAC+SVD, ISS+FPFH+LG, outputs_txt combo 사용) |
| `test_registration_resample2_iss_shot_3D_ransac_rmse.py` | ISS+SHOT+LG, Open3D RANSAC + RMSE 산출 (참고용, 현재 미사용) |
| `test_registration_resample2_iss_shot.py` | ISS+SHOT+LG, Custom SVD RANSAC (1000 iter) — **eval 스크립트의 기준 구현** |
| `eval_registration_iss_shot.py` | ISS+SHOT+LG 양방향 Transform RMSE 평가 |
| `eval_registration_iss_fpfh.py` | ISS+FPFH+LG 양방향 Transform RMSE 평가 |
| `gluefactory/configs/0402_resample2_iss_fpfh_lg.yaml` | ISS+FPFH config (input_dim=33, gt_radius=20) |
| `gluefactory/datasets/mitsubishi_resample2_iss_fpfh_dataset.py` | Resample_2 ISS+FPFH 데이터셋 |
| `gluefactory/train_resample2_iss_fpfh.py` | 학습 루프 |

### Registration 평가 스크립트 (`eval_registration_iss_*.py`)
- **좌표계**: Grid `(u*GRID_DX, v*GRID_DY, depth_real)` 사용 — RANSAC, SVD, RMSE, overlay 모두 동일 공간
  - XYZ `((u-cx)*Z/fx, ...)` 버전도 존재(`eval_registration_xyz.py`)하나, Grid 대비 RANSAC 성능 저하 (매칭 노이즈에 민감)
  - 원인: XYZ에서 depth 오차가 X,Y에 coupling (perspective 효과), Grid는 depth와 x,y가 독립
- **RANSAC**: Custom SVD RANSAC (`ransac_rigid`, 1000 iter, inlier_th=5.0) — Open3D 미사용
- **T_gt**: GT CSV 비-occluded 대응점 → Grid 좌표 변환 → SVD
- **RMSE**: `T_est vs T_gt` 30K 샘플 포인트 클라우드, 양방향(forward+reverse) 평균. 단위는 Grid 좌표 단위
- **테스트**: `tests/test_registration_eval.py` (pixel_to_grid3d, rigid_transform_svd, compute_transform_rmse)


## 공통
| 파일 | 역할 |
|---|---|
| `gluefactory/models/two_view_pipeline.py` | 파이프라인 (filter_zero_depth 포함) |
| `gluefactory/models/extractors/superpoint_open.py` | SP extractor |
| `gluefactory/models/extractors/superpoint_fpfh_cached.py` | FPFH pass-through |

### 캐시 경로
```
gluefactory/datasets/mitsubishi/fpfh_cache_r{r}/               # depth용
gluefactory/datasets/mitsubishi/fpfh_resample_cache_r{r}/       # resample용
gluefactory/datasets/mitsubishi/iss_fpfh_resample2_cache_r{r}_xyz/  # resample_2 ISS+FPFH(XYZ)
```
결과: `outputs/training/{EXPERIMENT}/`, 시각화: `results/`

---

# 성능

## Resample (5761→2880)
| 실험 | recall | 비고 |
|---|---|---|
| resample_sp_lg (SP 2D) | ~0.780 | E13, gt_radius=11 |
| 0323_resample_sp_fpfh_lg (FPFH r=0.5) | ~0.145 | E5 |
| 0324_resample_sp_hybrid_lg (Hybrid r=0.5) | ~0.263 | E0 |

### Resample 3D 거리 참고 (2880px)
- Dense cloud: ~100만점, 1-NN ≈ 0.097
- SP keypoint간: 1-NN median 0.4~1.2, 5-NN median 0.8~3.9
- gt_radius=11px ≈ ~1 (Blender scene unit, depth ~370 기준)

## Resample_2 (5761→1751, crop 3502×3502)
| 실험 | recall | 비고 |
|---|---|---|
| 0402_resample2_iss_fpfh_lg (fake 3D) | 0.0 | E0, gt_radius=6, positive sim≈random sim → 학습 불가 |
| 0402_resample2_iss_fpfh_xyz_lg | 진행 중 | gt_radius=20, FPFH in true (X,Y,Z) |

### 0402 실험 핵심 결정사항
- **ISS**: (u,v,depth_scaled) 공간에서 검출 유지 — 위치 검출에는 충분
- **FPFH**: (X,Y,Z) 진짜 3D에서 계산 — rigid transform invariance 보장
- **gt_radius**: 6→20px (6px에서 positive pair 3.7%로 너무 적음)
- **fake 3D FPFH 실패 원인**: positive pair cosine sim=0.702 ≈ random sim=0.716 → discriminative 불가

### GT/Crop 검증 결과 (2026-04-06 확인)
- **occluded 컬럼**: pandas가 `bool` 타입으로 읽음 → `== False` 정상 동작 (문자열 비교 버그 없음)
- **in_bounds 비율**: 샘플 10개 기준 100% — GT 쌍이 crop 범위 밖으로 나가는 경우 없음
- **GT 좌표계**: 원본(5761×5761) 기준 → crop offset(1129, 1081) 빼기 → scale(×0.5) 순서 정확
- **keypoint 좌표계**: 캐시에 resized(1751px) 기준으로 저장 → GT와 동일 공간 ✓

### DataLoader 설정 (0402_resample2_iss_fpfh_lg.yaml / train_resample2_iss_fpfh.py)
- `num_workers: 12` (병렬 데이터 로딩, 2026-04-06 4→12 변경)
- `pin_memory=True` (CPU→GPU 전송 속도 향상, train/val 모두 적용)

### Resample_2 3D 거리 참고 (1751px, crop 3502→1751)
- ISS keypoints: mean ~5823개 검출, 512개로 subsampling
- gt_radius=20px → unique matched kp ~37%, gt_radius=6px → ~3.7%
- depth 단위: Blender scene unit (CLIP_START=0.1, CLIP_END=1000.0), 원본에 단위 명시 없음 (추정 mm)