# 프로젝트 개요

2D Depth 이미지 간 Feature Matching + 3D Registration 연구.
SuperPoint keypoint + LightGlue matcher. FPFH descriptor 실험 포함.

- **conda 환경**: `LightGlue`
- **작업 경로**: `/home/jhs/work/Registration/glue-factory_depth/`

---

# 데이터셋

## Depth (800×800)
- 경로: `gluefactory/datasets/mitsubishi/dataset_depth/`
- Calib: fx=fy=1111.11, cx=cy=400, clip_start=0.1, clip_end=1000.0
- 분할: ~(전체-200) train / 100 val / 100 test

## Resample (5761×5761)
- 경로: `gluefactory/datasets/mitsubishi/dataset_resample/`
- Calib: fx=fy=8001.39, cx=cy=2880.5, clip_start=0.1, clip_end=1000.0
- 학습 시 2880px로 리사이즈 (INTER_NEAREST)
- 641개 이미지, combination.csv 기반 페어링
- depth=0 영역 keypoint는 `filter_zero_depth: true`로 마스킹

### Depth 변환
```python
depth_real = clip_start + (raw_uint16 / 65535.0) * (clip_end - clip_start)
depth_real[raw == 0] = 0.0
```

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

## 3. SP + Hybrid + LG (2D+3D descriptor)
SP 256D + FPFH 33D → concat 289D → LightGlue 매칭 (input_dim=289).

---

# 주요 파일

## Resample 실험 (현재 활성)
| 파일 | 역할 |
|---|---|
| `train_resample.sh` | SP+LG 학습 (GPU, EXP, IMAGE_SIZE, GT_RADIUS, MAX_KP, BATCH, --restore) |
| `train_resample_fpfh.sh` | SP+FPFH+LG 학습 (캐시 자동 생성, --restore) |
| `train_resample_hybrid.sh` | SP+Hybrid+LG 학습 (캐시 자동 생성, --restore) |
| `precompute_fpfh_resample.py` | Dense FPFH 캐시 생성 (2880px) |
| `precompute_hybrid_resample.py` | Hybrid용 FPFH 캐시 생성 (2880px) |
| `test_resample.py` | 매칭 시각화 (4색: skyblue/purple/limegreen/red) |
| `test_registration_resample.py` | 3D Registration (RANSAC+SVD) |
| `gluefactory/configs/resample_sp_lg.yaml` | SP+LG config (gt_radius=11) |
| `gluefactory/configs/resample_sp_fpfh_lg.yaml` | FPFH config (input_dim=33→36) |
| `gluefactory/configs/resample_sp_hybrid_lg.yaml` | Hybrid config (input_dim=289) |
| `gluefactory/datasets/mitsubishi_resample_hybrid_dataset.py` | Hybrid 데이터셋 |
| `gluefactory/train_resample_hybrid.py` | Hybrid 학습 모듈 |

## Depth 실험 (이전)
| 파일 | 역할 |
|---|---|
| `train_depth_fpfh_dense.sh` | Dense FPFH 학습 |
| `precompute_fpfh_dense.py` | Dense FPFH 캐시 (800px) |
| `test_depth_fpfh.py` / `test_depth.py` | 테스트 |

## 공통
| 파일 | 역할 |
|---|---|
| `gluefactory/models/two_view_pipeline.py` | 파이프라인 (filter_zero_depth 포함) |
| `gluefactory/models/extractors/superpoint_open.py` | SP extractor |
| `gluefactory/models/extractors/superpoint_fpfh_cached.py` | FPFH pass-through |

### 캐시 경로
```
gluefactory/datasets/mitsubishi/fpfh_cache_r{r}/          # depth용
gluefactory/datasets/mitsubishi/fpfh_resample_cache_r{r}/  # resample용
```

---

# 학습/테스트

```bash
# Resample SP+LG
bash train_resample.sh 0 "resample_sp_lg" 2880 11 512 4

# Resample SP+LG resume
bash train_resample.sh 0 "resample_sp_lg" 2880 11 512 4 --restore

# Resample FPFH (캐시 없으면 자동 생성)
bash train_resample_fpfh.sh 0 "0323_resample_sp_fpfh_lg" 0.5 2880 11

# Resample FPFH resume
bash train_resample_fpfh.sh 0 "0323_resample_sp_fpfh_lg" 0.5 2880 11 --restore

# Resample Hybrid
bash train_resample_hybrid.sh 0 "0324_resample_sp_hybrid_lg" 0.5 2880 11

# Resample Hybrid resume
bash train_resample_hybrid.sh 0 "0324_resample_sp_hybrid_lg" 0.5 2880 11 --restore

# Registration 테스트
python3 test_registration_resample.py --experiment resample_sp_lg --indices 0 5 10
```

결과: `outputs/training/{EXPERIMENT}/`, 시각화: `results/`

---

# 성능

## Depth (800×800)
| 실험 | recall | 비고 |
|---|---|---|
| Depth_only_sp2 (SP 2D) | ~0.91 | 수렴 (E32) |
| Dense FPFH r=1.0 | ~0.25 | |
| Dense FPFH r=1.5 | ~0.29 | |
| Dense FPFH r=2.0 | ~0.30 | |

## Resample (5761→2880)
| 실험 | recall | 비고 |
|---|---|---|
| resample_sp_lg (SP 2D) | ~0.780 | E13, gt_radius=11 |
| 0323_resample_sp_fpfh_lg (FPFH r=0.5) | ~0.145 | E5 |
| 0324_resample_sp_hybrid_lg (Hybrid r=0.5) | ~0.263 | E0 |

### Resample 3D 거리 참고 (2880px)
- Dense cloud: ~100만점, 1-NN ≈ 0.097
- SP keypoint간: 1-NN median 0.4~1.2, 5-NN median 0.8~3.9
- gt_radius=11px ≈ **~1mm** (depth 370mm 기준)

---

# 분석 노트북

- `analyze_pointcloud.ipynb` — Dense point cloud KNN/radius 분석
- `analyze_pointcloud_sparse.ipynb` — Sparse vs Dense 비교

# 코드 관리
- 큰 수정이나 버전관리를 위해 새로 코드를 추가할 것인지 recommand 