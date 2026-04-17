# ISS + FPFH + LightGlue Pipeline (0402)

## 요약

resample_2 depth 데이터에서 ISS(3D keypoint detector) + FPFH(3D descriptor) + LightGlue(matcher) 파이프라인 구현.
기존 SP+FPFH+LG(0331)와 동일 구조에서 detector를 SP→ISS로 교체하고, GT를 `outputs_txt`(ISS 기반)로 변경.

## 배경

- SP+LG(0328): recall 0.744 (ep24)
- SP+FPFH+LG(0331): recall 0.482 (ep11) — SP 2D descriptor 대비 FPFH 3D descriptor 성능 열위
- ISS GT(`outputs_txt`): pair당 ~4,180 valid correspondences (기존 `outputs_resample_2`의 24배)
- ISS detector는 3D 기하학적으로 의미 있는 keypoint를 검출하므로 FPFH descriptor와 일관성 높음

## 아키텍처

```
[Precompute (1회)]
depth image → crop(1129,1081,3502×3502) → (u,v,depth_real) point cloud
  → ISS keypoint 검출 (max 512, 부족 시 random 보충)
  → Dense FPFH 계산 → ISS keypoint 위치에서 KDTree lookup
  → .npz 캐시 저장 (keypoints[resized], keypoint_scores, fpfh_descriptors, n_valid)

[학습/추론]
.npz 캐시 로드 → pass-through extractor → LightGlue matcher
GT: outputs_txt/combination.csv + outputs_txt/{idx}.csv
```

## 데이터

- 이미지: `dataset_resample_2/depth_raw_*.png` (641개, 5761×5761, uint16)
- GT: `outputs_txt/` (64,000 pairs, ISS matching 결과)
  - combination.csv: 상대경로 (`dataset_resample_2/...`, `outputs_txt/...`)
  - CSV 컬럼: master_x, master_y, input_x, input_y, occluded
  - Valid/pair: ~4,180개 (occluded 27.1%)
  - 좌표: 원본 이미지 기준 (crop offset 필요)
- 고정 crop: (1129, 1081), 3502×3502
- resize: 3502 → 1751

## ISS Keypoint 설정

- Open3D `compute_iss_keypoints` 사용
- (u, v, depth_real) 좌표계 — 카메라 intrinsic 불필요
- depth_real = CLIP_START + (raw/65535) × (CLIP_END - CLIP_START)
- CLIP_START=0.1, CLIP_END=1000.0
- Parameters:
  - gamma_21=0.5, gamma_32=0.5 (기본 0.95에서 완화)
  - salient_radius = 6 × avg_nn_dist
  - non_max_radius = 2 × salient_radius
  - min_neighbors=5
- 분석 결과: 이미지당 평균 668개 검출 (min=476, max=779)
- max_num_keypoints=512, 부족 시 depth>0 픽셀에서 random sampling

## FPFH 설정

- 기존 0331과 동일
- fpfh_radius=5.0, normal_radius=10.0, fpfh_max_nn=100
- Dense point cloud에서 FPFH 계산 후 KDTree로 ISS keypoint 위치 lookup
- L2 normalize, 33차원

## LightGlue 설정

- 기존 0331과 동일
- input_dim=33, descriptor_dim=36, num_heads=3
- filter_threshold=0.1, flash=false, checkpointed=true

## 학습 설정

- gt_radius=6
- batch_size=32 (캐시 방식, 큰 배치 가능)
- epochs=100, lr=1e-4
- best_key=match_recall

## 생성 파일

| 파일 | 역할 |
|---|---|
| `precompute_iss_fpfh_resample2.py` | ISS keypoint + Dense FPFH 캐시 생성 |
| `gluefactory/datasets/mitsubishi_resample2_iss_fpfh_dataset.py` | ISS 캐시 + outputs_txt GT 데이터셋 |
| `gluefactory/configs/0402_resample2_iss_fpfh_lg.yaml` | Config |
| `gluefactory/train_resample2_iss_fpfh.py` | 학습 모듈 |
| `train_resample2_iss_fpfh_0402.sh` | 학습 스크립트 |
| `test_resample2_iss_fpfh_0402.py` | 테스트/시각화 |

## 기존 코드 재사용

- `superpoint_fpfh_cached.py`: pass-through extractor (detector 무관, 그대로 사용)
- `gt_pair_matcher.py`: GT matcher (그대로 사용)
- `train_resample2_fpfh.py`: 학습 모듈 (데이터셋/collate만 교체)
- `two_view_pipeline.py`: 파이프라인 (그대로 사용)

## 기존 0331과의 차이 요약

1. Precompute: SP forward → ISS (Open3D), SP 모델 불필요
2. Dataset: `outputs_resample_2` → `outputs_txt` GT 사용, combination.csv 상대경로 처리
3. Keypoint 부족 처리: zero padding → random sampling 보충
4. 나머지 (FPFH, LG, 학습 설정): 동일
