# New Dataset: ISS + FPFH/SHOT + LightGlue — Design Spec

**작성일**: 2026-04-13
**상태**: Draft (사용자 리뷰 대기)

## 1. 목표

새로 sampling된 Feature Matching 데이터셋(`/mnt/aict_nas/snuailab_datasets/Feature_Matching/dataset_output`, 641 scene / 64K pair)에 대해 **ISS detector + FPFH/SHOT descriptor + LightGlue matcher** 파이프라인을 구축한다. FPFH·SHOT은 단일 통합 파이프라인에서 config 스위칭으로 지원하며, FPFH 먼저 검증 후 SHOT C++ bin 도착 시 precompute만 추가하여 바로 학습 가능한 구조로 설계한다.

산출물:
- 학습 스크립트(FPFH / SHOT 양대)
- 매칭 시각화 스크립트
- Registration RMSE 평가(스켈레톤)

## 2. 데이터셋 구조 분석 결과

### 2.1 디렉토리 구성

```
dataset_output/
├── combination.csv                # 64,000 pair (master_zmap_path, input_zmap_path, csv_path)
├── zmap_XXXX.png                  # 641개, 16-bit uint16 depth (W=2413 고정, H 가변 1556~2975)
├── pcd_XXXX.ply                   # 641개, binary double PLY (point cloud, 2.4M+ pts, object frame)
├── config_XXXX.yaml               # 641개, per-scene metadata
├── pair_{master}_{input}.csv      # 64,000개, (master_x,y, input_x,y, occluded) in 픽셀
└── visualizations/                # 참고용 시각화 PNG (학습에 사용 안 함)
```

### 2.2 zmap shape 분포 (641 scene)

- **W = 2413 고정** (센서 폭)
- **H = 1556 ~ 2975 가변** (463 unique shapes, 스캔 길이에 비례)
- resolution_mm 모든 scene 동일: `lateral=0.056, transport=0.056, vertical=0.0085`

→ **배치 시 H 변동 대응 필수**.

### 2.3 좌표계 (검증 완료, RMSE 0.0014 mm)

| Frame | 식 | 용도 |
|---|---|---|
| **Camera/sensor frame** (PNG-native) | `X = u·0.056, Y = v·0.056, Z = raw·0.0085` (mm) | ISS, FPFH, SHOT 계산 |
| **Object/world frame** (pcd.ply) | `R_cam @ X_cam + t_cam` (camera_rt 적용) | Registration 평가 (두 view 비교) |
| **pair CSV `*_x/y`** | **픽셀 (u,v)** — resample_2와 동일 | GT 매칭 |

**카메라 모델**: Orthographic/parallel projection (intrinsics 없음, resolution_mm이 pixel pitch 역할). resample_2(perspective, fx/fy/cx/cy 사용)와 구분됨 — X/Y와 Z가 독립이라 Grid 좌표 철학과 자연스럽게 일치.

### 2.4 pair CSV 의미

- 각 `pair_{master}_{input}.csv` 는 master zmap의 ISS keypoint set에 대해 input zmap에서의 대응 위치 기록
- `occluded=0`: input에서 visible → valid GT 매칭
- `occluded=1`: 기하학적으로 가려짐 → GT에서 제외
- 한 master 기준 pair CSV들의 `master_x/y` union = 해당 master zmap의 ISS kp 전체 집합 (예: scene 0 → 852개)
- **기존 Mitsubishi 하이퍼파라미터(gamma_21/32=0.5, min_neighbors=5, r=6·nn)로 ISS 재검출 시 pair CSV와 pixel-perfect로 일치하지 않음** (verification: 1px 일치 4%, 5px 일치 20.8%). 데이터셋은 다른 구현/파라미터로 생성된 것으로 추정.

## 3. 핵심 설계 결정

| 결정 | 선택 | 근거 |
|---|---|---|
| Descriptor 지원 | **FPFH + SHOT 통합 파이프라인** (config 스위칭) | 공통 dataset/train/test 1세트, precompute만 descriptor별 분리 |
| Keypoint 소스 | **ISS 재검출** (pair CSV 사용 안 함) | 기존 Mitsubishi와 동일 방식, 검증된 setup |
| GT 매칭 방식 | `gt_pair_matcher` with `gt_radius` tolerance | 재검출 kp가 pair CSV와 pixel-perfect 불일치하므로 tolerance 필수 |
| 이미지 처리 | **1/2 resize** (aspect 보존, non-square 유지) | 메모리 절약 + resolution 손실 최소화 |
| Batching 시 H 가변 | **collate_fn dynamic padding** (batch 내 max_H로 zero-pad) | 정보 손실 0, 메모리 효율적, 코드 변경 최소 |
| Split | **Scene-based 95/5** (train 609 / val 32, test 별도 없음) | Data leakage 방지, seed=0 고정 |
| 좌표계 표준 | **Camera/sensor frame `(u·0.056, v·0.056, raw·0.0085)` mm** | SHOT C++/Python precompute 통일, pcd.ply 의존 제거 |
| pcd.ply 활용 | **Registration 평가용 only** (precompute엔 사용 안 함) | frame 변환 복잡도 회피 |
| 코드 구조 | **신규 package `gluefactory/datasets/new_dataset/`** | 기존 Mitsubishi 코드 수정 0 (진행 중인 실험 보호) |

## 4. 하이퍼파라미터 (기존 Mitsubishi 값 최대한 유지)

| 파라미터 | 값 | 근거 |
|---|---|---|
| `max_num_keypoints` | 512 | 기존 동일 |
| ISS `gamma_21`, `gamma_32` | 0.5, 0.5 | ISS 고유 |
| ISS `min_neighbors` | 5 | ISS 고유 |
| ISS `salient_r`, `non_max_r` | `6·nn_avg`, `2·salient_r` | 자동 계산 |
| `erode_boundary` | 5 | boundary artifact 억제 |
| `fpfh_radius` | **10.0 mm** | 사용자 경험 (Mitsubishi 최근 관찰에서 양호) |
| `fpfh_normal_radius` | **20.0 mm** (2× fpfh_r) | 기존 convention, config 노출해서 튜닝 가능 |
| SHOT C++ 파라미터 | `voxel=1.0, normal_r=5.0, shot_r=10.0` 고정 | C++ 내부 spec, 건드리지 않음 |
| `gt_radius` | **20 px (1/2 resized 기준)** | 기존 Mitsubishi 검증값, smoke-test 후 조정 가능 |
| LightGlue `input_dim` | 33 (FPFH) / 352 (SHOT) | descriptor 차원 |
| LightGlue `descriptor_dim` (내부) | 36 | 기존 동일 |
| LightGlue `num_heads, filter_threshold, flash, checkpointed` | 3, 0.1, false, true | 기존 동일 |
| `batch_size` | 32 | 1/2 resize라 메모리 여유 |
| `epochs`, `lr`, `lr_schedule` | 100, 1e-4, exp div_10=10 from epoch 20 | 기존 동일 |
| `best_key` | `match_recall` | 기존 동일 |

## 5. 전체 데이터 흐름

### 5.1 Offline precompute (scene 단위)

```
zmap_XXXX.png + config_XXXX.yaml
  │
  ▼
[iss_detection.py]
  ① build_iss_pcd_uvd_scaled(zmap, erode=5)     # (u, v, depth_scaled) PCD
  ② detect_iss_keypoints(γ=.5/.5, minN=5, r=6·nn)
  ③ select_keypoints(max_N=512, random_fill)
     → kp_uv_orig (원본 해상도), kp_resized (1/2)
  │
  ├──▶ [precompute_new_iss_fpfh.py]
  │      build_camera_frame_pcd(zmap)            # (u·lat, v·lat, raw·vert) mm
  │      estimate_normals(r=20.0, max_nn=30)
  │      compute_fpfh_feature(r=10.0, max_nn=100)
  │      KDTree lookup at kp_xyz → (512, 33)
  │
  └──▶ [precompute_new_iss_shot.py]
         load_shot352_bin(<scene_stem>_shot352.bin)  # pts (M,3) mm, desc (M,352)
         KDTree lookup at kp_xyz → (512, 352)
  │
  ▼
cache_new_iss_{fpfh_r10.0 | shot352}/{scene:04d}.npz:
  keypoints(512,2) float32   # 1/2 resized 좌표
  keypoint_scores(512,)      # 1.0 ISS / 0.5 random-fill / 0.0 pad
  descriptors(512, D)        # D=33 or 352
  n_valid(int), n_iss(int)
```

### 5.2 Online training (pair 단위)

```
combination.csv (scene split filtered)
  │
  ▼
NewDatasetISSDescDataset.__getitem__(idx):
  ① zmap 로드 + 1/2 resize (INTER_NEAREST)
  ② master/input cache npz 로드
  ③ pair_####.csv 로드 → occluded==0 필터 → 1/2 scale → in_bounds 필터
  ④ 반환: view0, view1, gt_matches, meta
  │
  ▼
collate_fn_dynamic_pad(batch):
  W=1207 고정, H=batch_max_H 로 zero-pad → (B,1,max_H,1207)
  image_size 는 per-sample actual (H_resized, W_resized)
  │
  ▼
two_view_pipeline:
  extractor = SuperPointFPFHCached (pass-through)
  matcher   = LightGlue (input_dim=33 or 352)
  ground_truth = gt_pair_matcher (gt_radius=20)
  filter_zero_depth: true
  │
  ▼
match_recall / loss
```

### 5.3 Registration 평가 (별도 스크립트, 후속)

```
pair forward → predicted matches (resized u,v)
  → 원본 u,v → cam_xyz → world_xyz (R_cam @ X + t_cam)
  → custom SVD RANSAC → T_est
GT:
  master/input camera_rt → T_gt_rel = T_input_world⁻¹ · T_master_world
T_est vs T_gt 양방향 RMSE (기존 eval_registration_iss_shot.py 패턴)
```

## 6. 디렉토리 구조 & 파일 책임

```
glue-factory_depth/
├── gluefactory/
│   ├── datasets/
│   │   └── new_dataset/                    # 신규 패키지
│   │       ├── __init__.py                 # Dataset 재export
│   │       ├── dataset.py                  # 데이터셋 + collate_fn_dynamic_pad
│   │       ├── coords.py                   # (u,v,raw)↔(X,Y,Z) 변환, camera_rt 유틸
│   │       ├── iss_detection.py            # ISS 검출 + kp 선택
│   │       ├── split.py                    # scene-based split (seed=0)
│   │       └── constants.py                # LAT/TRANS/VERT_MM, RESIZE_FACTOR, DATA_ROOT
│   │
│   ├── configs/
│   │   ├── 0413_new_iss_fpfh_lg.yaml       # FPFH: input_dim=33
│   │   └── 0413_new_iss_shot_lg.yaml       # SHOT: input_dim=352
│   │
│   └── train_new_iss_desc.py               # train_resample2_iss_fpfh.py 기반
│
├── precompute_new_iss_fpfh.py
├── precompute_new_iss_shot.py
├── test_new_iss_desc.py
├── eval_registration_new_iss_desc.py       # 스켈레톤 (후속)
│
├── train_new_iss_fpfh.sh                   # GPU, EXP, FPFH_RADIUS, BATCH, RESTORE
├── train_new_iss_shot.sh                   # GPU, EXP, BATCH, RESTORE
│
└── tests/
    ├── test_new_dataset_coords.py
    ├── test_new_dataset_split.py
    ├── test_new_dataset_iss_detection.py
    └── test_new_dataset_integration.py
```

**기존 자산 재사용 (수정 없음)**:
- `gluefactory/models/extractors/superpoint_fpfh_cached.py`
- `gluefactory/models/matchers/lightglue.py`
- `gluefactory/models/matchers/gt_pair_matcher.py`
- `gluefactory/models/two_view_pipeline.py`
- `gluefactory/visualization/visualize_batch.py::make_match_figures_depth`

## 7. 모듈 인터페이스

### 7.1 `coords.py`

```python
@dataclass
class SceneConfig:
    zmap_shape: tuple[int, int]      # (H, W)
    R_cam: np.ndarray                # (3,3)
    t_cam: np.ndarray                # (3,)

def load_scene_config(scene_id: int, data_root: Path) -> SceneConfig
def pixel_to_cam_xyz(u, v, raw) -> np.ndarray  # (N, 3) mm
def cam_to_world_xyz(xyz_cam, cfg) -> np.ndarray
def build_camera_frame_pcd(zmap, erode_boundary=0) -> (o3d.PointCloud, erode_mask)
```

### 7.2 `iss_detection.py`

```python
def build_iss_pcd_uvd_scaled(zmap, erode_boundary=5) -> (pcd, erode_mask, depth_scale, depth_min)
def detect_iss_keypoints(pcd, gamma_21=0.5, gamma_32=0.5, min_neighbors=5) -> np.ndarray  # (K, 3)
def select_keypoints(iss_kp_3d, zmap, max_num_keypoints=512, erode_mask=None,
                     resize_factor=0.5, rng=None)
    -> (keypoints_resized, scores, n_valid, kp_uv_orig)
```

### 7.3 `split.py`

```python
def scene_split(n_scenes=641, val_ratio=0.05, seed=0) -> (train_scenes, val_scenes)
def pair_filename_to_scene_ids(pair_filename: str) -> (master_id, input_id)
def filter_pairs_by_scenes(combo_df, scenes: set[int]) -> pd.DataFrame
```

### 7.4 `dataset.py`

```python
class NewDatasetISSDescDataset(Dataset):
    def __init__(self, split, cache_dir, data_root=None, resize_factor=0.5,
                 val_ratio=0.05, seed=0): ...
    def __getitem__(idx) -> dict: ...

def collate_fn_dynamic_pad(batch: list[dict]) -> dict
```

## 8. Cache 포맷 (descriptor-agnostic)

```
cache_new_iss_fpfh_r10.0/{scene:04d}.npz
cache_new_iss_shot352/{scene:04d}.npz
scenes_train.txt, scenes_val.txt      # 캐시 루트 공유
precompute_config.json                # CLI args + git hash + 시각
```

각 npz:

| 필드 | shape / dtype | 의미 |
|---|---|---|
| `keypoints` | `(512, 2) float32` | 1/2 resized (u, v) |
| `keypoint_scores` | `(512,) float32` | 1.0 ISS / 0.5 random-fill / 0.0 pad |
| `descriptors` | `(512, D) float32` | D=33 (FPFH) or 352 (SHOT) |
| `n_valid` | int | padding 경계 |
| `n_iss` | int | ISS kp 수 (진단) |

캐시 크기: FPFH 약 43 MB, SHOT 약 462 MB (전 scene).

## 9. SHOT C++ bin 포맷 (기존 spec 재사용 + convention 명시)

- 포맷: `uint32 N, uint32 D=352, per-point [float32 x, y, z, float32[352]]`
- **좌표계 convention (spec lock)**: `X = u·0.056, Y = v·0.056, Z = raw·0.0085` (mm, camera/sensor frame)
- 파일명 규칙: `<scene_stem>_shot352.bin` (예: `zmap_0042_shot352.bin`)
- NaN descriptor 필터링은 precompute 시 적용 (`~np.isnan(desc).any(axis=1)`)

## 10. 엣지 케이스 & 대응

| 케이스 | 대응 |
|---|---|
| n_iss < max_N (ISS 부족) | random fill (scores=0.5), n_valid<3이면 zero-desc + mask |
| 캐시 누락 | Dataset `__init__`에서 검증, `FileNotFoundError` 명시 |
| SHOT bin 누락 | precompute_shot은 SKIP + continue, FPFH 경로 무영향 |
| `input_x/y` out-of-bounds | `in_bounds` 필터 (resized 기준) |
| `occluded=1` | 제외 (`occluded == 0` 필터, int 타입 주의) |
| FPFH KDTree lookup 거리 큼 | 임계(2·fpfh_r) 초과 시 zero-desc + 통계 로깅 |
| batch H 편차 큰 경우 pad 낭비 | 수용, 필요 시 bucketed sampler (후속) |
| reproducibility | seed=0, split 결과 txt로 저장 |

## 11. 검증 전략 (Validate-first)

| Phase | 내용 |
|---|---|
| 0 | Precompute smoke test: scene 0~2만 FPFH 처리, shape/범위/ISS-pair 상관성 리포트 |
| 1 | Dataset smoke test: split 재현성, 10 pair 샘플, collate_fn dynamic pad |
| 2 | Forward pass smoke test: 10 iter, filter_zero_depth 통계, LG 출력 shape |
| 3 | Mini-training: 10 scene × 100 pair, 10 epoch, loss ↓ 확인, positive pair ratio (5~20%) → gt_radius 조정 |
| 4 | Full training: 609 scene, 1 epoch 후 val `match_recall > 0.1` 체크포인트 |
| 5 | SHOT 도착 후: precompute_shot 3 scene smoke → Phase 1~4 config 바꿔 반복 |

## 12. 자동화 테스트

```
tests/
├── test_new_dataset_coords.py
│   - pixel_to_cam_xyz, cam↔world 왕복, yaml 파싱
├── test_new_dataset_split.py
│   - scene_split 재현성, disjoint, filter_pairs 정확성
├── test_new_dataset_iss_detection.py
│   - zero-depth 제외, kp 선택 3케이스 shape·scores
└── test_new_dataset_integration.py
    - 1 scene precompute → cache → __getitem__ → collate 검증 (1분 이내)
```

CI 실행 고려 안 함 (데이터 mount 경로 의존). 로컬 `pytest tests/test_new_dataset_*.py`.

## 13. 롤백·안전 장치

- `gluefactory/datasets/new_dataset/` 은 완전 신규 패키지 (기존 Mitsubishi 파일 수정 0)
- 기존 config `0402_resample2_iss_fpfh_lg.yaml` 건드리지 않음
- 공통 모델 파일(two_view_pipeline, lightglue, gt_pair_matcher, superpoint_fpfh_cached, make_match_figures_depth) 재사용만, 내부 수정 금지

## 14. 실험명 / 출력 경로

- FPFH: `0413_new_iss_fpfh_lg` → `outputs/training/0413_new_iss_fpfh_lg/`
- SHOT: `0413_new_iss_shot_lg` → `outputs/training/0413_new_iss_shot_lg/`
- 테스트 시각화: `results/0413_new_iss_{fpfh|shot}_lg/`

## 15. 성공 기준

- **Precompute**: 641 scene 전부 처리 성공, n_valid=512 비율 > 95%
- **Phase 3 mini-training**: 10 epoch 후 positive pair ratio 5~20% 확보, loss 감소
- **Phase 4 full training**: val `match_recall > 0.1` (1 epoch 시점), 수렴 후 최종 목표는 smoke-test 결과 기반으로 설정
- **시각화**: 기존 4색 convention(skyblue/purple/limegreen/red)으로 매칭 결과가 합리적으로 표시됨

## 16. 후속 (별도 plan)

- Registration RMSE 평가 스크립트 본구현 (현재 spec엔 스켈레톤만)
- Bucketed sampler (H 유사한 pair끼리 batch) — 메모리 낭비 관측 시 고려
- LightGlue `descriptor_dim` 확장 (36 → 256) ablation (SHOT 학습 정체 시)
