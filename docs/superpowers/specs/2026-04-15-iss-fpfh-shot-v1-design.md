# ISS + FPFH/SHOT (v1) + LightGlue — Design Spec

**작성일**: 2026-04-15
**상태**: Draft (사용자 리뷰 대기)
**대체 관계**: 2026-04-13 spec과 같은 데이터셋이지만 keypoint-only descriptor + voxel=1mm 결정으로 분기. 두 spec은 별도 실험 계열로 공존.

---

## 1. 목표

`/mnt/aict_nas/snuailab_datasets/Feature_Matching/dataset_output` (641 scene / 64,000 pair) 데이터셋에 대해 **ISS detector(mm 공간) + FPFH/SHOT descriptor(keypoint-only, voxel=1mm) + LightGlue matcher** 파이프라인을 구축한다. FPFH variant와 SHOT variant를 **별도 학습 실험**으로 분리해 비교한다.

산출물:
- 학습 스크립트 두 세트 (FPFH / SHOT)
- shot_module.cpp 에 keypoint-only SHOT 함수 추가 + rebuild
- Precompute / Dataset / Trainer / Test (matching·registration·RMSE) 스크립트

---

## 2. 데이터셋 구조

### 2.1 디렉토리
```
/mnt/aict_nas/snuailab_datasets/Feature_Matching/dataset_output/
├── combination.csv                  # 64,000 pair + 헤더 (master_zmap_path, input_zmap_path, csv_path)
├── zmap_XXXX.png                    # 641개, 16-bit uint16 depth
├── pcd_XXXX.ply                     # 641개, point cloud (object frame)
├── config_XXXX.yaml                 # 641개, per-scene metadata
├── pair_{master}_{input}.csv        # 64,000개, GT 픽셀 매칭 (master_x/y, input_x/y, occluded)
└── visualizations/                  # 학습 무관
```

### 2.2 zmap shape
- **width 가변** 2601~2874 (scene별 카메라 위치/스캔 영역 차이)
- **height 일정** 2413
- → batch 통일 필요 (2.5절 결정)

### 2.3 좌표계 / 단위 (mm 명시)

| Frame | 식 | 용도 |
|---|---|---|
| **Camera/sensor frame** (PNG-native) | `X = u·0.056, Y = v·0.056, Z = raw·0.0085` (mm) | ISS, FPFH, SHOT 모두 |
| **Object/world frame** (pcd.ply) | `R_cam·X_cam + t_cam` (`config.camera_rt` 적용) | Registration 평가 |
| **pair CSV `*_x/y`** | 픽셀 (u,v), original zmap 좌표계 | GT 매칭 |

`resolution_mm` 모든 scene 동일: `lateral=0.056, transport=0.056, vertical=0.0085`. Orthographic/parallel projection (intrinsics 없음).

### 2.4 GT 페어링
`pair_*.csv` 5컬럼: `master_x, master_y, input_x, input_y, occluded`. 학습 시 `occluded=False` 필터 + zero-pad 후 좌표 그대로 사용.

### 2.5 Image batch 통일 결정 — Zero-pad
- 가변 width를 동일 (W, H) 로 통일하기 위해 **zero-pad** (resize 안 함)
- 이유: mm/픽셀 비율을 모든 scene에서 동일하게 유지 → ISS saliency 분포 일관
- **Pad target = (W=2432, H=3008)** — 2026-04-15 NAS 641장 스캔: W는 2413 상수, H는 1556~2975 가변. 32 배수 round-up 후 H 여유 33px 확보.
- Pad 영역은 z=0이라 ISS/PCD에서 자연 제외 (별도 mask 처리 불필요)

---

## 3. 핵심 설계 결정 요약

| 결정 | 선택 | 근거 |
|---|---|---|
| FPFH/SHOT 통합 vs 분리 | **분리 variant 2개** | 별도 비교 실험 (앞선 spec과 차이) |
| ISS 검출 좌표계 | **mm `(X,Y,Z)` 공간** | `voxel=1mm, nr=20mm` 등 모든 파라미터가 mm — 일관성 |
| Voxel size | **1.0 mm** | 사용자 명시 ("v1") |
| FPFH normal radius / fpfh radius | 20 mm / 20 mm | 사용자 명시 |
| SHOT normal radius / shot radius | 20 mm / 40 mm | 사용자 명시 |
| Keypoint-only descriptor | **SHOT은 진짜 keypoint-only** (shot_module 수정), **FPFH는 voxel cloud dense → index selecting** | SHOT352 dense 캐시 ~140 MB/img 폭발(약 9 TB 전체) → 진짜 keypoint-only 필수. FPFH는 33-D + 가벼워 dense 후 selecting 비용 무시 가능 |
| Keypoint 수 | 512 (subsample/pad) | 사용자 명시 |
| 512 미달 보충 | **P_vox 위 random sampling**, score=0.0 | 기존 패턴 동일, 좌표가 cache된 voxel cloud 위에 존재해 lookup 일관 |
| Image batch 통일 | **Zero-pad (W=2432, H=3008)** | mm/픽셀 비율 보존 |
| Train/val/test split | **이미지 수준 deterministic, 613/14/14** (zmap_id 기반) | 페어 수준 random은 leakage. val=test=14는 사용자 명시 |
| 네이밍 컨벤션 | `*_v1_20260415_*` (모든 코드/캐시/yaml/sh) | 사용자 지시; "v1"은 voxel_size=1 의미 |

---

## 4. 파이프라인 상세

### 4.1 공통 precompute 흐름

```
zmap_*.png (uint16)
  ├─ load_zmap_to_cloud(lateral=0.056, transport=0.056, vertical=0.0085)
  │     mask = (raw > 0)
  │     PCD_dense = (u·0.056, v·0.056, raw·0.0085) mm  + (u,v) 매핑 보존
  ├─ Open3D VoxelGrid(voxel_size=1.0)
  │     P_vox (M, 3) mm
  ├─ Open3D ISS keypoint detection on P_vox
  │     salient_radius = 6 · avg_nn_dist(P_vox)   (≈ 6 mm)
  │     non_max_radius = 2 · salient_radius
  │     gamma_21 = 0.5, gamma_32 = 0.5, min_neighbors = 5
  │     iss_kp_xyz (K, 3) mm    (K가 가변)
  └─ subsample/pad to 512
        K ≥ 512: random 512, score=1.0
        K < 512: ISS 모두 (score=1.0) + (512−K) random from P_vox \ iss_kp (score=0.0)
        kp_xyz_mm   (512, 3)
        kp_uv_image (512, 2)   = (kp_xyz_mm[:, :2] / 0.056), zero-pad 좌표계 기준
        kp_indices  (512,)     P_vox 내 인덱스 (FPFH selecting용)
```

### 4.2 Branch A — FPFH (voxel cloud dense → index selecting)

```python
# Open3D
pcd_vox.estimate_normals(KDTreeSearchParamRadius(20.0))
fpfh_dense = compute_fpfh_feature(pcd_vox, KDTreeSearchParamRadius(20.0))
# fpfh_dense.data: (33, M)
fpfh_kp = fpfh_dense.data[:, kp_indices].T  # (512, 33)
# 캐시는 raw 저장 (정규화는 trainer 옵션). 단, 모든 0인 row는 valid_mask 처리 권장.
```

### 4.3 Branch B — SHOT (keypoint-only via shot_module 신규 함수)

#### 4.3.1 `shot_module.cpp` 신규 함수
기존 `extract_shot` 유지 + 신규 `extract_shot_at_keypoints` 추가:

```cpp
py::dict extract_shot_at_keypoints(
    py::array_t<float> points_np,     // (N, 3) dense PCD mm
    py::array_t<float> keypoints_np,  // (K, 3) ISS keypoint XYZ mm
    float voxel_size,                  // default 1.0
    float normal_radius,               // default 20.0
    float shot_radius);                // default 40.0
```

내부 흐름:
```
P_vox = VoxelGrid(points_np, voxel_size)
normals = NormalEstimationOMP(P_vox, normal_radius)
NaN normal 제거 → cleanCloud, cleanNormals
kp_cloud = pcl::PointCloud<PointXYZ>(keypoints_np)
SHOTEstimationOMP shot
shot.setSearchSurface(cleanCloud)         // ← 이웃 검색 (dense)
shot.setInputNormals(cleanNormals)
shot.setInputCloud(kp_cloud)              // ← descriptor 계산 위치 (K개)
shot.setSearchMethod(KdTree)
shot.setRadiusSearch(shot_radius)
shot.compute(descriptors)                  // 결과 (K, 352)
NaN descriptor → 0벡터로 채우고 valid_mask=False
```

return:
```python
{
    "descriptors":    (K, 352) float32,   # NaN → 0 처리
    "valid_mask":     (K,) bool,
    "num_input":      int,                  # N
    "num_voxel":      int,                  # P_vox 크기
    "num_keypoints":  int,                  # K
    "num_valid_desc": int,                  # valid_mask.sum()
}
```

#### 4.3.2 빌드
```bash
cd /home/jhs/work/Registration/glue-factory_depth/pybind_shot_linux
# CMakeLists.txt 변경 없음 (단일 cpp)
bash build.sh   # 또는 cmake --build build -j
# 결과 .so 갱신 → import shot_module 시 신규 함수 노출
```

기존 `extract_shot.py` 와 별도로 `extract_shot_at_keypoints.py` 작성하거나 동일 스크립트에 함수 추가 (precompute 스크립트가 직접 호출하므로 별도 CLI 불필요).

### 4.4 Dataset class

#### 4.4.1 `MitsubishiV1ISSFPFHDataset` / `MitsubishiV1ISSSHOTDataset`
구조는 기존 `mitsubishi_resample2_iss_fpfh_dataset.py` 와 거의 동일. 변경점:

| 항목 | 변경 |
|---|---|
| `data_root` | `/mnt/aict_nas/snuailab_datasets/Feature_Matching/dataset_output` |
| combination.csv 컬럼 | `master_zmap_path`, `input_zmap_path`, `csv_path` |
| Crop 단계 제거 | zmap이 이미 ROI 형태 → crop 없음 |
| Resize 단계 제거 | **Zero-pad to (W=2432, H=3008)** |
| GT scale | offset/scale 없음 (원본 zmap 픽셀 좌표 그대로) |
| Cache 경로 | `iss_{fpfh|shot}_v1_20260415_cache_vox1_nr20_{fr20|sr40}/{stem}.npz` |
| 추가 메타 | `config_{stem}.yaml` 의 `camera_rt` 옵션 로드 (registration eval용) |

#### 4.4.2 Split (deterministic, image-level)
```python
# zmap_id ∈ [0, 640]
# 한 번 계산해 splits/v1_20260415_split.json 으로 저장 후 commit (재현성)
rng_val  = np.random.default_rng(42)
val_ids  = sorted(rng_val.choice(641, 14, replace=False).tolist())
remain   = sorted(set(range(641)) - set(val_ids))
rng_test = np.random.default_rng(43)
test_ids = sorted(rng_test.choice(np.array(remain), 14, replace=False).tolist())
train_ids = sorted(set(remain) - set(test_ids))
# 합계: train 613, val 14, test 14
```

페어 필터: `master_zmap_id ∈ split & input_zmap_id ∈ split` (양쪽 모두 같은 split에 속하는 페어만)

#### 4.4.3 GT match 산출
기존 패턴 그대로:
```python
gt = pd.read_csv(pair_csv)
valid = gt["occluded"] == False
master_xy = gt.loc[valid, ["master_x", "master_y"]].values  # 픽셀, zero-pad 좌표 동일
input_xy  = gt.loc[valid, ["input_x",  "input_y" ]].values
gt_matches = np.concatenate([master_xy, input_xy], axis=1)  # (N, 4)
```

(crop offset · resize scale 계산이 모두 사라짐 — zero-pad라 좌표 변환 불필요)

### 4.5 Trainer / config / sh
기존 `train_resample2_iss_fpfh.py` 거의 그대로. yaml에서 `data:`, `model.matcher.input_dim`, cache 경로만 분기.

```yaml
# gluefactory/configs/iss_fpfh_v1_20260415_lg.yaml
data:
  name: mitsubishi_v1_20260415_iss_fpfh
  data_root: /mnt/aict_nas/snuailab_datasets/Feature_Matching/dataset_output
  cache_dir: ./gluefactory/datasets/mitsubishi/iss_fpfh_v1_20260415_cache_vox1_nr20_fr20
  pad_size: [2432, 3008]
  num_workers: 12
  batch_size: 4
model:
  matcher:
    input_dim: 33
train:
  gt_radius: 20    # 기존과 동일 시작값 → 검증 후 조정
  ...
```

```yaml
# gluefactory/configs/iss_shot_v1_20260415_lg.yaml
# 위와 동일하되:
data.name: mitsubishi_v1_20260415_iss_shot
data.cache_dir: ./gluefactory/datasets/mitsubishi/iss_shot_v1_20260415_cache_vox1_nr20_sr40
model.matcher.input_dim: 352
```

### 4.6 Test 스크립트
기존 패턴들 (matching 시각화 / 3D registration / 양방향 RMSE)을 v1 데이터셋·캐시·zero-pad에 맞춰 신규 생성 (FPFH/SHOT 각각):
- `test_iss_fpfh_v1_20260415.py` — 매칭 시각화
- `test_iss_shot_v1_20260415.py` — 매칭 시각화
- `test_registration_iss_fpfh_v1_20260415.py` — 3D registration (Custom SVD RANSAC)
- `test_registration_iss_shot_v1_20260415.py`
- `eval_registration_iss_fpfh_v1_20260415.py` — 양방향 RMSE
- `eval_registration_iss_shot_v1_20260415.py`

Registration 단위는 mm (캐시의 `keypoints_xyz_mm` 사용).

---

## 5. 캐시 npz 명세

### 5.1 FPFH variant — `iss_fpfh_v1_20260415_cache_vox1_nr20_fr20/{stem}.npz`
| 필드 | shape / dtype | 설명 |
|---|---|---|
| `keypoints` | (512, 2) float32 | image (u,v), zero-pad (W=2432,H=3008) 좌표계 |
| `keypoint_scores` | (512,) float32 | ISS 1.0, 보충 0.0 |
| `keypoints_xyz_mm` | (512, 3) float32 | mm 좌표 (registration 평가용) |
| `descriptors` | (512, 33) float32 | FPFH (정규화 X — raw 값. trainer/yaml에서 옵션 적용) |
| `n_iss` | int | ISS 검출 원본 수 |
| `n_valid` | int | 실제 채워진 keypoint 수 |
| `meta` | dict (json bytes) | `{voxel:1.0, normal_r:20.0, fpfh_r:20.0, lateral:0.056, transport:0.056, vertical:0.0085, pad:[2432,3008], dataset:'v1_20260415'}` |

### 5.2 SHOT variant — `iss_shot_v1_20260415_cache_vox1_nr20_sr40/{stem}.npz`
위와 동일하되:
- `descriptors` shape = **(512, 352)**
- `meta` 의 `fpfh_r` → `shot_r=40.0`
- 추가 필드 `valid_mask` (512,) bool — SHOT 계산 성공 여부 (NaN descriptor 추적)

---

## 6. 파일/디렉토리 생성·수정 매핑

### 6.1 신규 파일
| 경로 | 역할 |
|---|---|
| `precompute_iss_fpfh_v1_20260415.py` | FPFH 캐시 생성 |
| `precompute_iss_shot_v1_20260415.py` | SHOT 캐시 생성 (shot_module.extract_shot_at_keypoints 호출) |
| `gluefactory/datasets/mitsubishi_v1_20260415_iss_fpfh_dataset.py` | FPFH dataset class |
| `gluefactory/datasets/mitsubishi_v1_20260415_iss_shot_dataset.py` | SHOT dataset class |
| `gluefactory/configs/iss_fpfh_v1_20260415_lg.yaml` | FPFH config |
| `gluefactory/configs/iss_shot_v1_20260415_lg.yaml` | SHOT config |
| `train_iss_fpfh_v1_20260415.sh` | FPFH 학습 entry |
| `train_iss_shot_v1_20260415.sh` | SHOT 학습 entry |
| `test_iss_fpfh_v1_20260415.py` | matching 시각화 |
| `test_iss_shot_v1_20260415.py` | matching 시각화 |
| `test_registration_iss_fpfh_v1_20260415.py` | 3D registration |
| `test_registration_iss_shot_v1_20260415.py` | 3D registration |
| `eval_registration_iss_fpfh_v1_20260415.py` | 양방향 RMSE |
| `eval_registration_iss_shot_v1_20260415.py` | 양방향 RMSE |
| `gluefactory/datasets/mitsubishi/splits/v1_20260415_split.json` | train/val/test zmap_id 리스트 (재현성) |

### 6.2 수정 파일
| 경로 | 변경 |
|---|---|
| `pybind_shot_linux/shot_module.cpp` | `extract_shot_at_keypoints` 함수 + PYBIND11_MODULE 등록 추가 |
| `pybind_shot_linux/shot_module.cpython-310-...so` | 빌드 갱신 (백업 후) |

### 6.3 변경하지 않는 파일
- `gluefactory/train_resample2_iss_fpfh.py` (트레이너 코드) — config/dataset만 다르게 호출
- 기존 `precompute_iss_fpfh_resample2.py`, 기존 dataset 등 — 그대로 보존

---

## 7. 검증 계획

### 7.1 단위 검증 (precompute 단계)
1. **단일 zmap 스모크 테스트**: `--max_images 1` 옵션으로 zmap_0000 처리. 캐시 npz 필드 존재/shape/dtype 검증.
2. **mm 좌표 일관성**: `keypoints_xyz_mm[:, 0:2]` ≈ `keypoints * 0.056` 확인.
3. **FPFH dense vs keypoint selection 일치**: `fpfh_dense.data[:, kp_indices].T == fpfh_kp` (수치적으로).
4. **SHOT keypoint-only vs dense 비교**: 동일 keypoint에 대해 두 방식 결과 cosine sim ≥ 0.99 (PCL `setSearchSurface` 패턴이 dense 후 lookup과 본질적으로 동일).
5. **valid_mask 비율**: 90% 이상 keypoint에서 SHOT 계산 성공 확인 (그렇지 않으면 normal_radius/shot_radius 재조정).

### 7.2 학습 검증
- 1 epoch 짧은 학습으로 loss 감소 확인
- Val recall이 random baseline(~1/512) 보다 의미 있게 높은지

### 7.3 Registration 평가
- val/test 14 zmap 페어에 대해 RMSE 산출
- 비교 baseline: 기존 `eval_registration_iss_fpfh.py` resample_2 결과

---

## 8. 미해결 사항 / 리스크

| 항목 | 리스크 | 완화책 |
|---|---|---|
| **shot_module rebuild 환경** | `build.sh` 가 PCL/Eigen/OMP 등 의존 — 환경 차이로 빌드 실패 가능 | 빌드 전 환경 문서화. 실패 시 옵션 (A) voxel cloud dense + selecting fallback |
| **1mm voxel 영역 크기** | scene별 영역이 클 경우 voxel cloud M >> 100K → SHOT 계산 시간 ↑ | precompute 첫 zmap에서 M·소요시간 측정. 임계 초과 시 (a) `OMP_NUM_THREADS` 증가, (b) zmap 단위 멀티프로세싱, (c) FPFH variant 학습 우선 진행 후 SHOT은 백그라운드 캐시 생성. **voxel_size=1mm은 고정**이며 조정 대상 아님. |
| **GT match와 ISS keypoint 위치 mismatch** | pair_*.csv 의 master_x/y 는 다른 ISS 구현으로 생성됨 (1px 일치 4%) | gt_radius=20px (기존 결과)로 충분히 흡수. 추가로 ISS 검출 후 GT pair 매칭률 측정 |
| **val/test 14 zmap 너무 적은가** | metric variance | val × ~100 페어 = ~1400 평가샘플로 수치적으로는 충분. 단 zmap 다양성은 제한 |
| **Zero-pad 메모리** | (W=2432, H=3008) batch_size=4 × 2 views → ~234 MB image tensor (float32) | float16/uint16 전환 검토 (LightGlue가 image 자체를 사용 안 하면 dummy 가능) |
| **descriptor L2 norm** | FPFH 정규화 여부에 따라 LightGlue 학습 분포 변화 | 캐시는 raw, trainer/yaml 에서 L2 norm 옵션 (기존 resample_2 trainer 동일 패턴) |

---

## 9. 향후 작업 순서 (writing-plans 단계에서 상세화)

1. shot_module.cpp 신규 함수 추가 + 빌드 + 단위 테스트
2. precompute_iss_fpfh_v1 작성 + 1 zmap 스모크
3. precompute_iss_shot_v1 작성 + 1 zmap 스모크
4. 전체 641 zmap 캐시 생성 (FPFH·SHOT 각각 1회)
5. Dataset class + split 파일 작성 + dataloader 단위 테스트
6. yaml/sh 작성 + 1 epoch 학습 smoke
7. 본 학습 (FPFH / SHOT) + tensorboard 모니터링
8. Test/Eval 스크립트 작성 + RMSE 산출
9. 결과 비교 (FPFH vs SHOT vs 기존 resample_2 baseline)
