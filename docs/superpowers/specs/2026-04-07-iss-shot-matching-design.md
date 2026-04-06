# ISS + SHOT Feature Matching 설계

**날짜**: 2026-04-07
**목표**: 기존 ISS+FPFH 파이프라인에서 descriptor를 FPFH(33D) → SHOT(336D)으로 교체.
먼저 discriminability 검증 후 전체 학습 진행.

---

## 배경 및 동기

- 기존 FPFH fake-3D 실패: positive sim ≈ random sim → 학습 불가
- ISS+FPFH(XYZ) 실험 진행 중 (`0402_resample2_iss_fpfh_xyz_lg`)
- SHOT은 local reference frame(LRF) 기반으로 FPFH 대비 강한 회전 불변성 제공
- **전략**: cosine sim 검증 먼저 → pass 시 전체 파이프라인 구축

---

## 설계 결정사항

| 항목 | 결정 | 비고 |
|---|---|---|
| Keypoint detector | ISS (기존 동일) | (u,v,depth_scaled) 공간 |
| Descriptor | SHOT (pyshot) | XYZ 공간, 336D |
| Mesh 생성 방식 | Grid mesh | 인접 픽셀 삼각형 연결 |
| SHOT radius | 10.0mm | FPFH 5.0mm 대비 2배 |
| n_bins | 20 | → 16×21 = 336D |
| use_normalization | True | L2 정규화 |
| 검증 방식 | Cosine sim (positive vs random) | 학습 전 게이트 |

---

## 섹션 1: `validate_shot.py`

**목적**: SHOT descriptor discriminability 검증.

**동작 흐름**:
```
combination.csv에서 N개 이미지 쌍 샘플링 (기본 10쌍)
  ↓
각 이미지: crop → XYZ PCD → grid mesh → pyshot.get_descriptors (336D)
  ↓
ISS keypoint (X,Y,Z) 위치에서 KDTree lookup → keypoint별 SHOT descriptor
  ↓
GT 매칭 좌표로 positive pair descriptor 추출 (이미지 A kp ↔ 이미지 B kp)
  ↓
같은 이미지 내 랜덤 쌍 = random pair
  ↓
cosine sim 분포 출력: mean ± std, histogram
```

**성공 기준**: positive sim이 random sim보다 유의미하게 높을 것 (목표: +0.1 이상).
실패 시 radius, n_bins 조정 후 재검증.

---

## 섹션 2: Grid Mesh 구성 + SHOT 계산

**`build_grid_mesh(depth_crop_raw, fx, fy, cx, cy)` 함수**:
1. valid pixel mask 생성 (depth > 0)
2. vertex index map: (u,v) → 정수 인덱스
3. verts: (N, 3) — (X, Y, Z) in mm (카메라 intrinsics 사용)
4. faces: 각 2×2 블록에서 4 corner 모두 유효 시 삼각형 2개 생성
   - `(u,v)-(u+1,v)-(u,v+1)` 및 `(u+1,v)-(u+1,v+1)-(u,v+1)`

**`compute_shot_for_keypoints(verts, faces, kp_xyz, n_valid, shot_radius, n_bins)` 함수**:
1. `pyshot.get_descriptors(verts, faces, radius=shot_radius, local_rf_radius=shot_radius, n_bins=n_bins, use_normalization=True)`
2. KDTree(verts) → ISS keypoint XYZ 위치에서 nearest neighbor lookup
3. 결과: (max_num, 336), L2 normalized

**파라미터**:
| 파라미터 | 값 |
|---|---|
| radius | 10.0mm |
| local_rf_radius | 10.0mm |
| n_bins | 20 → 336D |
| min_neighbors | 4 |
| use_normalization | True |

---

## 섹션 3: `precompute_iss_shot_resample2.py` 및 캐시

기존 `precompute_iss_fpfh_resample2.py`에서 최소 변경:

**변경점**:
| 기존 (FPFH) | 신규 (SHOT) |
|---|---|
| `compute_fpfh_for_keypoints()` | `compute_shot_for_keypoints()` |
| `build_xyz_pcd()` → `pcd` | `build_grid_mesh()` → `verts, faces` |
| `fpfh_radius=5.0` | `shot_radius=10.0` |
| 출력: 33D | 출력: 336D |
| 캐시 키: `fpfh_descriptors` | 캐시 키: `shot_descriptors` |

**재사용 함수** (수정 없음):
- `depth_crop_to_pcd()`, `extract_iss_keypoints()`, `select_keypoints()`, `kp_crop_to_xyz()`

**캐시 경로**: `gluefactory/datasets/mitsubishi/iss_shot_resample2_cache_r{radius}/`

**캐시 npz 구조**:
```
keypoints         (512, 2)   — resized 좌표
keypoint_scores   (512,)     — ISS=1.0, random=0.5, pad=0.0
shot_descriptors  (512, 336)
n_valid           int
n_iss             int
```

---

## 섹션 4: 학습 파이프라인

**신규 파일** (기존 파일 수정 없음):

| 파일 | 기반 | 변경점 |
|---|---|---|
| `gluefactory/configs/0407_resample2_iss_shot_lg.yaml` | `0402_resample2_iss_fpfh_lg.yaml` | `input_dim: 33 → 336` |
| `gluefactory/datasets/mitsubishi_resample2_iss_shot_dataset.py` | `mitsubishi_resample2_iss_fpfh_dataset.py` | `fpfh_descriptors → shot_descriptors`, 캐시 경로 |
| `gluefactory/train_resample2_iss_shot.py` | `gluefactory/train_resample2_iss_fpfh.py` | dataset 클래스명 |
| `train_resample2_iss_shot_0407.sh` | `train_resample2_iss_fpfh_0331.sh` | config/experiment명 |

**전체 실행 순서**:
```bash
# Step 1: 검증 (필수 게이트)
python validate_shot.py

# Step 2: 통과 시 전체 precompute
python precompute_iss_shot_resample2.py --shot_radius 10.0

# Step 3: 학습
bash train_resample2_iss_shot_0407.sh
```

---

## 파일 변경 범위 요약

```
신규 생성:
  validate_shot.py
  precompute_iss_shot_resample2.py
  gluefactory/configs/0407_resample2_iss_shot_lg.yaml
  gluefactory/datasets/mitsubishi_resample2_iss_shot_dataset.py
  gluefactory/train_resample2_iss_shot.py
  train_resample2_iss_shot_0407.sh

기존 파일: 수정 없음 (FPFH 실험과 완전 독립)
```
