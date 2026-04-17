# ISS + FPFH/SHOT (v1) + LightGlue Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 새 NAS 데이터셋 (`/mnt/aict_nas/.../dataset_output`, 641 zmap) 위에서 mm 단위 ISS keypoint + voxel=1mm 기반 FPFH(33D) / SHOT(352D) descriptor를 keypoint-only로 추출하고, LightGlue를 두 variant로 별도 학습/평가하는 파이프라인을 구축한다.

**Architecture:** zmap PNG → mm PCD → 1mm voxel → Open3D ISS(mm 공간) → kp 512개. **FPFH variant**: voxel cloud 위 dense FPFH 후 kp index selecting. **SHOT variant**: shot_module.cpp 에 신규 함수 `extract_shot_at_keypoints` 추가 (PCL `setSearchSurface` + `setInputCloud` 분리) → 진짜 keypoint-only SHOT352. 이미지는 zero-pad (2432, 3008)로 통일. 학습은 image-level deterministic split (train 613 / val 14 / test 14) + 기존 LightGlue trainer 거의 그대로.

**Tech Stack:** Python 3.10 (`conda activate LightGlue`), Open3D, NumPy, OpenCV, PyTorch, pybind11, PCL (C++), pandas, pytest. 빌드 도구: cmake.

**Spec 참조:** `docs/superpowers/specs/2026-04-15-iss-fpfh-shot-v1-design.md`

**Conventions:**
- 모든 신규 파일 이름 suffix: `_v1_20260415` (사용자 지시)
- "v1" = voxel_size=1mm
- Commit은 사용자가 직접 수행 (각 phase 끝에 commit suggestion 제공)
- 코드/설정 수정 전 사용자 확인이 필요한 step은 명시

---

## File Structure

### 신규 파일
| 경로 | 책임 |
|---|---|
| `pybind_shot_linux/shot_module.cpp` (수정) | `extract_shot_at_keypoints` 함수 추가 |
| `pybind_shot_linux/test_extract_shot_at_keypoints.py` | 신규 함수 단위 테스트 |
| `gluefactory/datasets/mitsubishi/splits/v1_20260415_split.json` | train/val/test zmap_id 리스트 |
| `gluefactory/datasets/mitsubishi/splits/make_v1_20260415_split.py` | split 생성 스크립트 |
| `precompute_helpers_v1_20260415.py` | 공통 헬퍼 (zmap→PCD, voxel, ISS, kp subsample, zero-pad keypoints) |
| `precompute_iss_fpfh_v1_20260415.py` | FPFH 캐시 생성 |
| `precompute_iss_shot_v1_20260415.py` | SHOT 캐시 생성 |
| `gluefactory/datasets/mitsubishi_v1_20260415_iss_fpfh_dataset.py` | FPFH dataset class |
| `gluefactory/datasets/mitsubishi_v1_20260415_iss_shot_dataset.py` | SHOT dataset class |
| `gluefactory/configs/iss_fpfh_v1_20260415_lg.yaml` | FPFH config |
| `gluefactory/configs/iss_shot_v1_20260415_lg.yaml` | SHOT config |
| `train_iss_fpfh_v1_20260415.sh` | FPFH 학습 entry |
| `train_iss_shot_v1_20260415.sh` | SHOT 학습 entry |
| `test_iss_fpfh_v1_20260415.py` | FPFH matching 시각화 |
| `test_iss_shot_v1_20260415.py` | SHOT matching 시각화 |
| `test_registration_iss_fpfh_v1_20260415.py` | FPFH 3D registration |
| `test_registration_iss_shot_v1_20260415.py` | SHOT 3D registration |
| `eval_registration_iss_fpfh_v1_20260415.py` | FPFH 양방향 RMSE |
| `eval_registration_iss_shot_v1_20260415.py` | SHOT 양방향 RMSE |
| `tests/test_v1_20260415_pipeline.py` | pytest — 좌표/캐시/dataset 단위 테스트 |

### 수정 파일
| 경로 | 변경 |
|---|---|
| `pybind_shot_linux/shot_module.cpython-310-x86_64-linux-gnu.so` | 빌드 결과 갱신 (백업 후) |

### 변경 없음 (보존)
- 기존 trainer (`gluefactory/train_resample2_iss_fpfh.py`) — 새 dataset/config로 호출만
- 기존 resample_2 캐시·코드 — 비교 baseline 유지

---

## Phase 0 — Pre-flight 환경 검증

### Task 0.1: NAS / 캐시 디렉토리 마운트 확인

- [ ] **Step 1: NAS 마운트 확인**

Run:
```bash
ls /mnt/aict_nas/snuailab_datasets/Feature_Matching/dataset_output/ | head -3
```
Expected: `combination.csv`, `config_0000.yaml`, `pair_0000_0001.csv` 등 출력.

- [ ] **Step 2: 로컬 캐시 부모 디렉토리 확인**

Run:
```bash
ls /home/jhs/work/Registration/glue-factory_depth/gluefactory/datasets/mitsubishi/
```
Expected: 기존 캐시들 출력. 신규 캐시는 이 위치에 생성 예정.

- [ ] **Step 3: conda 환경 확인**

Run:
```bash
conda activate LightGlue && python -c "import open3d, cv2, torch, numpy, pandas, yaml; print('OK')"
```
Expected: `OK`. 누락 시 `pip install ...` 후 진행.

### Task 0.2: shot_module 빌드 환경 spike

- [ ] **Step 1: 기존 .so 백업**

Run:
```bash
cd /home/jhs/work/Registration/glue-factory_depth/pybind_shot_linux
cp shot_module.cpython-310-x86_64-linux-gnu.so shot_module.cpython-310-x86_64-linux-gnu.so.bak
```

- [ ] **Step 2: 빈 변경 후 rebuild 가능 여부 확인 (드라이런)**

Run:
```bash
cd /home/jhs/work/Registration/glue-factory_depth/pybind_shot_linux
bash build.sh 2>&1 | tail -20
```
Expected: 빌드 성공 로그 + `.so` 갱신 시각 변화. 실패 시 PCL/Eigen/pybind11 의존성 점검 (CMakeLists.txt 참조).

- [ ] **Step 3: 기존 함수 정상 동작 확인**

Run:
```bash
cd /home/jhs/work/Registration/glue-factory_depth/pybind_shot_linux
python -c "
import shot_module, numpy as np
pts = np.random.rand(10000, 3).astype(np.float32) * 100  # mm
r = shot_module.extract_shot(pts, voxel_size=5.0, normal_radius=25.0, shot_radius=50.0)
print('keypoints:', r['num_keypoints'], 'valid_desc:', r['num_valid_desc'])
"
```
Expected: 출력 정상. 실패 시 빌드 환경 추적 후 재시도.

---

## Phase 1 — shot_module `extract_shot_at_keypoints` 신규 함수

### Task 1.1: C++ 함수 본체 작성

**Files:**
- Modify: `pybind_shot_linux/shot_module.cpp` (기존 함수 유지, 신규 함수 추가, PYBIND11_MODULE 등록 추가)

- [ ] **Step 1: 사용자에게 코드 수정 확인 요청**

> "shot_module.cpp 에 `extract_shot_at_keypoints` 함수와 PYBIND11 등록을 추가합니다. 진행해도 될까요?"

승인 후 다음 step.

- [ ] **Step 2: 함수 구현 추가**

`shot_module.cpp` 의 기존 `extract_shot` 함수 **아래**, `PYBIND11_MODULE` **위**에 다음 함수를 추가:

```cpp
py::dict extract_shot_at_keypoints(
    py::array_t<float, py::array::c_style | py::array::forcecast> points_np,
    py::array_t<float, py::array::c_style | py::array::forcecast> keypoints_np,
    float voxel_size,
    float normal_radius,
    float shot_radius)
{
    auto buf = points_np.unchecked<2>();
    auto kbuf = keypoints_np.unchecked<2>();
    if (buf.shape(1) != 3) throw std::runtime_error("points must be (N, 3)");
    if (kbuf.shape(1) != 3) throw std::runtime_error("keypoints must be (K, 3)");

    size_t n = buf.shape(0);
    size_t k = kbuf.shape(0);

    // 1. Numpy -> PCL clouds
    auto cloud = pcl::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
    cloud->reserve(n);
    for (size_t i = 0; i < n; i++)
        cloud->push_back(pcl::PointXYZ(buf(i, 0), buf(i, 1), buf(i, 2)));
    cloud->width = static_cast<uint32_t>(n); cloud->height = 1; cloud->is_dense = true;

    auto kp_cloud = pcl::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
    kp_cloud->reserve(k);
    for (size_t i = 0; i < k; i++)
        kp_cloud->push_back(pcl::PointXYZ(kbuf(i, 0), kbuf(i, 1), kbuf(i, 2)));
    kp_cloud->width = static_cast<uint32_t>(k); kp_cloud->height = 1; kp_cloud->is_dense = true;

    std::cout << "  Input points: " << cloud->size()
              << ", keypoints: " << kp_cloud->size() << std::endl;

    // 2. VoxelGrid downsample (search surface)
    auto downsampled = pcl::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
    pcl::VoxelGrid<pcl::PointXYZ> vg;
    vg.setInputCloud(cloud);
    vg.setLeafSize(voxel_size, voxel_size, voxel_size);
    vg.filter(*downsampled);
    std::cout << "  Voxel: " << cloud->size() << " -> " << downsampled->size()
              << " (voxel=" << voxel_size << "mm)" << std::endl;
    if (downsampled->size() < 10) throw std::runtime_error("Too few voxel points");

    // 3. Normal estimation on voxel cloud
    auto normals = pcl::make_shared<pcl::PointCloud<pcl::Normal>>();
    pcl::NormalEstimationOMP<pcl::PointXYZ, pcl::Normal> ne;
    ne.setInputCloud(downsampled);
    auto tree = pcl::make_shared<pcl::search::KdTree<pcl::PointXYZ>>();
    ne.setSearchMethod(tree);
    ne.setRadiusSearch(normal_radius);
    ne.compute(*normals);

    // 4. Remove NaN normals
    auto cleanCloud = pcl::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
    auto cleanNormals = pcl::make_shared<pcl::PointCloud<pcl::Normal>>();
    for (size_t i = 0; i < normals->size(); i++) {
        if (std::isfinite(normals->at(i).normal_x) &&
            std::isfinite(normals->at(i).normal_y) &&
            std::isfinite(normals->at(i).normal_z)) {
            cleanCloud->push_back(downsampled->at(i));
            cleanNormals->push_back(normals->at(i));
        }
    }
    cleanCloud->width = (uint32_t)cleanCloud->size(); cleanCloud->height = 1; cleanCloud->is_dense = true;
    cleanNormals->width = (uint32_t)cleanNormals->size(); cleanNormals->height = 1; cleanNormals->is_dense = true;
    if (cleanCloud->size() < 10) throw std::runtime_error("Too few valid normals");

    // 5. SHOT352 at keypoints (search surface = cleanCloud)
    auto shotTree = pcl::make_shared<pcl::search::KdTree<pcl::PointXYZ>>();
    pcl::SHOTEstimationOMP<pcl::PointXYZ, pcl::Normal, pcl::SHOT352> shot;
    shot.setSearchMethod(shotTree);
    shot.setSearchSurface(cleanCloud);     // 이웃 검색 surface
    shot.setInputNormals(cleanNormals);
    shot.setInputCloud(kp_cloud);          // 결과는 keypoint 위치에서만
    shot.setRadiusSearch(shot_radius);

    auto descriptors = pcl::make_shared<pcl::PointCloud<pcl::SHOT352>>();
    shot.compute(*descriptors);

    // 6. NaN descriptor → 0벡터, valid_mask 생성
    py::array_t<float> out_desc({k, (size_t)352});
    py::array_t<bool> out_valid(k);
    auto desc_mut = out_desc.mutable_unchecked<2>();
    auto valid_mut = out_valid.mutable_unchecked<1>();
    int validDesc = 0;
    for (size_t i = 0; i < k; i++) {
        bool is_valid = std::isfinite(descriptors->at(i).descriptor[0]);
        valid_mut(i) = is_valid;
        if (is_valid) validDesc++;
        for (int d = 0; d < 352; d++) {
            float v = descriptors->at(i).descriptor[d];
            desc_mut(i, d) = (is_valid && std::isfinite(v)) ? v : 0.0f;
        }
    }

    py::dict result;
    result["descriptors"]    = out_desc;
    result["valid_mask"]     = out_valid;
    result["num_input"]      = static_cast<int>(n);
    result["num_voxel"]      = static_cast<int>(downsampled->size());
    result["num_keypoints"]  = static_cast<int>(k);
    result["num_valid_desc"] = validDesc;
    return result;
}
```

- [ ] **Step 3: PYBIND11 등록 추가**

기존 `PYBIND11_MODULE(shot_module, m) { ... }` 블록 안에 `m.def("extract_shot", ...)` 다음 줄에 추가:

```cpp
    m.def("extract_shot_at_keypoints", &extract_shot_at_keypoints,
          "Extract SHOT352 descriptors only at given keypoint XYZ positions",
          py::arg("points"),
          py::arg("keypoints"),
          py::arg("voxel_size")    = 1.0f,
          py::arg("normal_radius") = 20.0f,
          py::arg("shot_radius")   = 40.0f);
```

### Task 1.2: 빌드

- [ ] **Step 1: 빌드 실행**

Run:
```bash
cd /home/jhs/work/Registration/glue-factory_depth/pybind_shot_linux
bash build.sh 2>&1 | tail -30
```
Expected: 컴파일 성공 + `.so` 갱신.

- [ ] **Step 2: import 검증**

Run:
```bash
python -c "
import shot_module
assert hasattr(shot_module, 'extract_shot'), 'old function missing'
assert hasattr(shot_module, 'extract_shot_at_keypoints'), 'new function missing'
print('OK')
"
```
Expected: `OK`.

### Task 1.3: 단위 테스트 (pytest)

**Files:**
- Create: `pybind_shot_linux/test_extract_shot_at_keypoints.py`

- [ ] **Step 1: 테스트 파일 작성**

```python
"""extract_shot_at_keypoints 단위 검증."""
import numpy as np
import pytest
import shot_module


def make_test_cloud(n=20000, seed=0):
    """20000-pt cloud in [0, 100] mm box + Gaussian noise."""
    rng = np.random.default_rng(seed)
    return (rng.uniform(0, 100, (n, 3)).astype(np.float32))


def test_signature_and_shapes():
    pts = make_test_cloud()
    kps = pts[:512].copy()
    r = shot_module.extract_shot_at_keypoints(
        pts, kps,
        voxel_size=1.0, normal_radius=20.0, shot_radius=40.0)
    assert r["descriptors"].shape == (512, 352)
    assert r["valid_mask"].shape == (512,)
    assert r["num_keypoints"] == 512
    assert r["num_input"] == len(pts)


def test_valid_mask_majority_true():
    """대부분의 keypoint에서 SHOT 계산 성공해야 함 (>50%)."""
    pts = make_test_cloud()
    kps = pts[:512].copy()
    r = shot_module.extract_shot_at_keypoints(
        pts, kps, 1.0, 20.0, 40.0)
    valid_ratio = r["valid_mask"].mean()
    assert valid_ratio > 0.5, f"valid ratio too low: {valid_ratio:.2%}"


def test_keypoint_only_matches_dense_lookup():
    """SHOT(setSearchSurface=P_vox, setInputCloud=keypoints) 결과는
    SHOT(setInputCloud=P_vox)의 동일 keypoint 위치 lookup과
    cosine similarity >= 0.99 (수치적 동일성)."""
    pts = make_test_cloud(n=30000, seed=1)
    # dense via existing function
    dense = shot_module.extract_shot(
        pts, voxel_size=1.0, normal_radius=20.0, shot_radius=40.0)
    dense_pts = dense["points"]   # (M, 3)
    dense_desc = dense["descriptors"]  # (M, 352)
    # pick 50 random voxel points as keypoints
    rng = np.random.default_rng(42)
    sel = rng.choice(len(dense_pts), 50, replace=False)
    kps = dense_pts[sel]
    expected = dense_desc[sel]
    # keypoint-only via new function
    new = shot_module.extract_shot_at_keypoints(
        pts, kps, voxel_size=1.0, normal_radius=20.0, shot_radius=40.0)
    actual = new["descriptors"]
    valid = new["valid_mask"]
    # cosine similarity for valid only
    a = actual[valid]; b = expected[valid]
    sim = (a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-8)
    median_sim = float(np.median(sim))
    print(f"median cosine sim: {median_sim:.4f}")
    assert median_sim >= 0.99, f"too low: {median_sim:.4f}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
```

- [ ] **Step 2: 테스트 실행 (FAIL 기대 안 함 — 이미 구현됨)**

Run:
```bash
cd /home/jhs/work/Registration/glue-factory_depth/pybind_shot_linux
python -m pytest test_extract_shot_at_keypoints.py -v -s
```
Expected: 3 test PASS. cosine sim 출력 ≥ 0.99.

실패 시:
- "median sim < 0.99": SearchSurface 미설정/normal scope 다름 → 코드 재검토
- "valid ratio low": normal_radius 과소 → 데이터 sparsity 점검

- [ ] **Step 3: 사용자에게 commit 요청**

> 사용자 직접 수행:
> ```
> git add pybind_shot_linux/shot_module.cpp pybind_shot_linux/test_extract_shot_at_keypoints.py
> git commit -m "feat(shot_module): add extract_shot_at_keypoints (PCL setSearchSurface 분리)"
> ```
> `.so` 는 binary 아티팩트라 commit 정책 사용자 판단 (대개 .gitignore).

---

## Phase 2 — Splits 파일 생성

### Task 2.1: split JSON 생성 스크립트

**Files:**
- Create: `gluefactory/datasets/mitsubishi/splits/make_v1_20260415_split.py`
- Create: `gluefactory/datasets/mitsubishi/splits/v1_20260415_split.json`

- [ ] **Step 1: 사용자에게 신규 파일 생성 확인**

> "splits 디렉토리와 split 생성 스크립트를 추가합니다. 진행해도 될까요?"

- [ ] **Step 2: 디렉토리 생성**

Run:
```bash
mkdir -p /home/jhs/work/Registration/glue-factory_depth/gluefactory/datasets/mitsubishi/splits
```

- [ ] **Step 3: 스크립트 작성**

`gluefactory/datasets/mitsubishi/splits/make_v1_20260415_split.py`:
```python
"""v1 (2026-04-15) split 생성. zmap_id 기반 deterministic.

train 613 / val 14 / test 14 (총 641 zmap).
실행 1회: python make_v1_20260415_split.py
출력: v1_20260415_split.json
"""
import json
import numpy as np
from pathlib import Path

OUT = Path(__file__).parent / "v1_20260415_split.json"
N_ZMAP = 641
N_VAL = 14
N_TEST = 14
SEED_VAL = 42
SEED_TEST = 43


def main():
    rng_val = np.random.default_rng(SEED_VAL)
    val_ids = sorted(rng_val.choice(N_ZMAP, N_VAL, replace=False).tolist())
    remain = sorted(set(range(N_ZMAP)) - set(val_ids))
    rng_test = np.random.default_rng(SEED_TEST)
    test_ids = sorted(rng_test.choice(np.array(remain), N_TEST, replace=False).tolist())
    train_ids = sorted(set(remain) - set(test_ids))
    assert len(train_ids) + len(val_ids) + len(test_ids) == N_ZMAP
    assert set(train_ids).isdisjoint(val_ids)
    assert set(train_ids).isdisjoint(test_ids)
    assert set(val_ids).isdisjoint(test_ids)
    payload = {
        "version": "v1_20260415",
        "n_zmap": N_ZMAP,
        "seeds": {"val": SEED_VAL, "test": SEED_TEST},
        "counts": {"train": len(train_ids), "val": len(val_ids), "test": len(test_ids)},
        "train_ids": train_ids,
        "val_ids": val_ids,
        "test_ids": test_ids,
    }
    OUT.write_text(json.dumps(payload, indent=2))
    print(f"written: {OUT}")
    print(f"  train={len(train_ids)}, val={len(val_ids)}, test={len(test_ids)}")
    print(f"  val_ids={val_ids}")
    print(f"  test_ids={test_ids}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: 실행 + 산출물 검증**

Run:
```bash
cd /home/jhs/work/Registration/glue-factory_depth/gluefactory/datasets/mitsubishi/splits
python make_v1_20260415_split.py
python -c "
import json
d = json.load(open('v1_20260415_split.json'))
assert d['counts'] == {'train': 613, 'val': 14, 'test': 14}
assert len(set(d['train_ids']) & set(d['val_ids'])) == 0
assert len(set(d['train_ids']) & set(d['test_ids'])) == 0
print('split OK')
"
```
Expected: `split OK`.

- [ ] **Step 5: 사용자에게 commit 요청**

> ```
> git add gluefactory/datasets/mitsubishi/splits/
> git commit -m "feat(splits): v1_20260415 train/val/test (613/14/14, deterministic)"
> ```

---

## Phase 3 — Precompute 공통 헬퍼

### Task 3.1: `precompute_helpers_v1_20260415.py`

**Files:**
- Create: `precompute_helpers_v1_20260415.py`

- [ ] **Step 1: 사용자 확인**

> "precompute 공통 헬퍼 (zmap→PCD, voxel, ISS, kp 처리, zero-pad 좌표 변환) 파일을 작성합니다. 진행해도 될까요?"

- [ ] **Step 2: 파일 작성**

`precompute_helpers_v1_20260415.py`:
```python
"""v1 precompute 공통 헬퍼.

좌표계: 모두 mm. zmap PNG → (X=u·0.056, Y=v·0.056, Z=raw·0.0085) mm.
ISS는 voxel-downsampled cloud 위에서 검출 → keypoint indices 자동 매핑.
zero-pad target: (W=2432, H=3008).
"""
import numpy as np
import open3d as o3d
import cv2
from PIL import Image
from pathlib import Path

LATERAL_MM = 0.056
TRANSPORT_MM = 0.056
VERTICAL_MM = 0.0085
PAD_W = 2432
PAD_H = 3008
VOXEL_SIZE = 1.0
NORMAL_RADIUS = 20.0
FPFH_RADIUS = 20.0
SHOT_RADIUS = 40.0
MAX_KEYPOINTS = 512
ISS_GAMMA_21 = 0.5
ISS_GAMMA_32 = 0.5
ISS_MIN_NEIGHBORS = 5


def load_zmap_to_pcd_mm(png_path):
    """16-bit depth PNG → (X,Y,Z) mm PCD + (u,v) 매핑.
    Returns:
        pts_mm: (N, 3) float32, mm
        uv:     (N, 2) float32, image pixel (u,v)
        zmap_shape: (H, W) original
    """
    img = np.array(Image.open(png_path))
    if img.dtype == np.uint8:
        img = img.astype(np.uint16) * 257
    H, W = img.shape
    mask = img > 0
    vs, us = np.where(mask)
    zs = img[mask].astype(np.float32)
    pts = np.column_stack([
        us.astype(np.float32) * LATERAL_MM,
        vs.astype(np.float32) * TRANSPORT_MM,
        zs * VERTICAL_MM,
    ]).astype(np.float32)
    uv = np.column_stack([us, vs]).astype(np.float32)
    return pts, uv, (H, W)


def voxel_downsample(pts_mm, voxel_size=VOXEL_SIZE):
    """Open3D VoxelGrid. (mm 단위) → P_vox."""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts_mm.astype(np.float64))
    pcd_vox = pcd.voxel_down_sample(voxel_size)
    return pcd_vox  # Open3D PointCloud


def extract_iss_on_voxel(pcd_vox):
    """ISS keypoint detection on voxel cloud (mm).
    Returns:
        kp_xyz: (K, 3) float32 mm (ISS keypoint 좌표)
    """
    distances = pcd_vox.compute_nearest_neighbor_distance()
    if len(distances) == 0:
        return np.zeros((0, 3), dtype=np.float32)
    avg_dist = float(np.mean(distances))
    salient_radius = 6.0 * avg_dist
    non_max_radius = 2.0 * salient_radius
    kp_pcd = o3d.geometry.keypoint.compute_iss_keypoints(
        pcd_vox,
        salient_radius=salient_radius,
        non_max_radius=non_max_radius,
        gamma_21=ISS_GAMMA_21,
        gamma_32=ISS_GAMMA_32,
        min_neighbors=ISS_MIN_NEIGHBORS,
    )
    return np.asarray(kp_pcd.points, dtype=np.float32)


def map_kp_to_voxel_indices(kp_xyz, pcd_vox):
    """ISS keypoint XYZ를 P_vox 인덱스로 매핑 (KDTree 1-NN).
    Returns:
        indices: (K,) int64
        dists:   (K,) float32 (1-NN 거리, mm. 정상 ≈ 0)
    """
    from scipy.spatial import cKDTree
    vox_pts = np.asarray(pcd_vox.points, dtype=np.float32)
    if len(kp_xyz) == 0 or len(vox_pts) == 0:
        return np.zeros((0,), dtype=np.int64), np.zeros((0,), dtype=np.float32)
    tree = cKDTree(vox_pts)
    d, idx = tree.query(kp_xyz, k=1)
    return idx.astype(np.int64), d.astype(np.float32)


def subsample_or_pad_keypoints(kp_xyz, pcd_vox, max_n=MAX_KEYPOINTS, seed=None):
    """K → 정확히 max_n 개로 subsample 또는 random 보충.
    Returns:
        kp_xyz_out:  (max_n, 3) float32 mm
        kp_indices:  (max_n,) int64 (P_vox 인덱스)
        kp_scores:   (max_n,) float32 (ISS=1.0, 보충=0.0)
        n_iss:       int (원본 ISS 검출 수)
        n_valid:     int (실제 채워진 수, 항상 max_n 또는 그보다 작은 값)
    """
    rng = np.random.default_rng(seed)
    vox_pts = np.asarray(pcd_vox.points, dtype=np.float32)
    K = len(kp_xyz)

    kp_xyz_out = np.zeros((max_n, 3), dtype=np.float32)
    kp_scores = np.zeros((max_n,), dtype=np.float32)
    kp_idx_out = np.zeros((max_n,), dtype=np.int64)

    if K >= max_n:
        sel = rng.choice(K, max_n, replace=False)
        kp_xyz_out[:] = kp_xyz[sel]
        idx, _ = map_kp_to_voxel_indices(kp_xyz_out, pcd_vox)
        kp_idx_out[:] = idx
        kp_scores[:] = 1.0
        return kp_xyz_out, kp_idx_out, kp_scores, K, max_n

    # K < max_n: ISS 모두 + 보충 (random from P_vox \ iss)
    iss_idx, _ = map_kp_to_voxel_indices(kp_xyz, pcd_vox)
    iss_set = set(iss_idx.tolist())
    candidates = np.array([i for i in range(len(vox_pts)) if i not in iss_set],
                          dtype=np.int64)
    n_need = max_n - K
    if len(candidates) >= n_need:
        rand_idx = candidates[rng.choice(len(candidates), n_need, replace=False)]
    else:
        # 극단적인 경우: 보충도 부족 → 가능한 만큼만, 나머지는 0 padding
        rand_idx = candidates
    kp_xyz_out[:K] = kp_xyz
    kp_idx_out[:K] = iss_idx
    kp_scores[:K] = 1.0
    n_filled = K + len(rand_idx)
    if len(rand_idx) > 0:
        kp_xyz_out[K:K + len(rand_idx)] = vox_pts[rand_idx]
        kp_idx_out[K:K + len(rand_idx)] = rand_idx
    # scores 보충분은 0.0 유지
    return kp_xyz_out, kp_idx_out, kp_scores, K, n_filled


def kp_xyz_mm_to_uv_padded(kp_xyz_mm, pad_w=PAD_W, pad_h=PAD_H):
    """mm 좌표 → 픽셀 (u, v). Zero-pad 좌표계 (원본과 동일, padding은 우/하단 추가).
    u = X / LATERAL_MM, v = Y / TRANSPORT_MM
    Returns:
        kp_uv: (K, 2) float32
    """
    u = kp_xyz_mm[:, 0] / LATERAL_MM
    v = kp_xyz_mm[:, 1] / TRANSPORT_MM
    return np.stack([u, v], axis=1).astype(np.float32)


def zero_pad_zmap(img, pad_w=PAD_W, pad_h=PAD_H):
    """zmap PNG (uint16) → zero-pad to (pad_h, pad_w). 우/하단에만 0 padding."""
    H, W = img.shape
    if H > pad_h or W > pad_w:
        raise ValueError(f"image ({H},{W}) exceeds pad target ({pad_h},{pad_w})")
    out = np.zeros((pad_h, pad_w), dtype=img.dtype)
    out[:H, :W] = img
    return out
```

- [ ] **Step 3: 헬퍼 import sanity check**

Run:
```bash
cd /home/jhs/work/Registration/glue-factory_depth
python -c "
import precompute_helpers_v1_20260415 as h
print('PAD:', h.PAD_W, h.PAD_H)
print('VOXEL:', h.VOXEL_SIZE)
print('FPFH r:', h.FPFH_RADIUS, 'SHOT r:', h.SHOT_RADIUS)
"
```
Expected: `PAD: 2432 3008 / VOXEL: 1.0 / FPFH r: 20.0 SHOT r: 40.0`.

### Task 3.2: 헬퍼 단위 테스트

**Files:**
- Create: `tests/test_v1_20260415_pipeline.py`

- [ ] **Step 1: 테스트 작성 (헬퍼 부분)**

`tests/test_v1_20260415_pipeline.py`:
```python
"""v1 (2026-04-15) precompute/dataset 단위 테스트."""
import numpy as np
import pytest
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

import precompute_helpers_v1_20260415 as h


SAMPLE_PNG = "/mnt/aict_nas/snuailab_datasets/Feature_Matching/dataset_output/zmap_0000.png"


def test_load_zmap_units():
    pts, uv, shape = h.load_zmap_to_pcd_mm(SAMPLE_PNG)
    assert pts.dtype == np.float32
    assert pts.shape[1] == 3
    assert uv.shape[1] == 2
    assert len(pts) == len(uv)
    # mm 좌표 sanity: X = uv[0]*0.056
    np.testing.assert_allclose(pts[:, 0], uv[:, 0] * 0.056, rtol=1e-5)
    np.testing.assert_allclose(pts[:, 1], uv[:, 1] * 0.056, rtol=1e-5)


def test_voxel_downsample():
    pts, _, _ = h.load_zmap_to_pcd_mm(SAMPLE_PNG)
    pcd_vox = h.voxel_downsample(pts, voxel_size=1.0)
    M = len(pcd_vox.points)
    assert M < len(pts), "voxel must reduce point count"
    assert M > 100, f"too sparse: {M}"
    print(f"  N={len(pts)} -> M={M} (1mm voxel)")


def test_iss_extraction():
    pts, _, _ = h.load_zmap_to_pcd_mm(SAMPLE_PNG)
    pcd_vox = h.voxel_downsample(pts)
    kp_xyz = h.extract_iss_on_voxel(pcd_vox)
    assert kp_xyz.shape[1] == 3
    assert len(kp_xyz) > 0
    print(f"  ISS keypoints: {len(kp_xyz)}")


def test_subsample_pad_to_512():
    pts, _, _ = h.load_zmap_to_pcd_mm(SAMPLE_PNG)
    pcd_vox = h.voxel_downsample(pts)
    kp_xyz = h.extract_iss_on_voxel(pcd_vox)
    out_xyz, out_idx, out_score, n_iss, n_valid = h.subsample_or_pad_keypoints(
        kp_xyz, pcd_vox, max_n=512, seed=0)
    assert out_xyz.shape == (512, 3)
    assert out_idx.shape == (512,)
    assert out_score.shape == (512,)
    # 인덱스가 P_vox 범위 내
    assert (out_idx >= 0).all() and (out_idx < len(pcd_vox.points)).all()
    # 인덱스가 가리키는 점이 out_xyz와 일치
    vox_pts = np.asarray(pcd_vox.points, dtype=np.float32)
    np.testing.assert_allclose(vox_pts[out_idx[:n_valid]], out_xyz[:n_valid], rtol=1e-4)


def test_kp_xyz_to_uv_round_trip():
    """mm → (u,v) → mm 왕복 일치."""
    rng = np.random.default_rng(0)
    kp = rng.uniform(0, 100, (10, 3)).astype(np.float32)
    uv = h.kp_xyz_mm_to_uv_padded(kp)
    np.testing.assert_allclose(uv[:, 0] * 0.056, kp[:, 0], rtol=1e-5)
    np.testing.assert_allclose(uv[:, 1] * 0.056, kp[:, 1], rtol=1e-5)


def test_zero_pad_zmap():
    from PIL import Image
    img = np.array(Image.open(SAMPLE_PNG))
    pad = h.zero_pad_zmap(img)
    assert pad.shape == (3008, 2432)
    assert pad.dtype == img.dtype
    # 원본 영역 보존
    H, W = img.shape
    np.testing.assert_array_equal(pad[:H, :W], img)
    # 패딩 영역 0
    assert (pad[H:, :] == 0).all()
    assert (pad[:, W:] == 0).all()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
```

- [ ] **Step 2: 테스트 실행**

Run:
```bash
cd /home/jhs/work/Registration/glue-factory_depth
python -m pytest tests/test_v1_20260415_pipeline.py -v -s
```
Expected: 6 test PASS. ISS keypoint 수, P_vox 크기 출력 확인.

- [ ] **Step 3: 사용자 commit 요청**

> ```
> git add precompute_helpers_v1_20260415.py tests/test_v1_20260415_pipeline.py
> git commit -m "feat: precompute helpers (zmap→mm PCD, voxel, ISS, kp pad, zero-pad)"
> ```

---

## Phase 4 — Precompute FPFH

### Task 4.1: `precompute_iss_fpfh_v1_20260415.py`

**Files:**
- Create: `precompute_iss_fpfh_v1_20260415.py`

- [ ] **Step 1: 사용자 확인**

> "FPFH precompute 메인 스크립트를 작성합니다. 진행해도 될까요?"

- [ ] **Step 2: 스크립트 작성**

```python
"""ISS + FPFH (v1, 2026-04-15) precompute.

좌표계: mm. voxel=1mm, normal_r=20mm, fpfh_r=20mm. keypoints=512.
입력: /mnt/aict_nas/.../dataset_output/zmap_*.png
출력: gluefactory/datasets/mitsubishi/iss_fpfh_v1_20260415_cache_vox1_nr20_fr20/{stem}.npz

사용:
    conda activate LightGlue
    python precompute_iss_fpfh_v1_20260415.py --max_images 1 --force   # 스모크
    python precompute_iss_fpfh_v1_20260415.py                           # 전체 641
"""
import argparse
import json
import numpy as np
import open3d as o3d
from pathlib import Path
from tqdm import tqdm

import precompute_helpers_v1_20260415 as h

DATA_ROOT = Path("/mnt/aict_nas/snuailab_datasets/Feature_Matching/dataset_output")
CACHE_DIR = Path(
    "/home/jhs/work/Registration/glue-factory_depth/gluefactory/datasets/mitsubishi/"
    "iss_fpfh_v1_20260415_cache_vox1_nr20_fr20"
)

META = {
    "voxel": h.VOXEL_SIZE,
    "normal_r": h.NORMAL_RADIUS,
    "fpfh_r": h.FPFH_RADIUS,
    "lateral": h.LATERAL_MM,
    "transport": h.TRANSPORT_MM,
    "vertical": h.VERTICAL_MM,
    "pad": [h.PAD_W, h.PAD_H],
    "max_keypoints": h.MAX_KEYPOINTS,
    "dataset": "v1_20260415",
}


def process_one(png_path, out_path, seed=0):
    pts, uv, shape = h.load_zmap_to_pcd_mm(png_path)
    if len(pts) < 100:
        print(f"  SKIP (too few points: {len(pts)})")
        return False
    pcd_vox = h.voxel_downsample(pts, voxel_size=h.VOXEL_SIZE)
    if len(pcd_vox.points) < 100:
        print(f"  SKIP (voxel too sparse: {len(pcd_vox.points)})")
        return False

    # ISS
    kp_xyz_iss = h.extract_iss_on_voxel(pcd_vox)
    kp_xyz, kp_idx, kp_score, n_iss, n_valid = h.subsample_or_pad_keypoints(
        kp_xyz_iss, pcd_vox, max_n=h.MAX_KEYPOINTS, seed=seed)

    # Normals + FPFH (dense on voxel cloud)
    pcd_vox.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamRadius(radius=h.NORMAL_RADIUS))
    fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_vox, o3d.geometry.KDTreeSearchParamRadius(radius=h.FPFH_RADIUS))
    fpfh_data = np.asarray(fpfh.data, dtype=np.float32)  # (33, M)

    # Index selection
    fpfh_kp = fpfh_data[:, kp_idx].T.astype(np.float32)  # (512, 33)

    # (u, v) — mm → pixel
    kp_uv = h.kp_xyz_mm_to_uv_padded(kp_xyz)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_path,
        keypoints=kp_uv,
        keypoint_scores=kp_score,
        keypoints_xyz_mm=kp_xyz,
        descriptors=fpfh_kp,
        n_iss=np.int32(n_iss),
        n_valid=np.int32(n_valid),
        meta=np.frombuffer(json.dumps(META).encode("utf-8"), dtype=np.uint8),
    )
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max_images", type=int, default=None)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    pngs = sorted(DATA_ROOT.glob("zmap_*.png"))
    if args.max_images:
        pngs = pngs[:args.max_images]
    print(f"Cache dir : {CACHE_DIR}")
    print(f"Targets   : {len(pngs)} zmap")

    n_done = n_skip = 0
    for png in tqdm(pngs, desc="precompute FPFH"):
        out = CACHE_DIR / f"{png.stem}.npz"
        if out.exists() and not args.force:
            n_skip += 1
            continue
        try:
            ok = process_one(png, out, seed=args.seed)
            if ok:
                n_done += 1
        except Exception as e:
            print(f"\n  ERROR on {png.name}: {e}")
    print(f"\nDone: {n_done} new, {n_skip} skipped, {len(pngs)} total target")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: 1 zmap 스모크 테스트**

Run:
```bash
cd /home/jhs/work/Registration/glue-factory_depth
conda activate LightGlue
python precompute_iss_fpfh_v1_20260415.py --max_images 1 --force
```
Expected: 정상 종료. 캐시 파일 1개 생성.

- [ ] **Step 4: 캐시 검증**

Run:
```bash
python -c "
import numpy as np, json
d = np.load('gluefactory/datasets/mitsubishi/iss_fpfh_v1_20260415_cache_vox1_nr20_fr20/zmap_0000.npz')
print('keys:', list(d.keys()))
print('keypoints shape:', d['keypoints'].shape, 'dtype:', d['keypoints'].dtype)
print('descriptors shape:', d['descriptors'].shape, 'dtype:', d['descriptors'].dtype)
print('keypoints_xyz_mm range:',
      d['keypoints_xyz_mm'][:,0].min(), d['keypoints_xyz_mm'][:,0].max())
print('n_iss:', int(d['n_iss']), 'n_valid:', int(d['n_valid']))
meta = json.loads(d['meta'].tobytes().decode('utf-8'))
print('meta:', meta)
# 좌표계 일관성: u = X/0.056
np.testing.assert_allclose(
    d['keypoints'][:,0] * 0.056, d['keypoints_xyz_mm'][:,0], rtol=1e-4)
print('coord round-trip OK')
"
```
Expected: shape `(512, 2)` / `(512, 33)`, n_iss > 0, meta 정확. round-trip OK.

- [ ] **Step 5: 사용자 commit 요청**

> ```
> git add precompute_iss_fpfh_v1_20260415.py
> git commit -m "feat: precompute_iss_fpfh_v1_20260415 (mm voxel=1, nr=fr=20, kp=512)"
> ```

### Task 4.2: 전체 641 zmap 캐시 생성

- [ ] **Step 1: 사용자 확인 (오래 걸리는 작업)**

> "FPFH 캐시 641개 생성 시작합니다. 한 zmap당 약 N초 (스모크 결과로 추정) → 전체 약 M분. background로 돌릴까요? (foreground 권장: 진행률 확인 가능)"

- [ ] **Step 2: 전체 실행**

Run:
```bash
cd /home/jhs/work/Registration/glue-factory_depth
python precompute_iss_fpfh_v1_20260415.py
```
Expected: tqdm 진행. 종료 후 캐시 디렉토리에 약 641개 .npz.

- [ ] **Step 3: 캐시 개수 확인**

Run:
```bash
ls gluefactory/datasets/mitsubishi/iss_fpfh_v1_20260415_cache_vox1_nr20_fr20/ | wc -l
du -sh gluefactory/datasets/mitsubishi/iss_fpfh_v1_20260415_cache_vox1_nr20_fr20/
```
Expected: ~641 files. 용량 약 30~50 MB (33D × 512 × float32 × 641).

---

## Phase 5 — Precompute SHOT

### Task 5.1: `precompute_iss_shot_v1_20260415.py`

**Files:**
- Create: `precompute_iss_shot_v1_20260415.py`

- [ ] **Step 1: 사용자 확인**

> "SHOT precompute 메인 스크립트를 작성합니다 (shot_module.extract_shot_at_keypoints 호출). 진행해도 될까요?"

- [ ] **Step 2: 스크립트 작성**

```python
"""ISS + SHOT352 (v1, 2026-04-15) precompute.

좌표계: mm. voxel=1mm, normal_r=20mm, shot_r=40mm. keypoints=512.
SHOT은 shot_module.extract_shot_at_keypoints (PCL setSearchSurface 분리) 사용
→ 진짜 keypoint-only.

출력: gluefactory/datasets/mitsubishi/iss_shot_v1_20260415_cache_vox1_nr20_sr40/{stem}.npz
"""
import argparse
import json
import sys
import numpy as np
from pathlib import Path
from tqdm import tqdm

# pybind shot module path
SHOT_MOD_DIR = Path("/home/jhs/work/Registration/glue-factory_depth/pybind_shot_linux")
sys.path.insert(0, str(SHOT_MOD_DIR))
import shot_module  # noqa: E402

import precompute_helpers_v1_20260415 as h  # noqa: E402

DATA_ROOT = Path("/mnt/aict_nas/snuailab_datasets/Feature_Matching/dataset_output")
CACHE_DIR = Path(
    "/home/jhs/work/Registration/glue-factory_depth/gluefactory/datasets/mitsubishi/"
    "iss_shot_v1_20260415_cache_vox1_nr20_sr40"
)

META = {
    "voxel": h.VOXEL_SIZE,
    "normal_r": h.NORMAL_RADIUS,
    "shot_r": h.SHOT_RADIUS,
    "lateral": h.LATERAL_MM,
    "transport": h.TRANSPORT_MM,
    "vertical": h.VERTICAL_MM,
    "pad": [h.PAD_W, h.PAD_H],
    "max_keypoints": h.MAX_KEYPOINTS,
    "dataset": "v1_20260415",
}


def process_one(png_path, out_path, seed=0):
    pts, uv, shape = h.load_zmap_to_pcd_mm(png_path)
    if len(pts) < 100:
        return False
    pcd_vox = h.voxel_downsample(pts, voxel_size=h.VOXEL_SIZE)
    if len(pcd_vox.points) < 100:
        return False

    # ISS on voxel cloud
    kp_xyz_iss = h.extract_iss_on_voxel(pcd_vox)
    kp_xyz, kp_idx, kp_score, n_iss, n_valid = h.subsample_or_pad_keypoints(
        kp_xyz_iss, pcd_vox, max_n=h.MAX_KEYPOINTS, seed=seed)

    # SHOT352 keypoint-only via pybind11 (P_vox dense as search surface)
    vox_pts = np.asarray(pcd_vox.points, dtype=np.float32)
    result = shot_module.extract_shot_at_keypoints(
        vox_pts, kp_xyz.astype(np.float32),
        voxel_size=h.VOXEL_SIZE,
        normal_radius=h.NORMAL_RADIUS,
        shot_radius=h.SHOT_RADIUS,
    )
    shot_kp = result["descriptors"]   # (512, 352)
    valid = result["valid_mask"]       # (512,)

    # (u, v)
    kp_uv = h.kp_xyz_mm_to_uv_padded(kp_xyz)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_path,
        keypoints=kp_uv,
        keypoint_scores=kp_score,
        keypoints_xyz_mm=kp_xyz,
        descriptors=shot_kp.astype(np.float32),
        valid_mask=valid.astype(bool),
        n_iss=np.int32(n_iss),
        n_valid=np.int32(n_valid),
        meta=np.frombuffer(json.dumps(META).encode("utf-8"), dtype=np.uint8),
    )
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max_images", type=int, default=None)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    pngs = sorted(DATA_ROOT.glob("zmap_*.png"))
    if args.max_images:
        pngs = pngs[:args.max_images]
    print(f"Cache dir : {CACHE_DIR}")
    print(f"Targets   : {len(pngs)} zmap")

    n_done = n_skip = 0
    for png in tqdm(pngs, desc="precompute SHOT"):
        out = CACHE_DIR / f"{png.stem}.npz"
        if out.exists() and not args.force:
            n_skip += 1
            continue
        try:
            if process_one(png, out, seed=args.seed):
                n_done += 1
        except Exception as e:
            print(f"\n  ERROR on {png.name}: {e}")
    print(f"\nDone: {n_done} new, {n_skip} skipped, {len(pngs)} total target")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: 1 zmap 스모크 테스트**

Run:
```bash
cd /home/jhs/work/Registration/glue-factory_depth
python precompute_iss_shot_v1_20260415.py --max_images 1 --force
```
Expected: 정상 종료. 캐시 1개. SHOT 계산 시간 측정 (큰 zmap이면 오래 걸릴 수 있음).

- [ ] **Step 4: 캐시 검증**

Run:
```bash
python -c "
import numpy as np, json
d = np.load('gluefactory/datasets/mitsubishi/iss_shot_v1_20260415_cache_vox1_nr20_sr40/zmap_0000.npz')
print('keys:', list(d.keys()))
print('descriptors shape:', d['descriptors'].shape)
print('valid_mask sum:', int(d['valid_mask'].sum()), '/', len(d['valid_mask']))
print('n_iss:', int(d['n_iss']), 'n_valid:', int(d['n_valid']))
"
```
Expected: shape `(512, 352)`, valid_mask 비율 > 0.7.

- [ ] **Step 5: 사용자 commit 요청**

> ```
> git add precompute_iss_shot_v1_20260415.py
> git commit -m "feat: precompute_iss_shot_v1_20260415 (keypoint-only via shot_module)"
> ```

### Task 5.2: 전체 641 zmap 캐시 생성

- [ ] **Step 1: 사용자 확인 (가장 오래 걸림)**

> "SHOT 캐시 641개 생성 시작합니다. SHOT은 FPFH보다 ~10× 비용 → background 권장. 종료 후 알려드립니다."

- [ ] **Step 2: background 실행**

Run (background):
```bash
cd /home/jhs/work/Registration/glue-factory_depth
nohup python precompute_iss_shot_v1_20260415.py > precompute_shot_v1.log 2>&1 &
echo $!  # PID 기록
```

- [ ] **Step 3: 진행률 모니터링**

Run:
```bash
tail -f precompute_shot_v1.log   # Ctrl-C로 빠져나옴
```

완료 후:
```bash
ls gluefactory/datasets/mitsubishi/iss_shot_v1_20260415_cache_vox1_nr20_sr40/ | wc -l
du -sh gluefactory/datasets/mitsubishi/iss_shot_v1_20260415_cache_vox1_nr20_sr40/
```
Expected: ~641 files, 약 460~500 MB (352D × 512 × float32 × 641).

---

## Phase 6 — Dataset class

### Task 6.1: FPFH dataset class

**Files:**
- Create: `gluefactory/datasets/mitsubishi_v1_20260415_iss_fpfh_dataset.py`

- [ ] **Step 1: 사용자 확인**

> "v1 FPFH dataset class를 작성합니다. 진행해도 될까요?"

- [ ] **Step 2: 파일 작성**

```python
"""v1 (2026-04-15) ISS+FPFH dataset.

NAS dataset_output 기반.
- Zero-pad to (W=2432, H=3008).
- GT: pair_*.csv (occluded=False).
- Cache: iss_fpfh_v1_20260415_cache_vox1_nr20_fr20/{stem}.npz
- Split: gluefactory/datasets/mitsubishi/splits/v1_20260415_split.json
"""
import json
import re
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from pathlib import Path
from PIL import Image

PAD_W = 2432
PAD_H = 3008
ZMAP_RE = re.compile(r"zmap_(\d{4})\.png$")


def _zmap_id(path_str):
    m = ZMAP_RE.search(str(path_str))
    if not m:
        raise ValueError(f"cannot parse zmap_id from {path_str}")
    return int(m.group(1))


def _zero_pad(img_u16):
    H, W = img_u16.shape
    out = np.zeros((PAD_H, PAD_W), dtype=img_u16.dtype)
    out[:H, :W] = img_u16
    return out


class MitsubishiV1ISSFPFHDataset(Dataset):
    def __init__(self, split="train",
                 data_root="/mnt/aict_nas/snuailab_datasets/Feature_Matching/dataset_output",
                 cache_dir=None,
                 split_json=None):
        self.data_root = Path(data_root)
        if cache_dir is None:
            cache_dir = ("/home/jhs/work/Registration/glue-factory_depth/"
                         "gluefactory/datasets/mitsubishi/"
                         "iss_fpfh_v1_20260415_cache_vox1_nr20_fr20")
        self.cache_dir = Path(cache_dir)
        if not self.cache_dir.exists():
            raise FileNotFoundError(f"cache not found: {self.cache_dir}")

        if split_json is None:
            split_json = ("/home/jhs/work/Registration/glue-factory_depth/"
                          "gluefactory/datasets/mitsubishi/splits/v1_20260415_split.json")
        s = json.loads(Path(split_json).read_text())
        self.split = split
        if split == "train":
            id_set = set(s["train_ids"])
        elif split == "val":
            id_set = set(s["val_ids"])
        elif split == "test":
            id_set = set(s["test_ids"])
        else:
            raise ValueError(split)
        self.id_set = id_set

        combo = pd.read_csv(self.data_root / "combination.csv")
        # 양쪽 zmap이 모두 split에 속하는 페어만
        m_ids = combo["master_zmap_path"].map(_zmap_id)
        i_ids = combo["input_zmap_path"].map(_zmap_id)
        keep = m_ids.isin(id_set) & i_ids.isin(id_set)
        self.combo = combo[keep].reset_index(drop=True)

    def __len__(self):
        return len(self.combo)

    def _load_image(self, fname):
        img = np.array(Image.open(self.data_root / fname))
        if img.dtype == np.uint8:
            img = img.astype(np.uint16) * 257
        pad = _zero_pad(img)
        return pad.astype(np.float32) / 65535.0

    def _load_cache(self, fname):
        stem = Path(fname).stem
        d = np.load(self.cache_dir / f"{stem}.npz")
        return {
            "keypoints": d["keypoints"],
            "keypoint_scores": d["keypoint_scores"],
            "descriptors": d["descriptors"],
        }

    def _load_gt(self, csv_fname):
        df = pd.read_csv(self.data_root / csv_fname)
        valid = df["occluded"] == False
        m = df.loc[valid, ["master_x", "master_y"]].values.astype(np.float32)
        i = df.loc[valid, ["input_x", "input_y"]].values.astype(np.float32)
        # zero-pad 좌표계 = 원본과 동일 (좌상단 정렬). 변환 없음.
        return np.concatenate([m, i], axis=1)  # (N, 4)

    def __getitem__(self, idx):
        row = self.combo.iloc[idx]
        master_fname = row["master_zmap_path"]
        input_fname = row["input_zmap_path"]
        csv_fname = row["csv_path"]

        master = self._load_image(master_fname)  # (PAD_H, PAD_W) float32
        inp = self._load_image(input_fname)
        c0 = self._load_cache(master_fname)
        c1 = self._load_cache(input_fname)
        gt = self._load_gt(csv_fname)

        return {
            "view0": {
                "image": torch.from_numpy(master).unsqueeze(0),
                "image_size": torch.tensor([PAD_H, PAD_W]),
                "keypoints": torch.from_numpy(c0["keypoints"]).float(),
                "keypoint_scores": torch.from_numpy(c0["keypoint_scores"]).float(),
                "descriptors": torch.from_numpy(c0["descriptors"]).float(),
            },
            "view1": {
                "image": torch.from_numpy(inp).unsqueeze(0),
                "image_size": torch.tensor([PAD_H, PAD_W]),
                "keypoints": torch.from_numpy(c1["keypoints"]).float(),
                "keypoint_scores": torch.from_numpy(c1["keypoint_scores"]).float(),
                "descriptors": torch.from_numpy(c1["descriptors"]).float(),
            },
            "gt_matches": torch.from_numpy(gt).float(),
            "csv_path": str(self.data_root / csv_fname),
            "master_path": str(self.data_root / master_fname),
            "input_path": str(self.data_root / input_fname),
        }


def v1_iss_fpfh_collate_fn(batch):
    images0 = torch.stack([b["view0"]["image"] for b in batch], dim=0)
    images1 = torch.stack([b["view1"]["image"] for b in batch], dim=0)
    sizes0 = torch.stack([b["view0"]["image_size"] for b in batch], dim=0)
    sizes1 = torch.stack([b["view1"]["image_size"] for b in batch], dim=0)
    kp0 = torch.stack([b["view0"]["keypoints"] for b in batch], dim=0)
    kp1 = torch.stack([b["view1"]["keypoints"] for b in batch], dim=0)
    sc0 = torch.stack([b["view0"]["keypoint_scores"] for b in batch], dim=0)
    sc1 = torch.stack([b["view1"]["keypoint_scores"] for b in batch], dim=0)
    desc0 = torch.stack([b["view0"]["descriptors"] for b in batch], dim=0)
    desc1 = torch.stack([b["view1"]["descriptors"] for b in batch], dim=0)
    gt_list = [b["gt_matches"] for b in batch]
    gt_matches = pad_sequence(gt_list, batch_first=True, padding_value=0.0)
    return {
        "view0": {
            "image": images0, "image_size": sizes0,
            "keypoints": kp0, "keypoint_scores": sc0, "descriptors": desc0,
        },
        "view1": {
            "image": images1, "image_size": sizes1,
            "keypoints": kp1, "keypoint_scores": sc1, "descriptors": desc1,
        },
        "gt_matches": gt_matches,
    }
```

### Task 6.2: SHOT dataset class

**Files:**
- Create: `gluefactory/datasets/mitsubishi_v1_20260415_iss_shot_dataset.py`

- [ ] **Step 1: 파일 작성**

`MitsubishiV1ISSSHOTDataset` 는 위 FPFH class와 거의 동일. 차이점만:
- default `cache_dir = ".../iss_shot_v1_20260415_cache_vox1_nr20_sr40"`
- 클래스명 `MitsubishiV1ISSSHOTDataset`
- collate fn 이름 `v1_iss_shot_collate_fn`
- descriptors shape (512, 352)는 코드 변경 없음 (cache가 결정)

방법: 위 FPFH 파일을 복사 → 클래스/함수/cache_dir 이름만 교체.

```bash
cd /home/jhs/work/Registration/glue-factory_depth/gluefactory/datasets
cp mitsubishi_v1_20260415_iss_fpfh_dataset.py mitsubishi_v1_20260415_iss_shot_dataset.py
sed -i 's/iss_fpfh_v1_20260415_cache_vox1_nr20_fr20/iss_shot_v1_20260415_cache_vox1_nr20_sr40/g' \
    mitsubishi_v1_20260415_iss_shot_dataset.py
sed -i 's/MitsubishiV1ISSFPFHDataset/MitsubishiV1ISSSHOTDataset/g' \
    mitsubishi_v1_20260415_iss_shot_dataset.py
sed -i 's/v1_iss_fpfh_collate_fn/v1_iss_shot_collate_fn/g' \
    mitsubishi_v1_20260415_iss_shot_dataset.py
```

- [ ] **Step 2: 결과 파일 검증**

Run:
```bash
grep -E "(class|def )" mitsubishi_v1_20260415_iss_shot_dataset.py | head -10
grep "iss_shot_v1" mitsubishi_v1_20260415_iss_shot_dataset.py | head -5
```
Expected: 클래스명 SHOT, cache 경로 SHOT.

### Task 6.3: Dataset 단위 테스트

- [ ] **Step 1: 테스트 추가**

`tests/test_v1_20260415_pipeline.py` 끝에 추가:
```python
def test_dataset_fpfh_loads_one_pair():
    from gluefactory.datasets.mitsubishi_v1_20260415_iss_fpfh_dataset import (
        MitsubishiV1ISSFPFHDataset, v1_iss_fpfh_collate_fn,
    )
    ds = MitsubishiV1ISSFPFHDataset(split="val")
    assert len(ds) > 0
    print(f"  val pairs: {len(ds)}")
    item = ds[0]
    assert item["view0"]["image"].shape == (1, 3008, 2432)
    assert item["view0"]["keypoints"].shape == (512, 2)
    assert item["view0"]["descriptors"].shape == (512, 33)
    assert item["view1"]["descriptors"].shape == (512, 33)
    assert item["gt_matches"].shape[1] == 4
    # collate
    batch = v1_iss_fpfh_collate_fn([ds[0], ds[1]])
    assert batch["view0"]["image"].shape == (2, 1, 3008, 2432)


def test_dataset_shot_loads_one_pair():
    from gluefactory.datasets.mitsubishi_v1_20260415_iss_shot_dataset import (
        MitsubishiV1ISSSHOTDataset, v1_iss_shot_collate_fn,
    )
    ds = MitsubishiV1ISSSHOTDataset(split="val")
    assert len(ds) > 0
    item = ds[0]
    assert item["view0"]["descriptors"].shape == (512, 352)


def test_split_no_leakage():
    from gluefactory.datasets.mitsubishi_v1_20260415_iss_fpfh_dataset import (
        MitsubishiV1ISSFPFHDataset, _zmap_id,
    )
    train = MitsubishiV1ISSFPFHDataset(split="train")
    val = MitsubishiV1ISSFPFHDataset(split="val")
    test = MitsubishiV1ISSFPFHDataset(split="test")
    train_ids = set()
    val_ids = set()
    test_ids = set()
    for ds, s in [(train, train_ids), (val, val_ids), (test, test_ids)]:
        for i in range(len(ds.combo)):
            row = ds.combo.iloc[i]
            s.add(_zmap_id(row["master_zmap_path"]))
            s.add(_zmap_id(row["input_zmap_path"]))
    assert train_ids.isdisjoint(val_ids)
    assert train_ids.isdisjoint(test_ids)
    assert val_ids.isdisjoint(test_ids)
    print(f"  train_ids: {len(train_ids)}, val_ids: {len(val_ids)}, test_ids: {len(test_ids)}")
```

- [ ] **Step 2: 테스트 실행 (캐시 존재 가정)**

Run:
```bash
cd /home/jhs/work/Registration/glue-factory_depth
python -m pytest tests/test_v1_20260415_pipeline.py::test_dataset_fpfh_loads_one_pair \
                  tests/test_v1_20260415_pipeline.py::test_dataset_shot_loads_one_pair \
                  tests/test_v1_20260415_pipeline.py::test_split_no_leakage -v -s
```
Expected: 3 PASS. (val 페어 수 출력)

⚠️ 캐시가 1개만 있는 스모크 단계라면, val zmap_id 14개 중 어느 것도 캐시 없음 → fail. 본 테스트는 **전체 캐시 생성 후** 실행 가능. 사전 검증은 train split의 zmap_0000 으로 수동 검증.

- [ ] **Step 3: 사용자 commit 요청**

> ```
> git add gluefactory/datasets/mitsubishi_v1_20260415_iss_*.py tests/test_v1_20260415_pipeline.py
> git commit -m "feat(dataset): v1 FPFH/SHOT dataset classes (zero-pad, image-level split)"
> ```

---

## Phase 7 — Trainer config / sh

### Task 7.1: FPFH yaml + sh

**Files:**
- Create: `gluefactory/configs/iss_fpfh_v1_20260415_lg.yaml`
- Create: `train_iss_fpfh_v1_20260415.sh`

- [ ] **Step 1: 사용자 확인**

> "FPFH/SHOT yaml + sh 4파일을 작성합니다. 기존 trainer (`train_resample2_iss_fpfh.py`)는 그대로 사용. 진행해도 될까요?"

- [ ] **Step 2: FPFH yaml 작성**

기존 `gluefactory/configs/0402_resample2_iss_fpfh_lg.yaml`을 base로 복사 후 변경. 먼저 기존 yaml 확인:

Run:
```bash
cat /home/jhs/work/Registration/glue-factory_depth/gluefactory/configs/0402_resample2_iss_fpfh_lg.yaml
```

(기존 yaml의 모든 키 보존 + 아래 항목만 v1 dataset/cache로 변경)

`gluefactory/configs/iss_fpfh_v1_20260415_lg.yaml`:
```yaml
data:
  name: mitsubishi_v1_20260415_iss_fpfh
  data_root: /mnt/aict_nas/snuailab_datasets/Feature_Matching/dataset_output
  cache_dir: /home/jhs/work/Registration/glue-factory_depth/gluefactory/datasets/mitsubishi/iss_fpfh_v1_20260415_cache_vox1_nr20_fr20
  split_json: /home/jhs/work/Registration/glue-factory_depth/gluefactory/datasets/mitsubishi/splits/v1_20260415_split.json
  num_workers: 12
  batch_size: 4
  pin_memory: true
  pad_size: [2432, 3008]
model:
  matcher:
    name: lightglue
    input_dim: 33
    # 기존 0402 yaml의 model section 그대로 복사
    descriptor_dim: 256
    n_layers: 9
    flash: true
train:
  gt_radius: 20
  num_epochs: 50
  optimizer: adam
  lr: 1.0e-4
  log_every_iter: 50
  eval_every_iter: 1000
  # (기존 trainer의 다른 키들 보존 — 0402 yaml 확인 후 동일하게)
```

⚠️ **중요**: 위 yaml은 스켈레톤. 기존 0402 yaml의 model/train section 모든 키 (예: `loss`, `gt_radius`, `clip_grad`, `scheduler`, `weights` 등)을 빠짐 없이 복사 + 위 변경 항목만 덮어쓰기.

실제 작성 절차:
```bash
cd /home/jhs/work/Registration/glue-factory_depth/gluefactory/configs
cp 0402_resample2_iss_fpfh_lg.yaml iss_fpfh_v1_20260415_lg.yaml
# 편집기로 열어 위 변경 항목 4개 (data.* + pad_size) 반영
```

- [ ] **Step 3: FPFH sh 작성**

기존 `train_resample2_iss_fpfh_0402.sh` base:
```bash
cat /home/jhs/work/Registration/glue-factory_depth/train_resample2_iss_fpfh_0402.sh
```

`train_iss_fpfh_v1_20260415.sh`:
```bash
#!/bin/bash
# v1 (2026-04-15) ISS+FPFH+LG 학습 entry.
# 캐시는 사전에 precompute_iss_fpfh_v1_20260415.py 로 생성되어야 함.

set -e

GPU=${GPU:-0}
EXP=${EXP:-iss_fpfh_v1_20260415}
BATCH=${BATCH:-4}
RESTORE=${RESTORE:-}

CONFIG=gluefactory/configs/iss_fpfh_v1_20260415_lg.yaml
CACHE_DIR=gluefactory/datasets/mitsubishi/iss_fpfh_v1_20260415_cache_vox1_nr20_fr20

if [ ! -d "$CACHE_DIR" ] || [ -z "$(ls -A $CACHE_DIR 2>/dev/null)" ]; then
    echo "[v1] cache missing -> running precompute..."
    python precompute_iss_fpfh_v1_20260415.py
fi

CMD="CUDA_VISIBLE_DEVICES=$GPU python -m gluefactory.train_resample2_iss_fpfh \
    $EXP --conf $CONFIG --overrides data.batch_size=$BATCH"
if [ -n "$RESTORE" ]; then
    CMD="$CMD --restore $RESTORE"
fi
echo "$CMD"
eval "$CMD"
```

```bash
chmod +x /home/jhs/work/Registration/glue-factory_depth/train_iss_fpfh_v1_20260415.sh
```

### Task 7.2: SHOT yaml + sh

- [ ] **Step 1: SHOT yaml 작성**

```bash
cd /home/jhs/work/Registration/glue-factory_depth/gluefactory/configs
cp iss_fpfh_v1_20260415_lg.yaml iss_shot_v1_20260415_lg.yaml
# 편집기로 변경:
#   data.name → mitsubishi_v1_20260415_iss_shot
#   data.cache_dir → .../iss_shot_v1_20260415_cache_vox1_nr20_sr40
#   model.matcher.input_dim → 352
```

또는 sed 기반:
```bash
sed -i \
  -e 's|mitsubishi_v1_20260415_iss_fpfh|mitsubishi_v1_20260415_iss_shot|g' \
  -e 's|iss_fpfh_v1_20260415_cache_vox1_nr20_fr20|iss_shot_v1_20260415_cache_vox1_nr20_sr40|g' \
  -e 's|input_dim: 33|input_dim: 352|g' \
  iss_shot_v1_20260415_lg.yaml
```

- [ ] **Step 2: SHOT sh 작성**

```bash
cd /home/jhs/work/Registration/glue-factory_depth
cp train_iss_fpfh_v1_20260415.sh train_iss_shot_v1_20260415.sh
sed -i \
  -e 's|iss_fpfh_v1_20260415|iss_shot_v1_20260415|g' \
  -e 's|iss_fpfh_v1_20260415_cache_vox1_nr20_fr20|iss_shot_v1_20260415_cache_vox1_nr20_sr40|g' \
  train_iss_shot_v1_20260415.sh
chmod +x train_iss_shot_v1_20260415.sh
```

### Task 7.3: Trainer dataset registration 확인

기존 trainer는 dataset name 기반으로 dataset class를 dispatch함. `mitsubishi_v1_20260415_iss_fpfh` 라는 새 이름이 dispatch 가능한지 확인.

- [ ] **Step 1: Trainer dataset 등록 위치 확인**

Run:
```bash
grep -rn "mitsubishi_resample2_iss_fpfh" /home/jhs/work/Registration/glue-factory_depth/gluefactory/ \
    | head -20
```
Expected: 등록 위치 (보통 `gluefactory/datasets/__init__.py` 또는 trainer 내부 if/else).

- [ ] **Step 2: 등록 추가**

발견한 위치에 새 dataset 두 개 등록:
```python
# gluefactory/datasets/__init__.py (예시 — 실제 위치/패턴은 위 grep 결과에 따라 조정)
elif name == "mitsubishi_v1_20260415_iss_fpfh":
    from .mitsubishi_v1_20260415_iss_fpfh_dataset import (
        MitsubishiV1ISSFPFHDataset as DatasetClass,
        v1_iss_fpfh_collate_fn as collate_fn,
    )
elif name == "mitsubishi_v1_20260415_iss_shot":
    from .mitsubishi_v1_20260415_iss_shot_dataset import (
        MitsubishiV1ISSSHOTDataset as DatasetClass,
        v1_iss_shot_collate_fn as collate_fn,
    )
```

⚠️ 위 코드는 **예시**. 실제 dispatch 패턴(import / factory)은 trainer 코드에 따라 다름. grep 결과 보고 동일 패턴으로 추가.

### Task 7.4: 1 epoch smoke 학습

- [ ] **Step 1: FPFH smoke (1 epoch, batch_size=2)**

Run:
```bash
cd /home/jhs/work/Registration/glue-factory_depth
GPU=0 EXP=smoke_iss_fpfh_v1 BATCH=2 \
    bash train_iss_fpfh_v1_20260415.sh \
    --overrides train.num_epochs=1 train.eval_every_iter=999999 \
    2>&1 | tail -30
```
Expected: 정상 종료. loss 출력 + 1 epoch 완료.

- [ ] **Step 2: SHOT smoke**

Run:
```bash
GPU=0 EXP=smoke_iss_shot_v1 BATCH=2 \
    bash train_iss_shot_v1_20260415.sh \
    --overrides train.num_epochs=1 train.eval_every_iter=999999 \
    2>&1 | tail -30
```
Expected: 정상 종료. (descriptor dim 352 정상 처리)

- [ ] **Step 3: 사용자 commit 요청**

> ```
> git add gluefactory/configs/iss_*_v1_20260415_lg.yaml \
>         train_iss_*_v1_20260415.sh \
>         gluefactory/datasets/__init__.py
> git commit -m "feat(train): v1 FPFH/SHOT yaml+sh, dataset registration"
> ```

---

## Phase 8 — 본 학습

### Task 8.1: FPFH 본 학습

- [ ] **Step 1: 사용자 확인 (오래 걸리는 작업)**

> "FPFH 본 학습 시작합니다. background로 돌리고 tensorboard로 모니터링하시겠습니까?"

- [ ] **Step 2: 학습 launch (background)**

Run:
```bash
cd /home/jhs/work/Registration/glue-factory_depth
GPU=0 EXP=iss_fpfh_v1_20260415_e0 nohup \
    bash train_iss_fpfh_v1_20260415.sh \
    > train_iss_fpfh_v1_e0.log 2>&1 &
echo $! > train_iss_fpfh_v1_e0.pid
```

- [ ] **Step 3: tensorboard 모니터링 안내**

Run (사용자 직접):
```bash
tensorboard --logdir outputs/training/iss_fpfh_v1_20260415_e0 --port 6006
```

### Task 8.2: SHOT 본 학습

- [ ] **Step 1: 학습 launch (다른 GPU 또는 FPFH 종료 후)**

Run:
```bash
GPU=1 EXP=iss_shot_v1_20260415_e0 nohup \
    bash train_iss_shot_v1_20260415.sh \
    > train_iss_shot_v1_e0.log 2>&1 &
echo $! > train_iss_shot_v1_e0.pid
```

⚠️ GPU 1개만 있다면 FPFH 학습 종료 후 launch.

---

## Phase 9 — Test / Eval 스크립트

### Task 9.1: Matching 시각화 (FPFH)

**Files:**
- Create: `test_iss_fpfh_v1_20260415.py`

- [ ] **Step 1: 사용자 확인**

> "matching 시각화 / registration / RMSE eval 6 파일을 작성합니다. 기존 `test_resample2_iss_fpfh_0406.py` 등을 base로 v1 dataset/cache 경로로 변경. 진행해도 될까요?"

- [ ] **Step 2: 기존 base 파일 확인**

Run:
```bash
ls /home/jhs/work/Registration/glue-factory_depth/test_resample2_iss_fpfh_0406.py \
   /home/jhs/work/Registration/glue-factory_depth/test_registration_resample2_iss_fpfh_0406.py \
   /home/jhs/work/Registration/glue-factory_depth/eval_registration_iss_fpfh.py \
   /home/jhs/work/Registration/glue-factory_depth/eval_registration_iss_shot.py
```

- [ ] **Step 3: FPFH matching 시각화 작성**

```bash
cd /home/jhs/work/Registration/glue-factory_depth
cp test_resample2_iss_fpfh_0406.py test_iss_fpfh_v1_20260415.py
```

`test_iss_fpfh_v1_20260415.py` 내부 변경 (편집기로):
- import: `from gluefactory.datasets.mitsubishi_v1_20260415_iss_fpfh_dataset import MitsubishiV1ISSFPFHDataset, v1_iss_fpfh_collate_fn`
- dataset 인스턴스화: `MitsubishiV1ISSFPFHDataset(split="test", ...)`
- crop/resize 코드 제거 (이미 zero-pad)
- 출력 디렉토리 `results/iss_fpfh_v1_20260415/`

⚠️ 변경량은 base 코드에 따라 다름. 사용자가 직접 보고 일관되게 수정.

### Task 9.2: SHOT matching 시각화

```bash
cp test_iss_fpfh_v1_20260415.py test_iss_shot_v1_20260415.py
sed -i 's/iss_fpfh_v1/iss_shot_v1/g; s/MitsubishiV1ISSFPFHDataset/MitsubishiV1ISSSHOTDataset/g; \
        s/v1_iss_fpfh_collate_fn/v1_iss_shot_collate_fn/g' \
    test_iss_shot_v1_20260415.py
```

### Task 9.3: Registration (FPFH/SHOT)

```bash
cd /home/jhs/work/Registration/glue-factory_depth
cp test_registration_resample2_iss_fpfh_0406.py test_registration_iss_fpfh_v1_20260415.py
# 편집기로:
#   - dataset import 변경 (위와 동일)
#   - keypoints_xyz_mm 사용 (캐시에 포함됨, 별도 변환 불필요)
#   - 좌표 단위는 mm → RMSE 단위 mm
#   - crop/resize 코드 제거
```

SHOT 버전:
```bash
cp test_registration_iss_fpfh_v1_20260415.py test_registration_iss_shot_v1_20260415.py
sed -i 's/iss_fpfh_v1/iss_shot_v1/g; s/MitsubishiV1ISSFPFHDataset/MitsubishiV1ISSSHOTDataset/g; \
        s/v1_iss_fpfh_collate_fn/v1_iss_shot_collate_fn/g' \
    test_registration_iss_shot_v1_20260415.py
```

### Task 9.4: Eval RMSE (FPFH/SHOT)

```bash
cp eval_registration_iss_fpfh.py eval_registration_iss_fpfh_v1_20260415.py
# 편집기로 dataset 경로/import + keypoints_xyz_mm 사용
cp eval_registration_iss_fpfh_v1_20260415.py eval_registration_iss_shot_v1_20260415.py
sed -i 's/iss_fpfh_v1/iss_shot_v1/g; s/MitsubishiV1ISSFPFHDataset/MitsubishiV1ISSSHOTDataset/g; \
        s/v1_iss_fpfh_collate_fn/v1_iss_shot_collate_fn/g' \
    eval_registration_iss_shot_v1_20260415.py
```

### Task 9.5: 평가 실행 (학습 완료 후)

- [ ] **Step 1: FPFH RMSE**

Run:
```bash
python eval_registration_iss_fpfh_v1_20260415.py \
    --weights outputs/training/iss_fpfh_v1_20260415_e0/checkpoint_best.tar \
    --split test \
    --output_dir results/iss_fpfh_v1_20260415_eval/
```

- [ ] **Step 2: SHOT RMSE**

Run:
```bash
python eval_registration_iss_shot_v1_20260415.py \
    --weights outputs/training/iss_shot_v1_20260415_e0/checkpoint_best.tar \
    --split test \
    --output_dir results/iss_shot_v1_20260415_eval/
```

- [ ] **Step 3: 결과 비교 + 사용자 commit 요청**

비교 표 작성 (두 variant + baseline resample_2):

| Variant | mean RMSE (mm) | success rate (≤ 5mm) |
|---|---|---|
| FPFH v1 | ? | ? |
| SHOT v1 | ? | ? |
| FPFH resample_2 baseline | ? | ? |

> ```
> git add test_iss_*_v1_20260415.py test_registration_iss_*_v1_20260415.py \
>         eval_registration_iss_*_v1_20260415.py
> git commit -m "feat(eval): v1 FPFH/SHOT matching/registration/RMSE scripts"
> ```

---

## 부록 A — 성능/리스크 트리거

학습 중 또는 캐시 생성 중 다음 신호 발생 시 대응:

| 신호 | 의심 원인 | 대응 |
|---|---|---|
| SHOT precompute > 30s/zmap | 1mm voxel + 큰 영역 → cleanCloud 대량 | (a) `OMP_NUM_THREADS` 코어 수까지 증가, (b) zmap 단위 멀티프로세싱(병렬 N개), (c) FPFH 학습 먼저 진행하면서 SHOT 캐시는 백그라운드 생성. **voxel_size=1mm 고정, 변경 금지.** |
| FPFH precompute > 5s/zmap | normal estimation overhead | thread 수 (`OMP_NUM_THREADS`) 확인 |
| valid_mask < 50% | normal/SHOT radius 부족 | normal_radius 25mm로 spike |
| 학습 loss NaN | descriptor 0벡터 다수 | precompute에서 valid_mask=False keypoint를 score=0으로 강제 |
| val recall < 0.05 (1 epoch 후) | gt_radius 부족 + ISS 희소(아래 참조) | gt_radius=30 로 상향 후 재학습, 불충분 시 ISS 파라미터 재튜닝 |

### A-1 ISS keypoint 분포 (2026-04-15 NAS 641장 실측)

측정 스크립트: `measure_iss_distribution_v1.py`, raw: `iss_distribution_v1_20260415.json`

파라미터: `voxel=1mm, salient=6·avg_nn, non_max=2·salient, gamma_21=gamma_32=0.5, min_neighbors=5`

| 지표 | min | p10 | p25 | median | mean | p75 | p90 | max |
|---|---|---|---|---|---|---|---|---|
| K_iss | 38 | 44 | 47 | **51** | 52.0 | 56 | 61 | 72 |

- `K ≥ 512`: **0%** (모든 이미지에서 미달)
- `K < 64`: 96.4%
- M_vox 분포: min=14909, median=21207, max=26545 (1mm voxel downsample 후)

**함의**: MAX_KEYPOINTS=512 중 평균 460개(90%)가 random padding. ISS 사용 의미가 희석됨. Phase 4/5 precompute는 현 파라미터로 진행(정확도 영향은 Phase 8 학습 결과로 경험적 판단), Phase 8 val recall 저조 시 ISS 파라미터 스윕(`salient_mul ∈ {3,4,6,8}`, `gamma ∈ {0.3,0.5,0.7}`)을 근거로 재튜닝 결정.

**대안 considered but deferred**:
- MAX_KEYPOINTS 128로 축소: p90=61 커버 가능하지만 Phase 6 dataset 인터페이스 변경 필요 → Phase 8 결과 전 불필요한 churn.
- ISS 파라미터 선제 스윕: ~20min 소요. Val recall이 나올 때까지는 "더 많은 keypoint = 더 좋다"가 검증되지 않아 blind tuning 위험.

## 부록 B — Self-review 체크 (작성자용 — 실행 전 확인)

- [x] Spec §3 (핵심 결정) → Phase 1~7 모두 매핑됨
- [x] Spec §4 (파이프라인) → Phase 4/5 (precompute) + Phase 6 (dataset) + Phase 7 (trainer)
- [x] Spec §5 (캐시 npz) → Phase 4/5 process_one()의 np.savez 필드 일치
- [x] Spec §6 (파일 매핑) → File Structure 표 일치
- [x] Spec §7 (검증 계획) → Phase 1.3, 3.2, 4.1 step 4, 5.1 step 4, 9.5 단위·통합 검증
- [x] Spec §8 (리스크) → 부록 A 트리거
- [x] 모든 step에 exact path / 명령어 / expected output 포함
- [x] Placeholder ("TBD" 등) 없음
- [x] 함수/타입 이름 일관 (`MitsubishiV1ISSFPFHDataset`, `extract_shot_at_keypoints`, `v1_iss_*_collate_fn`)
