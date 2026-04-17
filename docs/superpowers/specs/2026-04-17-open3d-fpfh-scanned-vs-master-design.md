# Open3D FPFH Scanned vs Master Baseline — Design (2026-04-17)

## 배경

`experiments/v1_20260415/`에는 ISS+FPFH/SHOT+LightGlue 학습 파이프라인과 scanned(roi13) vs master(blender_master1) 추론 스크립트(`infer_scanned_vs_master_shot352.py`)가 있다. 학습 모델 대비 baseline 비교용으로 Open3D 순수 RANSAC FPFH global registration을 돌린 결과(내장 metric + 시각화 + 총 소요시간)가 필요하다.

`open3d_fpfh_func.py`는 Open3D 튜토리얼 코드를 그대로 복사해 둔 참고용 레퍼런스이며, 실제 데이터(roi13 zmap, blender_master1/master.png)에 대해서는 아직 실행한 적이 없다.

## 목표

Scanned(roi13) ↔ Master(blender_master1) 쌍에 대해:

1. Open3D `registration_ransac_based_on_feature_matching`이 반환하는 `fitness`와 `inlier_rmse` 기록
2. matplotlib 2D 3-view PNG로 before/after 비교 및 overlay 저장 (기존 SHOT 추론 스크립트와 동일 스타일)
3. 파이프라인 total wall-clock time 측정

**Non-goals**:

- GT 기반 RMSE 계산 없음 (scanned는 GT transform 없음)
- ICP refinement 없음 (RANSAC global 단계만)
- 단계별 세부 타이밍 분해 없음 (total 1개 숫자만)
- 학습 모델(ISS+FPFH/SHOT+LG) 실행 없음 — 순수 Open3D baseline

## 범위

- 입력 파일은 기존 SHOT 추론 스크립트와 동일 (`roi13_zmap 1.png`, `blender_master1/master.png`)
- master 180도 회전 기본 적용 (`--no_rotate`로 해제 가능)
- scanned 전처리(floor mask + bilateral)는 기존과 동일 (fair comparison)
- output: `experiments/v1_20260415/results/scanned_vs_open3d_fpfh_{param_mode}/`

## 아키텍처

새 스크립트 1개: `experiments/v1_20260415/registration/open3d_fpfh_scanned_vs_master.py`

기존 `run_fpfh.py`(학습된 LG 모델 기반)와 분리. Open3D 내장 RANSAC feature matching만 사용.

### 컴포넌트

#### 1. 입력 로드 & 전처리

- `mask_scanned_table(zmap)` / `apply_bilateral(zmap)`: `infer_scanned_vs_master_shot352.py`에서 import 재사용
- `zmap_to_pcd_mm(zmap)`: `infer_scanned_vs_master_shot352.py`의 구현 재사용 (mm 좌표 변환 + boundary erode=5)
- 출력: `src_pts_mm` (scan), `dst_pts_mm` (master) — 둘 다 `np.ndarray (N, 3) float32`

#### 2. Open3D 전처리 & FPFH

```python
def preprocess_point_cloud(pts_mm, voxel, normal_r, fpfh_r):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts_mm.astype(np.float64))
    pcd_down = pcd.voxel_down_sample(voxel)
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=normal_r, max_nn=30))
    fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=fpfh_r, max_nn=100))
    return pcd_down, fpfh
```

파라미터 모드:

- `--param_mode v1`: voxel=1.0, normal_r=20.0, fpfh_r=20.0 (학습과 동일)
- `--param_mode tutorial`: voxel=5.0, normal_r=10.0, fpfh_r=25.0 (voxel·2, voxel·5 관례)

개별 override: `--voxel`, `--normal_radius`, `--fpfh_radius`, `--distance_threshold`

#### 3. RANSAC global registration

`open3d_fpfh_func.py`의 `execute_global_registration`을 그대로 이식:

- `registration_ransac_based_on_feature_matching`
- `TransformationEstimationPointToPoint(False)`
- n_ransac=3, checkers=[`EdgeLength(0.9)`, `Distance(distance_threshold)`]
- `RANSACConvergenceCriteria(100000, 0.999)`
- distance_threshold 기본값 = voxel · 1.5, CLI로 override 가능

반환: `result.transformation` (4×4), `result.fitness`, `result.inlier_rmse`, `result.correspondence_set`

#### 4. 시각화

기존 `infer_scanned_vs_master_shot352.py`의 `_plot_registration`, `_plot_overlay` 함수 import 재사용.

- 샘플링: `_sample_pcd_mm(zmap, 15000, seed=42)` 재사용
- `pc_src`, `pc_dst` sample → `pc_est = (R @ pc_src.T).T + t` (result.transformation에서 R, t 추출)
- Z 반전: `zf = np.array([1, 1, -1])` (기존과 동일, 시각화용)
- 저장: `reg_open3d_fpfh.png` (2×3 grid: before/after × top/front/side), `overlay_open3d_fpfh.png` (1×3 overlay)
- 제목에 `fitness`, `inlier_rmse`, `elapsed_s`, param_mode 표시

#### 5. 결과 저장

`experiments/v1_20260415/results/scanned_vs_open3d_fpfh_{param_mode}/`:

- `reg_open3d_fpfh.png`
- `overlay_open3d_fpfh.png`
- `result.json`: `{param_mode, voxel, normal_radius, fpfh_radius, distance_threshold, fitness, inlier_rmse, n_correspondences, n_src_down, n_dst_down, elapsed_s, master_path, scan_path, rotate_master}`
- stdout에도 동일 정보 출력

## 데이터 흐름

```
scan_raw                                 master_raw
  │                                        │
  mask_scanned_table → apply_bilateral     (rotate 180° unless --no_rotate)
  │                                        │
  zmap_to_pcd_mm (erode=5)                 zmap_to_pcd_mm (erode=5)
  │                                        │
  ┌─── wall-clock t0 ─────────────────────────────────────────────┐
  │                                                               │
  preprocess_point_cloud(pts, voxel, nr, fr)                      │
  ├─ src_down, src_fpfh       dst_down, dst_fpfh                  │
  │                                                               │
  execute_global_registration(src_down, dst_down,                 │
                              src_fpfh, dst_fpfh,                 │
                              distance_threshold)                 │
  │                                                               │
  └─── wall-clock t1 ─────────────────────────────────────────────┘
  │
  transformation, fitness, inlier_rmse, correspondence_set
  │
  sample (15k pts) → _plot_registration, _plot_overlay → PNG
  │
  result.json
```

**Timer 범위**: preprocessing(zmap→PCD 변환)은 기존 전처리와 공유되니 타이머 밖. Open3D 고유 연산(voxel+normal+FPFH+RANSAC) 전체를 t0→t1로 측정. 이 범위는 stdout 로그에 명시.

## CLI

```bash
python experiments/v1_20260415/registration/open3d_fpfh_scanned_vs_master.py \
  [--param_mode v1|tutorial]           # 기본: v1
  [--voxel FLOAT]                      # param_mode override
  [--normal_radius FLOAT]
  [--fpfh_radius FLOAT]
  [--distance_threshold FLOAT]         # 기본: voxel * 1.5
  [--master PATH]                      # 기본: blender_master1/master.png
  [--no_rotate]                        # master 180° 회전 해제
  [--output_dir PATH]                  # 기본: 자동 생성
  [--seed INT]                         # 기본: 42 (sampling 용)
```

## 에러 처리

- 입력 PNG 없음 → `FileNotFoundError` (명시적)
- voxel이 너무 커서 downsample 결과 3점 미만 → Open3D가 RANSAC 단계에서 예외. 그대로 전파.
- `result.correspondence_set`이 비어 있거나 fitness=0 → 그대로 기록하고 시각화는 identity transform 대입 (degenerate 케이스도 눈으로 확인 가능하게)

## 테스트 전략

이 스크립트는 단일 파일·단일 데이터쌍에 대한 실행 스크립트라 전용 pytest는 필요 없음. 검증은 수동:

1. `--param_mode v1` 실행 → 결과 PNG 눈으로 확인 + `result.json` fitness/elapsed 확인
2. `--param_mode tutorial` 실행 → 두 모드 간 metric·시각 차이 확인
3. 기존 `scanned_vs_shot_dim352_iss_mm/` 결과와 시각적 비교

## 의존성

- Open3D (기존 환경 `LightGlue` conda env에 설치됨)
- OpenCV, numpy, matplotlib (기존 환경)
- `experiments/v1_20260415/infer_scanned_vs_master_shot352.py` 에서 함수 import (sibling 파일 — `sys.path` 추가 필요)
