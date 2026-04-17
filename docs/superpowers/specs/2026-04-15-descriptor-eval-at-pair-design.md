# Descriptor 평가 at User-defined Pair — 설계

**Date:** 2026-04-15
**Topic:** Scanned vs Blender descriptor discriminability 진단 (SHOT + FPFH)
**Status:** Approved, pending implementation

## 배경 / 동기

메모리 `project_shot_scan_domain_gap` 에 기록된 바와 같이, scanned ↔ blender
간 SHOT descriptor 매칭은 현재 scan-vs-train 에서 19-22/5 matches,
scan-vs-scan 에서 12/4 matches 수준으로 목표치 (30~100) 에 못 미친다.
"descriptor 자체가 제 역할을 하는가" 를 정량·정성으로 확인하기 위해,
동일 물리 지점에서 두 도메인의 descriptor 벡터를 직접 비교한다.

## 입력

| 항목 | 값 |
|---|---|
| Scanned depth | `gluefactory/datasets/scanned/roi13_zmap 1.png` (uint16) |
| Blender depth | `gluefactory/datasets/scanned/blender_master1/master.png` (uint16, rot180 적용) |
| User pair | `PAIRS = [((us,vs), (ub,vb))]` — 스크립트 상단 hardcoded. N=1 으로 시작. |

## 물리 spacing

| 축 | mm / raw unit |
|---|---|
| lateral (x, y) | 0.056 |
| vertical (z)   | 0.0085 |

## 파이프라인

1. **Load & preprocess**
   - `scanned_raw = cv2.imread(...)` → `mask_scanned_table()` (floor histogram
     peak 제거) → `apply_bilateral(d=5, σ_c=100, σ_s=3)` (scanned 전용).
   - `blender_raw = cv2.imread(...)` → `cv2.rotate(ROTATE_180)`.

2. **Build mm PCD** (`build_iss_pcd_mm`, ERODE_BOUNDARY=5)
   - `pcd_s` (scanned 전처리본), `pcd_b` (blender rot180).

3. **ISS keypoints** (`detect_iss_mm`)
   - `salient_r = 6 · avg_nn_mm`, `non_max_r = 2 · salient_r`,
     `γ21 = γ32 = 0.5`, `min_nbrs = 5`.
   - 출력: `kp_s_mm (Ns,3)`, `kp_b_mm (Nb,3)`.

4. **User pair → ISS snap**
   - user `(us, vs)` → `q_s_mm = (us·LAT, vs·LAT, zmap_s[vs,us]·VERT)`
     단, `zmap_s[vs,us] == 0` 이면 조기 `ValueError` (mask 영역 픽셀 가능성).
   - `snap_s = nearest in kp_s_mm` via `scipy.spatial.cKDTree`.
   - blender 도 동일 (`rot180 frame` 에서 lookup).
   - `snap_dist > 5mm` 이면 stderr warn + 진행.

5. **SHOT descriptor (pybind shot_module)**
   - `pts_s_shot, desc_s_shot = shot_module.extract_shot(pts_s_mm,
     voxel_size=1.0, normal_radius=10.0, shot_radius=20.0)` → (M_s, 3), (M_s, 352).
   - `d_s_shot = desc_s_shot[ cKDTree(pts_s_shot).query(snap_s, k=1)[1] ]` → (352,).
   - blender 동일.

6. **FPFH descriptor (Open3D)**
   - `pcd_s_down = pcd_s.voxel_down_sample(1.0)`.
   - `pcd_s_down.estimate_normals(KDTreeSearchParamRadius(radius=10.0))`.
   - `fpfh_s = o3d.pipelines.registration.compute_fpfh_feature(
         pcd_s_down, KDTreeSearchParamRadius(radius=20.0))` → shape (33, M_s').
   - `d_s_fpfh = fpfh_s.data[:, nearest_idx(pcd_s_down.points, snap_s)]` → (33,).
   - blender 동일.

7. **Similarity**
   - `cos_shot = np.dot(d_s_shot, d_b_shot) / (||·|| · ||·||)`.
   - `cos_fpfh = np.dot(d_s_fpfh, d_b_fpfh) / (||·|| · ||·||)`.

8. **Visualize** (아래 출력 섹션 참조).

## 파라미터 (확정값)

| 카테고리 | 키 | 값 |
|---|---|---|
| bilateral (scanned) | diameter | 5 px |
|  | σ_color | 100.0 raw ≈ 0.85 mm |
|  | σ_space | 3.0 px |
| ISS | γ21, γ32 | 0.5, 0.5 |
|  | min_neighbors | 5 |
|  | erode_boundary | 5 px |
| SHOT | voxel | 1.0 mm |
|  | normal_radius | 10.0 mm |
|  | shot_radius | **20.0 mm** |
| FPFH | voxel | 1.0 mm |
|  | normal_radius | 10.0 mm |
|  | fpfh_radius | **20.0 mm** |
| snap | warn threshold | 5.0 mm |

**Ablation 후보 (이 스펙 범위 외, 후속 작업)**: `shot_radius = fpfh_radius = 10mm`.

**SHOT/FPFH 파라미터 근거**: 프로젝트 학습 모델 (`0413_new_iss_shot_v5_lg`) 이
`normal_r=25, shot_r=50` (2:1 ratio) 를 사용. 이 비율을 유지하면서 voxel=1mm
scale 에 맞춰 `normal_r=10, shot_r=20` 으로 설정. FPFH 는 SHOT shot_r 과 의미
대응되도록 `fpfh_r=20`. `normal_r = feature_r` (1:1) 는 PCL/Open3D 관례상 normal
oversmoothing 을 유발해 discriminability 저하 가능.

## Error handling / sanity

| 상황 | 처리 |
|---|---|
| 이미지 파일 없음 | `assert path.exists()` — FileNotFoundError |
| user pixel depth == 0 (scanned/blender 둘 중 하나) | 조기 `ValueError("picked pixel has depth=0")` |
| snap_dist > 5 mm | stderr warn, 진행 |
| SHOT 결과 0점 | `RuntimeError` |
| FPFH all-zero descriptor | stderr warn ("fpfh_radius 확장 고려"), 진행 |
| descriptor 차원 불일치 (≠352, ≠33) | `assert` |

실행 로그에 다음 print: `n_pcd_{s,b}`, `n_iss_{s,b}`, `n_shot_{s,b}`,
`snap_dist_{s,b}`, descriptor stats (L2, zero_count, max_bin, cos_sim).

Unit test 없음 (일회성 평가 스크립트). Sanity 는 위 print 로만.

## 출력

경로: `vis_new/Descriptor__분석/`

| 파일 | 내용 |
|---|---|
| `eval_descriptor_at_pair_shot.png` | Row0: scanned·blender ISS overlay + picked ★ / Row1: SHOT 352-bin bar overlay (scanned red, blender blue, α=0.6). Title 에 cos_shot, snap_dist 포함. |
| `eval_descriptor_at_pair_fpfh.png` | Row0: 동일 ISS context / Row1: FPFH 33-bin bar overlay. Title 에 cos_fpfh, snap_dist 포함. |

- `figsize=(14, 9)` 각 figure. Row0 height=1.5, Row1 height=1.
- Row0 의 picked ★: yellow, s=150. ISS kp: cyan, s=4, α=0.5.
- Title 포맷: `"{SHOT|FPFH} @ pair (us,vs)↔(ub,vb) | cos=0.xx | snap: s=X.XXmm b=X.XXmm"`.

## 코드 구조

단일 파일 `vis_new/Descriptor__분석/eval_descriptor_at_pair.py`.

| 섹션 | 함수 / 행동 |
|---|---|
| Imports / consts | 위 파라미터 테이블의 상수들 |
| `apply_bilateral_depth(zmap)` | step1 에서 복붙 |
| `build_iss_pcd_mm(zmap, erode=5)` | step1 에서 복붙 |
| `detect_iss_mm(pcd)` | step1 에서 복붙 |
| `snap_to_iss(q_mm, kp_mm)` | cKDTree 로 NN 1개, `(snapped_xyz, dist_mm)` 리턴 |
| `compute_shot_desc_at(pcd_pts_mm, query_xyz_mm)` | `shot_module.extract_shot(...)` → cKDTree lookup → `(352,)` |
| `compute_fpfh_desc_at(pcd, query_xyz_mm)` | voxel_down_sample → estimate_normals → compute_fpfh → NN lookup → `(33,)` |
| `render_shot_figure(...)` → PNG | Row0 + SHOT bar |
| `render_fpfh_figure(...)` → PNG | Row0 + FPFH bar |
| `main()` | 모든 단계 orchestrate + 로그 print |

**step1 함수 복붙 근거**: step1_bilateral_fps.py 는 `__main__` 을 가진 스크립트 모듈로, import 시 `matplotlib.use("Agg")` 같은 전역 상태가 따라와 섞임. 1-pair 일회성 평가라 공통 util 로 리팩토링 비용보다 복붙이 저비용.

## 의존성

- `conda` env: `LightGlue`.
- Python: `numpy`, `scipy.spatial.cKDTree`, `opencv-python`, `open3d`, `matplotlib`.
- Repo-local: `vis_new.inference_scan_vs_train.mask_scanned_table`,
  `pybind_shot_linux/shot_module*.so` (이미 빌드됨).

## Out of scope

- pair 개수 > 1 에서의 similarity matrix (Form B) — 추후 확장.
- shot_radius=10 ablation — 후속.
- FPFH radius 독립 sweep — 후속.
- Descriptor 학습 모델 로딩 (`.tar`) 을 통한 trained SHOT 비교 — 별개 진단 경로.
- 매칭/registration 자체는 이 스펙 범위 외.

## 사용자 input 포맷

좌표 입력 예시:
```
scanned: x=1234, y=567
blender (rot180 기준): x=890, y=123
```
→ `PAIRS = [((1234, 567), (890, 123))]` 로 스크립트에 반영.
