# v1_20260415 Scanned×2 Inference Timing Demo — Design

**Date**: 2026-04-17
**Author**: brainstormed with Claude
**Status**: draft (awaiting user review)

---

## Goal

실제 3D line scanner 배포 시나리오 데모용 **end-to-end 추론 시간 측정 스크립트**.
입력: scanned depth zmap 2장. 출력: stage 별 소요 시간 테이블 + R, t.

## Scope

- **측정 대상 pipeline**: `2 scanned zmap → preprocess → descriptor (ISS+SHOT352 또는 ISS+FPFH33) → LightGlue matching → RANSAC SVD → R, t`
- **측정 방식**: cold-start 1회 실행 (warmup/반복 없음)
- **출력**: stdout 요약 테이블 only (PNG 시각화는 추후 필요시 추가)
- **descriptor**: SHOT352 와 FPFH33 양쪽 모두, 별도 스크립트로
- **비측정 범위**: Python import/CUDA context init — timer 시작 전에 완료된 것으로 간주 (정의상 "입력 도착 이후")

## Non-Goals

- Warm steady-state benchmark (필요시 후속 작업)
- Descriptor 간 정량 비교 실험 (timing 은 그에 필요한 raw 데이터 제공, 해석은 별도)
- Real-time streaming / pipelining

## Architecture

### 파일 구성

```
experiments/v1_20260415/profile/
├── _timer.py                 (신규, 공용 Timer + 출력 helper)
├── timing_demo_shot.py       (신규, SHOT352 pipeline)
└── timing_demo_fpfh.py       (신규, FPFH33 pipeline)
```

### 의존 관계

```
timing_demo_shot.py
  └── from ..infer_scanned_vs_master_shot352 import:
        mask_scanned_table, apply_bilateral, prepare_iss_shot,
        load_model, make_view, collate_single_pair, run_inference,
        _ransac_rigid

timing_demo_fpfh.py
  ├── from ..precompute import helpers as h
  │     (voxel_downsample, extract_iss_on_dense,
  │      subsample_or_pad_keypoints, kp_xyz_mm_to_uv_padded, ...)
  ├── from ..infer_scanned_vs_master_shot352 import:
  │     mask_scanned_table, apply_bilateral, load_model,
  │     make_view, collate_single_pair, run_inference, _ransac_rigid
  └── open3d (inline: estimate_normals, compute_fpfh_feature)

_timer.py
  └── 표준 라이브러리 only (time, contextlib)
```

### 기존 파일 수정

**없음**. `infer_scanned_vs_master_shot352.py` 와 `precompute/helpers.py` 는 import 전용으로 사용.

## Pipeline Stages

### 9-stage 세분 (`--detailed` 출력)

| # | Stage | SHOT 구현 | FPFH 구현 |
|---|---|---|---|
| 1 | `load_scan0` | `np.array(Image.open(path))` | 동일 |
| 2 | `load_scan1` | 동일 | 동일 |
| 3 | `preprocess_scan0` | `mask_scanned_table → apply_bilateral` | 동일 |
| 4 | `preprocess_scan1` | 동일 | 동일 |
| 5 | `iss_desc_scan0` | `prepare_iss_shot(zmap, iss_mode="mm")` | ISS+FPFH 조립 (아래) |
| 6 | `iss_desc_scan1` | 동일 | 동일 |
| 7 | `lg_forward` | `make_view ×2 + collate + run_inference` | 동일 |
| 8 | `extract_matches` | `pred["matches0"]` 필터링 → kp pair | 동일 |
| 9 | `registration` | pixel → 3D mm + `_ransac_rigid` | 동일 |

### 4-stage 요약 (기본 출력)

| 묶음 | 포함 stage |
|---|---|
| Preprocess | 1 + 2 + 3 + 4 |
| Descriptor | 5 + 6 |
| Matching | 7 + 8 |
| Registration | 9 |

### FPFH stage 5 내부 조립 (SHOT 에 없는 로직)

`timing_demo_fpfh.py` 안에 로컬 함수 + inline Open3D 호출:

```python
def zmap_array_to_pts_mm(zmap, lateral=0.056, transport=0.056, vertical=0.0085):
    vs, us = np.where(zmap > 0)
    d = zmap[vs, us].astype(np.float64)
    return np.stack([us * lateral, vs * transport, d * vertical], axis=1)

# stage 5 내부
pts = zmap_array_to_pts_mm(zmap_pre)
pcd_vox = h.voxel_downsample(pts)              # h.VOXEL_SIZE = 1.0 mm
kp_xyz_iss = h.extract_iss_on_dense(pts)
kp_xyz, kp_idx, kp_score, _, _ = h.subsample_or_pad_keypoints(
    kp_xyz_iss, pcd_vox, max_n=h.MAX_KEYPOINTS, seed=0)
pcd_vox.estimate_normals(                      # h.NORMAL_RADIUS = 20.0 mm
    search_param=o3d.geometry.KDTreeSearchParamRadius(h.NORMAL_RADIUS))
fpfh = o3d.pipelines.registration.compute_fpfh_feature(
    pcd_vox, o3d.geometry.KDTreeSearchParamRadius(h.FPFH_RADIUS))  # 20.0 mm
desc = np.asarray(fpfh.data, dtype=np.float32)[:, kp_idx].T
# v1_20260415_norm ckpt 전제 → L2 normalize (train cache 가 normalized 되어 있음)
desc = desc / (np.linalg.norm(desc, axis=1, keepdims=True) + 1e-12)
kp_uv = h.kp_xyz_mm_to_uv_padded(kp_xyz)
```

## Timer Abstraction

`_timer.py` 공개 API:

```python
class StageRecord:
    def __init__(self) -> None: ...

    @contextmanager
    def timer(self, name: str): ...
        # time.perf_counter() 기반

    def print_table(self, groups: dict[str, list[str]],
                    detailed: bool = False) -> None: ...
        # detailed=False: 4-stage 요약
        # detailed=True:  9-stage 개별 + 4-stage 요약
```

**동기화 정책**:
- pybind/Python/Open3D stage: sync 불필요 (동기 함수).
- `lg_forward` stage: timer block 진입 직전 `torch.cuda.synchronize()`, 종료 직후 한 번 더 `torch.cuda.synchronize()`.
- `extract_matches` stage: `.cpu().numpy()` 에서 자연 sync 가 일어나지만 명시적으로 앞에 `torch.cuda.synchronize()` 넣어 안전하게.

## CLI

공통 flags (두 스크립트 동일):

| flag | default | 의미 |
|---|---|---|
| `--scan0` | `gluefactory/datasets/scanned/scanned_data1.png` | 첫 zmap |
| `--scan1` | `gluefactory/datasets/scanned/roi13_zmap 1.png` | 둘째 zmap |
| `--checkpoint` | SHOT: `outputs/training/iss_shot_v1_20260415_dim352_0417/checkpoint_best.tar`<br/>FPFH: `outputs/training/iss_fpfh_v1_20260415_norm_0417/checkpoint_best.tar` | LG ckpt 경로 (padding bug fix 후 재학습된 `_0417` 변종) |
| `--detailed` | `False` | 9-stage 세부 출력 여부 |
| `--device` | `cuda` | torch device |

**실행 예시**:
```bash
conda activate LightGlue
cd /home/jhs/work/Registration/glue-factory_depth

python experiments/v1_20260415/profile/timing_demo_shot.py
python experiments/v1_20260415/profile/timing_demo_shot.py --detailed
python experiments/v1_20260415/profile/timing_demo_fpfh.py
```

## Output Format

**기본 (4-stage)**:
```
=== SHOT352 Timing (scanned×2) ===
input: scanned_data1.png + roi13_zmap 1.png
device: cuda
ckpt: iss_shot_v1_20260415_dim352/checkpoint_best.tar

[Preprocess  ]  0.342s
[Descriptor  ]  2.871s
[Matching    ]  0.058s
[Registration]  0.104s
─────────────────────────
[Total       ]  3.375s

R = [[...]]
t = [..., ..., ...]
n_matches = 142, n_inliers = 98
```

**`--detailed`**: 위 4-stage 요약 + 그 아래 9-stage 개별 라인.

## Edge Cases

| 상황 | 처리 |
|---|---|
| zmap 파일 없음 | `sys.exit(f"[err] scan path not found: {path}")` |
| ISS keypoint 0 개 | stage 5 직후 `sys.exit("[err] no ISS kp")` |
| valid match < 3 | `print("[warn] <3 matches, skip registration")` + stage 9 skip, Total 정상 출력 |
| ckpt 파일 없음 | `torch.load` 기본 에러 그대로 |
| CUDA 불가 | `torch.cuda.is_available()` 가 False 이면 자동으로 `cpu` 로 fallback 하고 stderr 에 warning 출력 (`--device cuda` 기본값이어도 동작). 사용자는 명시적 `--device cpu` 도 가능. |

## Testing

단위 테스트 생략. 실행 자체가 smoke test. (core helper 승격 시 `tests/test_profile_timer.py` 로 Timer 단위 테스트 추가 권장 — 이번 범위 밖.)

## Future Work (이 spec 범위 밖)

- warmup + N회 median 모드 (`--mode warm --runs 10`) 추가
- PNG overlay 시각화 (registration 결과 확인용)
- core helper 모듈 승격: `experiments/v1_20260415/pipelines/scanned_{shot352,fpfh}.py` 로 기존 infer 스크립트와 demo 가 동일 pipeline 공유

## Notes

- v1 FPFH ckpt 두 변종 존재: `iss_fpfh_v1_20260415` (cache non-norm, 추론도 non-norm) vs `iss_fpfh_v1_20260415_norm` (cache L2-normalized, 추론도 normalize). 이 demo 는 **`_norm` 변종 기준** 으로 설계 — descriptor slicing 뒤 L2 normalize 한 줄 포함. non-norm ckpt 로 돌리고 싶으면 해당 라인 skip + `--checkpoint` 경로 교체 필요.
- SHOT pybind module (`shot_module`) 은 `infer_scanned_vs_master_shot352.py` import 시점에 eager 로드됨 → demo timer 시작 이전에 준비됨.
- `_rigid_svd`, `_ransac_rigid` 는 underscore prefix (Python convention 상 private) 지만 demo 수명 짧다는 점을 고려해 직접 import 하여 재사용. core helper 승격 시 이름 정리 (Future Work 참조).
