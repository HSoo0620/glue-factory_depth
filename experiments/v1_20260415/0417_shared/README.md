# 0417_shared — Depth Registration (scanned → master)

**목적:** v1_20260415 ISS + FPFH/SHOT + LightGlue 모델을 Python GUI 에서 `import` 하여
scanned depth 이미지를 master 좌표계로 정합하는 패키지.

## TL;DR

```python
from depth_registration import register_pair, preload_model

preload_model("fpfh", device="cuda")
result = register_pair("scan.png", descriptor="fpfh")
print(result["T"])  # (4, 4) float64, mm 좌표계
print(result["success"], result["num_inliers"], result["inlier_ratio"])
```

`result["T"] @ [scanned_xyz, 1] = [master_xyz, 1]` (mm 단위).

## 설치

### Linux 개발 (현재 1차 타깃)

```bash
conda activate LightGlue          # Python 3.10, torch, open3d, etc.
cd experiments/v1_20260415/0417_shared
pytest tests/ -v                  # 19 passed 기대 (FPFH + SHOT 모두)
```

SHOT 을 쓰려면 `pybind_shot_linux/shot_module.cpython-310-x86_64-linux-gnu.so`
(repo 동봉) 가 있어야 한다. 없으면 `test_shot*` 는 SKIP 된다.
SHOT ckpt (`iss_shot_v1_dim352_0417.tar`, 256 MB) 는 repo 에 포함되지 않으므로
`outputs/training/iss_shot_v1_20260415_dim352_0417/checkpoint_best.tar` 로
심볼릭 링크/복사 필요.

### Windows 배포 (추후 보강)

`environment/setup_windows.md` 에 초안이 있으나 PCL 1.15.1 빌드와
`shot_module.cp313-win_amd64.pyd` 수치 일치 검증은 아직 수행 전.
FPFH 단독 경로는 동일 Python API 로 동작할 것으로 예상되나 미검증.

## 공개 API

```python
register_pair(
    scanned: str | Path | np.ndarray,
    master:  str | Path | np.ndarray | None = None,   # None → default master
    descriptor: "fpfh" | "shot" = "fpfh",
    *,
    cache_dir:   Path | None = None,
    inlier_th:   float = 5.0,                          # mm
    ransac_iter: int   = 1000,
    success_min_inlier_ratio: float = 0.1,
    device: str = "cuda",
) -> RegistrationResult
```

반환 dict 필드:

| 필드 | 타입 | 설명 |
|---|---|---|
| `T`, `R`, `t` | ndarray | (4,4)/(3,3)/(3,) float64, mm |
| `inlier_ratio` | float | RANSAC inlier / total matches |
| `num_matches`, `num_inliers` | int | LightGlue 매치 / RANSAC 생존 |
| `matches_scanned_xyz`, `matches_master_xyz` | ndarray | inlier-only, (M,3) float32 mm |
| `success` | bool | `num_inliers ≥ 3` 및 `inlier_ratio ≥ threshold` |
| `elapsed_ms` | dict | preproc / descriptor_scanned / descriptor_master / lightglue / ransac / total |

## 예제

- `examples/run_cli.py` — argparse 스모크
- `examples/gui_integration_demo.py` — GUI 통합 최소 예시

## 주의

- **FPFH descriptor 는 L2 정규화를 적용한다** (v1 `_norm_0417` 학습 캐시와 분포 일치).
  `SearchParamRadius` 기반 neighborhood 로 계산되며, `SearchParamHybrid` 사용 금지.
- **SHOT descriptor 는 PCL 내부 unit-sphere 정규화를 그대로 사용** (norm≈1).
  추가 L2 정규화 금지 — double-normalize 가 되어 훈련 분포와 어긋난다.
- **ISS / FPFH / SHOT 파라미터는 학습 설정에 고정** (voxel=1 mm, normal_r=20 mm,
  fpfh_r=20 mm, shot_r=40 mm, max_kp=512). 외부 override 불허.
- **모델은 한 번에 하나만 상주.** 필요 시 `preload_model` / `unload_model` 로 수동 제어.
- **Master 캐시 fingerprint** 는 `(image bytes, descriptor, params)` 를 SHA1 해시.
  파라미터 변경 시 자동 재계산되며, 기본 master 는 `cache/` 에 사전 빌드되어 있음.

## 트러블슈팅

- `ImportError: DLL load failed` (Windows SHOT) → `environment/setup_windows.md` §3, §6 Phase 1.
- `FileNotFoundError: ... iss_shot_v1_dim352_0417.tar` → SHOT ckpt 별도 전달 후
  `checkpoints/` 에 복사.
- `CUDA out of memory` → `device="cpu"` 호출, 또는 `unload_model()` 로 다른 descriptor 언로드.
- `success=False` (`inlier_ratio` 낮음) → scanned 전처리/마스크 설정 점검. 이 패키지 범위 밖.
