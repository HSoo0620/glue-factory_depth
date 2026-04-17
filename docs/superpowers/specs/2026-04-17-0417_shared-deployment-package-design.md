# 0417_shared — Depth Registration 배포 패키지 설계

**작성일:** 2026-04-17
**대상 독자:** 본인(개발자), 팀원(Windows GUI 담당)
**범위:** v1_20260415 ISS + FPFH/SHOT + LightGlue 모델을 Windows Python GUI 에서 `import` 해서 쓰도록 배포용 패키지로 재구성.
**비범위(Out of scope):** ONNX/TensorRT 변환, 신규 모델 학습, GUI 자체 구현, 전처리/매칭 결과 포맷 개선(본 버전은 "기존 동작 보존"을 목표로 하고, 개선은 이후 릴리즈에서 다룬다).

---

## 1. 배경 / 목적

팀원이 Windows PC 에서 Python GUI 를 이미 구현해둔 상태이다. 본인의 연구 자산(ISS keypoint + FPFH/SHOT descriptor + LightGlue 매칭 + RANSAC/SVD 정합)을 GUI 에서 `from depth_registration import register_pair` 한 줄로 호출할 수 있도록 **단일 폴더 패키지**로 정돈해 전달하는 것이 본 설계의 목적이다.

Synthetic(Master) 데이터는 precompute 캐시가 있으면 재사용하고 없으면 즉석 계산하며, Scanned 데이터는 매 호출마다 기존 파이프라인(floor mask → bilateral → ISS → descriptor → LightGlue → RANSAC/SVD)을 그대로 돌린다.

---

## 2. 핵심 결정사항 요약

| 축 | 결정 | 근거 |
|---|---|---|
| GUI ↔ 코드 연동 | Python in-process `import` | GUI 가 Python. env 단일화, 모델 상주로 반복 호출 빠름 |
| 배포 단위 | 연구 레포 내 `experiments/v1_20260415/0417_shared/` 폴더 | 기존 `share/` 옆. 새 레포·submodule 없이 폴더 하나로 자급자족 |
| Descriptor | FPFH + SHOT 둘 다 (함수 인자 선택, SHOT 은 lazy import) | 기존 코드 재활용, 팀원이 PCL 미설치 상태로도 FPFH 먼저 시도 가능 |
| Master | 1 장 고정. 기본 master 용 pre-built FPFH 캐시 동봉 + 타 master 들어오면 자동 생성 | "있으면 쓰고 없으면 만든다" 요구 반영 |
| T 방향 | `T @ scanned_xyz = master_xyz` (scanned → master) | master 기준 위치를 scanned 에서 찾는 사용 사례 |
| API 입력 | `scanned: str \| Path \| np.ndarray` Union | GUI 측 유연성, type-check 분기 간단 |
| API 출력 | `RegistrationResult` dict (TypedDict) | 확장성, GUI 에서 시각화·로깅 선택 |
| 모델 상주 | 한 번에 한 descriptor 만 메모리 상주 (`preload_model`/`unload_model` 제공) | VRAM 안전, 수동 제어 가능 |
| 에러 정책 | 환경/프로그래머 오류만 예외, 매칭 실패는 `success=False` | GUI 가 UI 메시지로 정상 처리 |
| gluefactory 의존성 | 필요 모듈만 `_vendored/` 로 복사 | 레포 없이도 자급자족, 연구 레포 변동과 독립 |
| FPFH ckpt | 레포에 포함 (3 MB) | clone 만으로 바로 동작 |
| SHOT ckpt | 레포 제외(`.gitignore`), 별도 전달 후 `checkpoints/` 배치 | 256 MB, LFS/Release 부담 회피 |
| Python 버전 | 3.13 고정 | 기존 `shot_module.cp313-win_amd64.pyd` 바인딩 일치 |
| PCL | 1.15.1, 팀원 수동 설치 (문서 가이드) | conda 배포 부재, `setup_windows.md` 로 안내 |
| SHOT Windows 동작 검증 | **미검증(unknown)**. 팀원 PC 스모크테스트로 실증, **검증 통과 후 통합 릴리즈** | Linux 에서만 확인된 상태. 배포 전에 본인이 먼저 뚫지 않음(사용자 결정 B+Q) |

---

## 3. 디렉토리 구조

```
experiments/v1_20260415/0417_shared/
├── README.md                         # TL;DR / 설치 / API / 예제 / 트러블슈팅
├── environment/
│   ├── environment.yml               # conda env (python=3.13, torch+cuda, open3d, ...)
│   ├── requirements.txt              # pip 폴백
│   └── setup_windows.md              # PCL 1.15.1 설치 + DLL 경로 체크리스트
│
├── depth_registration/               # GUI 가 import 하는 실제 패키지
│   ├── __init__.py                   # register_pair, preload_model, unload_model, RegistrationResult export
│   ├── api.py                        # 최상위 API
│   ├── preprocessing.py              # load_depth_raw / mask_scanned_table / apply_bilateral / zmap_to_pcd_mm
│   ├── iss.py                        # detect_iss_mm
│   ├── descriptors/
│   │   ├── __init__.py
│   │   ├── fpfh.py                   # compute_fpfh (Open3D)
│   │   └── shot.py                   # compute_shot (pybind shot_module, lazy import)
│   ├── cache.py                      # master_cache_key / load_master_cache / save_master_cache
│   ├── matcher.py                    # load_lightglue / run_lightglue
│   ├── registration.py               # ransac_rigid (SVD, 1000 iter)
│   ├── viz.py                        # render_overlay_png / render_matches_png (선택)
│   ├── params.py                     # 좌표/ISS/FPFH/SHOT/RANSAC 상수
│   └── _vendored/                    # gluefactory 에서 복사한 모듈 (아래 §9)
│
├── checkpoints/
│   ├── iss_fpfh_v1_norm_0417.tar     # FPFH ckpt (3 MB, 레포 포함)
│   └── iss_shot_v1_dim352_0417.tar   # SHOT ckpt (256 MB, .gitignore, 별도 전달)
│
├── cache/
│   └── master_<img8>_<params8>_fpfh.npz   # pre-built (동봉). SHOT 캐시는 첫 실행 시 생성
│
├── masters/
│   └── default_master.png            # gluefactory/datasets/scanned/blender_master1/master.png 복사본
│
├── examples/
│   ├── run_cli.py                    # argparse CLI 스모크
│   └── gui_integration_demo.py       # GUI 측 사용 예시 10여 줄
│
└── tests/
    ├── test_smoke.py                 # 팀원 PC 에서 설치 검증용 end-to-end
    ├── test_cache.py                 # 캐시 hit/miss, 손상 복구
    └── fixtures/
        └── sample_scanned.png
```

`pybind_shot_window/` 는 레포 루트에 이미 존재하므로 `0417_shared/` 안에 복사하지 않고, `descriptors/shot.py` 의 `_setup_shot_dll()` 이 상대 경로로 참조한다.

---

## 4. 공개 API

```python
# depth_registration/__init__.py
from .api import register_pair, preload_model, unload_model, RegistrationResult
__version__ = "0.1.0"
__all__ = ["register_pair", "preload_model", "unload_model", "RegistrationResult"]
```

### 4.1 `register_pair`

```python
from typing import TypedDict, Literal
import numpy as np
from pathlib import Path

Descriptor = Literal["fpfh", "shot"]
ImageLike  = str | Path | np.ndarray        # 경로 또는 uint16 (H,W) 배열

class RegistrationResult(TypedDict):
    T:            np.ndarray   # (4,4) float64, T @ [scanned_xyz, 1] = [master_xyz, 1]
    R:            np.ndarray   # (3,3) float64
    t:            np.ndarray   # (3,)  float64, mm
    inlier_ratio: float        # num_inliers / max(num_matches, 1)
    num_matches:  int          # LightGlue 매치 수 (RANSAC 이전)
    num_inliers:  int
    matches_scanned_xyz: np.ndarray   # (num_inliers, 3) float32 mm, RANSAC inlier 만
    matches_master_xyz:  np.ndarray   # (num_inliers, 3) float32 mm
    success:      bool
    elapsed_ms:   dict          # keys: preproc, descriptor, lightglue, ransac, total

def register_pair(
    scanned:     ImageLike,
    master:      ImageLike | None = None,        # None → params.DEFAULT_MASTER_PATH
    descriptor:  Descriptor = "fpfh",
    *,
    cache_dir:   str | Path | None = None,       # None → 0417_shared/cache/
    inlier_th:   float = 5.0,                    # mm
    ransac_iter: int = 1000,
    success_min_inlier_ratio: float = 0.1,
    device:      str = "cuda",
) -> RegistrationResult: ...
```

### 4.2 `preload_model` / `unload_model`

```python
def preload_model(descriptor: Descriptor, device: str = "cuda") -> None: ...
def unload_model(descriptor: Descriptor | None = None) -> None: ...
```

GUI 는 앱 시작 시 `preload_model("fpfh")` 을 호출해 첫 매칭 지연을 피할 수 있다.

---

## 5. 내부 모듈 시그니처

```python
# preprocessing.py
def load_depth_raw(img: ImageLike) -> np.ndarray:            # uint16 (H,W)
def mask_scanned_table(zmap: np.ndarray,                      # histogram floor 제거
                       band_fraction: float = 0.05,
                       bin_width: int = 50) -> np.ndarray:
def apply_bilateral(zmap: np.ndarray,
                    d: int = 5, sigma_c: float = 100.0,
                    sigma_s: float = 3.0) -> np.ndarray:
def zmap_to_pcd_mm(zmap: np.ndarray,
                   erode_boundary_px: int = 5) -> np.ndarray: # (N,3) float32 mm
def preprocess_scanned(img: ImageLike) -> np.ndarray:         # floor mask → bilateral → mm PCD
def preprocess_master (img: ImageLike) -> np.ndarray:         # mm PCD 만 (floor/bilateral 없음)

# iss.py
def detect_iss_mm(pts_mm: np.ndarray,
                  max_keypoints: int = 512,
                  seed: int = 0) -> np.ndarray:               # (K,3) float32

# descriptors/fpfh.py
def compute_fpfh(pts_mm: np.ndarray, kp_mm: np.ndarray,
                 voxel: float = 1.0, normal_r: float = 20.0,
                 fpfh_r: float = 20.0) -> np.ndarray:          # (K,33) float32, raw (no L2 norm)

# descriptors/shot.py
def compute_shot(pts_mm: np.ndarray, kp_mm: np.ndarray,
                 voxel: float = 1.0, normal_r: float = 20.0,
                 shot_r: float = 40.0) -> np.ndarray:          # (K,352) float32

# cache.py
def master_cache_key(zmap: np.ndarray, descriptor: Descriptor, params: dict) -> str:
def load_master_cache(key: str, cache_dir: Path) -> dict | None:
def save_master_cache(key: str, payload: dict, cache_dir: Path) -> None:

# matcher.py
def load_lightglue(descriptor: Descriptor, device: str) -> torch.nn.Module:
def run_lightglue(model, kp0_uv, desc0, kp1_uv, desc1,
                  image_size_wh: tuple[int,int]) -> dict:      # matches0, matching_scores0

# registration.py
def ransac_rigid(src_mm: np.ndarray, dst_mm: np.ndarray,
                 n_iter: int = 1000, inlier_th: float = 5.0,
                 seed: int = 0) -> tuple[np.ndarray, np.ndarray]:  # (T 4x4, inlier_mask M,)

# viz.py (선택)
def render_overlay_png(scanned_zmap, master_zmap, result, out_path=None) -> bytes:
def render_matches_png(scanned_zmap, master_zmap, result, out_path=None) -> bytes:
```

### 5.1 FPFH normalize 금지

`compute_fpfh` 는 **L2 정규화를 적용하지 않는다**. v1 학습 캐시가 raw histogram 으로 작성됐기 때문에 추론에서 정규화하면 train/test 분포가 어긋난다. SHOT 은 PCL 구현이 자체적으로 unit-sphere 정규화를 수행하므로 별도 처리 불요.

### 5.2 mm ↔ uv 역변환

LightGlue 는 uv 공간 positional encoding 을 사용하므로, mm 좌표의 keypoint 를 `u = X / LATERAL_MM`, `v = Y / TRANSPORT_MM` 로 변환해서 입력한다. 이미지 크기는 `(PAD_W, PAD_H) = (2432, 3008)` 로 고정.

---

## 6. 상수 — `params.py`

```python
# 좌표/단위 (v1_20260415)
LATERAL_MM   = 0.056
TRANSPORT_MM = 0.056
VERTICAL_MM  = 0.0085

# Descriptor
VOXEL_MM    = 1.0
NORMAL_R_MM = 20.0
FPFH_R_MM   = 20.0
SHOT_R_MM   = 40.0

# ISS
ERODE_BOUNDARY_PX  = 5
ISS_SALIENT_MULT   = 6.0
ISS_NONMAX_MULT    = 2.0
ISS_GAMMA_21       = 0.5
ISS_GAMMA_32       = 0.5
ISS_MIN_NEIGHBORS  = 5
MAX_KEYPOINTS      = 512

# Scanned 전처리
BILAT_D = 5
BILAT_SIGMA_C = 100.0
BILAT_SIGMA_S = 3.0
FLOOR_BAND_FRACTION = 0.05
FLOOR_BIN_WIDTH     = 50

# LightGlue 입력 (학습 zero-pad)
PAD_H = 3008
PAD_W = 2432

# RANSAC
DEFAULT_RANSAC_ITER = 1000
DEFAULT_INLIER_TH   = 5.0

# 기본 경로
PKG_DIR             = Path(__file__).resolve().parent
SHARED_DIR          = PKG_DIR.parent                      # 0417_shared/
DEFAULT_MASTER_PATH = SHARED_DIR / "masters" / "default_master.png"
DEFAULT_CACHE_DIR   = SHARED_DIR / "cache"
CKPT_FPFH = SHARED_DIR / "checkpoints" / "iss_fpfh_v1_norm_0417.tar"
CKPT_SHOT = SHARED_DIR / "checkpoints" / "iss_shot_v1_dim352_0417.tar"
```

사용자가 API 인자로 덮어쓸 수 있는 값은 `inlier_th`, `ransac_iter`, `success_min_inlier_ratio`, `device` 뿐이다. 좌표/ISS/Descriptor 파라미터는 학습 설정에 고정이며 변경 시 성능이 보장되지 않는다.

---

## 7. 데이터 / 제어 흐름

```
GUI
  │  from depth_registration import register_pair
  │  r = register_pair(scanned, descriptor="fpfh")
  ▼
api.register_pair
  ├─ (scanned) load_depth_raw → mask_scanned_table → apply_bilateral → zmap_to_pcd_mm
  │           → detect_iss_mm → compute_fpfh/compute_shot
  │
  ├─ (master) zmap_to_pcd_mm → cache.master_cache_key
  │           hit  → load_master_cache            (kp_mm, desc, pts_mm)
  │           miss → detect_iss_mm → compute_* → save_master_cache
  │
  ├─ matcher: load_lightglue(descriptor, device) [lazy, 모델 상주 정책 §8]
  │           kp_mm → uv 역변환 → run_lightglue → matches0, scores
  │
  ├─ mm 매칭쌍 정렬:   src_mm (scanned), dst_mm (master)
  │
  ├─ registration.ransac_rigid(src_mm, dst_mm, ransac_iter, inlier_th)
  │           → T (4,4), inlier_mask (M,)
  │
  └─ RegistrationResult dict 채워 반환
```

각 단계는 `time.perf_counter` 로 측정해 `elapsed_ms` 에 기록한다.

---

## 8. 모델 상주 정책

모듈 전역에 `_ACTIVE_MODEL: tuple[Descriptor, torch.nn.Module] | None = None` 를 둔다.

`register_pair(descriptor=X)` 호출 시:
1. `_ACTIVE_MODEL` 이 없거나 다른 descriptor 이면, 기존 모델이 있으면 `del` 후 `torch.cuda.empty_cache()`, 그 다음 X 로드.
2. `_ACTIVE_MODEL` 이 X 면 재사용.

`preload_model(descriptor)` 는 동일 절차를 명시적으로 수행한다. `unload_model(None)` 은 `_ACTIVE_MODEL = None` 으로 리셋 + cache empty.

스위칭 비용: FPFH ckpt 로딩 ~1 초 미만, SHOT ckpt 로딩 수 초. GUI 가 두 descriptor 를 자주 번갈아 쓰는 워크플로면 성능 영향이 있을 수 있다. 본 버전은 "한 번에 하나" 를 채택하되, 필요 시 `preload_model` 을 순차 호출하는 식으로 워밍업만 해둘 수 있다.

---

## 9. Vendored 모듈 — `_vendored/`

gluefactory 레포의 아래 파일들을 그대로 복사한다. 경로는 흡수 후 패키지 내부 임포트 경로로 변경한다.

| 원본 경로 | vendored 경로 |
|---|---|
| `gluefactory/models/matchers/lightglue.py` | `_vendored/lightglue.py` |
| `gluefactory/models/utils/*` (lightglue 의존 유틸) | `_vendored/model_utils/` |
| `gluefactory/utils/tensor.py`::`batch_to_device` | `_vendored/tensor.py` |
| `gluefactory/datasets/new_dataset/iss_detection.py` | `_vendored/iss_detection.py` |

흡수 후 `_vendored/__init__.py` 에서 최소 심볼만 re-export. 본 패키지의 `matcher.py`, `iss.py` 는 `from .._vendored import LightGlue, detect_iss_keypoints, ...` 형태로 참조한다. 연구 레포의 원본 파일은 건드리지 않는다 (단방향 복사).

업데이트 정책: v1 모델이 고정되어 있으므로 vendored 파일의 재동기화는 필요 없다. 모델을 교체하는 시점에 수동으로 diff 후 복사.

---

## 10. Master 캐시 정책

### 10.1 파일 포맷

경로: `{cache_dir}/master_{img_hash8}_{params_hash8}_{descriptor}.npz` (압축 npz)

필드:
- `kp_mm`: (K, 3) float32
- `desc`: (K, 33) FPFH or (K, 352) SHOT float32
- `pts_mm`: (N, 3) float32 — master 전체 mm PCD (시각화/디버깅)
- `img_hash`: str — `sha1(zmap.tobytes())[:8]`
- `params_hash`: str — `sha1(json.dumps(params, sort_keys=True))[:8]`
- `descriptor`: str
- `params_json`: str — 실제 파라미터 dict 직렬화
- `created_at`: ISO 8601 str
- `version`: 패키지 버전

### 10.2 해시 대상 파라미터

`VOXEL_MM`, `NORMAL_R_MM`, `FPFH_R_MM` or `SHOT_R_MM`, `ISS_SALIENT_MULT`, `ISS_NONMAX_MULT`, `ISS_GAMMA_21`, `ISS_GAMMA_32`, `ISS_MIN_NEIGHBORS`, `MAX_KEYPOINTS`, `ERODE_BOUNDARY_PX`, `LATERAL_MM`, `TRANSPORT_MM`, `VERTICAL_MM`.

### 10.3 동봉 / 자동 생성

- 기본 master 의 **FPFH 캐시** 는 빌드 스크립트로 미리 생성해 레포에 커밋한다.
- 기본 master 의 **SHOT 캐시** 는 레포에 포함하지 않고 첫 실행 시 자동 생성한다(용량 불확정, 일관성 확보).
- 사용자가 다른 master 이미지를 넘기면 miss 가 발생해 자동 생성된다.

### 10.4 엣지 케이스

| 상황 | 동작 |
|---|---|
| `cache_dir` 부재 | `mkdir(parents=True, exist_ok=True)` |
| npz 로드 실패 (손상) | warn 로그, 파일 삭제, 재계산 |
| 동시 실행 race | 단일 GUI 프로세스 가정, 락 미구현 |

---

## 11. 정합 알고리즘

`ransac_rigid(src_mm, dst_mm, n_iter=1000, inlier_th=5.0 mm)` 는 기존 `test_registration_resample2_iss_shot.py` 의 구현을 그대로 복사해 사용한다 (3점 샘플 → SVD 최소해 → inlier count). 방향은 `T @ [src, 1] = [dst, 1]` = scanned → master.

매치 < 3 이면 `T = I`, `inlier_mask = False` 반환하고, `api.register_pair` 에서 `success=False` 로 마감한다.

`success` 판정:
```
success = (num_inliers >= 3) and (inlier_ratio >= success_min_inlier_ratio)
```

---

## 12. 시각화 정책

시각화는 본 API 에 포함하지 않는다. GUI 가 `matches_*_xyz` 와 `T` 를 이용해 자체 렌더링한다.

편의용으로 `depth_registration/viz.py` 를 별도 제공한다:
- `render_overlay_png`: master / scanned-transformed / overlay 3-panel PNG (matplotlib)
- `render_matches_png`: 매칭 라인 4색 (기존 `test_resample2_iss_shot_0408.py` 와 동일 팔레트)

두 함수는 PNG 바이트를 반환하며 옵션으로 파일 저장도 가능하다. GUI 스모크 단계에서만 쓰고 최종 UI 에선 교체될 가능성이 높다.

---

## 13. 에러 처리 정책

| 상황 | 처리 |
|---|---|
| scanned depth 전부 0 | warn 로그, `success=False`, `T=I` |
| ISS < MAX_KEYPOINTS | voxel cloud 에서 랜덤 패딩 (v1 정책) — 정상 |
| LightGlue 매치 < 4 | `success=False`, RANSAC 스킵, `T=I` |
| inlier_ratio < threshold | `success=False`, T 는 채워서 반환 |
| `shot_module` import 실패 | `RuntimeError("SHOT requires PCL 1.15.1: see environment/setup_windows.md")` |
| ckpt 파일 없음 | `FileNotFoundError` (실제 경로 포함) |
| 미지원 descriptor 문자열 | `ValueError` |

환경/프로그래머 오류만 예외로 raise 한다. 매칭 실패는 예외가 아닌 `success=False` 필드로 표현해 GUI 가 UI 상에서 정상 처리할 수 있게 한다.

---

## 14. 환경 구축

### 14.1 `environment/environment.yml`

```yaml
name: depth_reg
channels: [pytorch, nvidia, conda-forge, defaults]
dependencies:
  - python=3.13
  - pytorch
  - pytorch-cuda=12.1     # 팀원 GPU CUDA 에 맞춰 조정
  - numpy>=2.0
  - opencv>=4.10
  - pillow>=11.0
  - open3d>=0.18
  - omegaconf
  - matplotlib
  - pytest
  - pip
  - pip:
      - pybind11>=3.0
```

### 14.2 `environment/requirements.txt`

conda 미사용 환경 폴백:
```
numpy>=2.0
torch>=2.0
opencv-python>=4.10
pillow>=11.0
open3d>=0.18
omegaconf
matplotlib
pytest
```

### 14.3 `environment/setup_windows.md`

팀원이 순서대로 따라가는 체크리스트:
1. Miniconda 설치 → `conda env create -f environment/environment.yml` → `conda activate depth_reg`.
2. `python -c "import torch; print(torch.cuda.is_available())"` 가 `True` 인지 확인. 아니면 `pytorch-cuda` 버전을 GPU 에 맞춰 조정 후 env 재생성.
3. (SHOT 사용 시) PCL 1.15.1 All-In-One MSVC2022 win64 설치. 기본 경로 `C:\Program Files\PCL 1.15.1\`.
4. (SHOT 사용 시) `iss_shot_v1_dim352_0417.tar` 를 별도 전달받아 `0417_shared/checkpoints/` 로 복사.
5. Smoke test: `pytest experiments/v1_20260415/0417_shared/tests/test_smoke.py -v`.

### 14.4 `descriptors/shot.py` 의 DLL 로더

```python
def _setup_shot_dll():
    if sys.platform != "win32":
        return
    pcl_root = Path(os.environ.get("PCL_ROOT", r"C:\Program Files\PCL 1.15.1"))
    for sub in ("bin", r"3rdParty\FLANN\bin", r"3rdParty\VTK\bin"):
        p = pcl_root / sub
        if p.exists():
            os.add_dll_directory(str(p))
    # __file__ = <repo>/experiments/v1_20260415/0417_shared/depth_registration/descriptors/shot.py
    # parents: 0=descriptors, 1=depth_registration, 2=0417_shared,
    #          3=v1_20260415, 4=experiments, 5=<repo_root>
    repo_root = Path(__file__).resolve().parents[5]
    pybind_dir = repo_root / "pybind_shot_window"
    sys.path.insert(0, str(pybind_dir))
```

import 시점(모듈 로딩)에 1회 실행.

---

## 15. 테스트

### 15.1 `tests/test_smoke.py`

```python
import numpy as np
from depth_registration import register_pair
from depth_registration.params import DEFAULT_MASTER_PATH

def test_fpfh_self_matching():
    """master ↔ master 매칭은 T ≈ I 여야 한다.

    scanned 경로에는 mask_scanned_table + bilateral 전처리가 들어가지만,
    default master 는 synthetic(Blender) 이라 floor band 가 없고 bilateral 은
    smoothing 에 불과하므로 keypoint 위치가 충분히 근접 → T ≈ I 가 성립한다.
    bilateral 영향을 고려해 atol 은 3 mm 로 여유를 둔다.
    """
    r = register_pair(DEFAULT_MASTER_PATH, DEFAULT_MASTER_PATH, descriptor="fpfh")
    assert r["success"]
    assert r["num_inliers"] >= 20
    assert np.allclose(r["T"], np.eye(4), atol=3.0)      # 3 mm 허용

def test_fpfh_sample_scanned():
    """샘플 scanned 로 파이프라인 관통."""
    r = register_pair("tests/fixtures/sample_scanned.png", descriptor="fpfh")
    assert r["num_matches"] > 0
    # success 여부는 fixture 에 따라 달라져 단정하지 않음
```

SHOT 테스트는 환경에 PCL 이 설치된 경우에만 조건부 실행(`@pytest.mark.skipif(not _has_pcl(), reason="PCL not installed")`).

### 15.2 `tests/test_cache.py`

- 동일 이미지·파라미터 두 번 호출 → 두 번째 hit
- 파라미터 한 가지 변경 → miss, 새 파일 생성
- npz 를 깨뜨린 뒤 호출 → 자동 재계산 + 파일 교체

### 15.3 Windows SHOT 검증 절차 (팀원 PC 에서 실행)

`shot_module.cp313-win_amd64.pyd` 는 현재 빌드만 된 상태이며 Windows 런타임 동작은 미검증이다. 팀원 PC 에서 아래 3단계 모두 통과해야 SHOT 배포가 유효하다고 본다. **3단계 통과 전에는 SHOT 경로를 사용하는 공식 릴리즈 를 선언하지 않는다**.

| Phase | 목표 | 명령 / 확인 |
|---|---|---|
| 1. Import | `.pyd` + PCL DLL 체인 로드 성공 | `python -c "import shot_module; print(dir(shot_module))"` → `extract_shot` 가 리스트에 있고 `ImportError` 없음 |
| 2. Extract 호출 | `extract_shot` 가 정상 dict 반환 | `python pybind_shot_window/extract_shot.py` (기본 master PNG 입력) → `points.shape = (M,3)`, `descriptors.shape = (M,352)`, `num_valid_desc > 0` |
| 3. 수치 일치 | Linux 결과와 수치적으로 근사 | 동일 master PNG 를 Linux 에서 한 번 돌려 `.npy` 로 내보낸 값과 비교: `np.linalg.norm(d_win - d_lin, axis=1).mean() < 1e-3` (tolerance 는 구현 시 결정 가능 — 현 스펙 값은 초기 가이드라인) |

Phase 1 실패 시: `ImportError: DLL load failed` 메시지 전문 확보 → `PCL_ROOT` 경로 재확인, MSVC 2022 VC++ Redistributable x64 설치 여부 확인, Python 3.13.x 실제 설치 버전 확인. 해결 불가 시 `pybind_shot_window/build.bat` 로 팀원 PC 에서 재빌드(§17 참조).

Phase 2 실패 시: Eigen alignment / PCL 내부 assert 가 의심. `/DEIGEN_MAX_ALIGN_BYTES=32` 가 빌드 플래그에 있었는지, MSVC 버전이 빌드 시점과 동일(v143, VS 2022)인지 확인.

Phase 3 불일치 시: PCL 버전이 Linux(시스템 패키지) ↔ Windows(1.15.1) 에서 다를 수 있음 → 허용 tolerance 와 원인 기록 후 진행 여부 판단.

---

## 16. 배포 단계

| # | 행동 | 확인 |
|---|---|---|
| 1 | `0417_shared/` 구조 생성, 기존 `share/depth_to_iss.py` 의 유용한 부분을 `preprocessing.py`/`iss.py` 로 통합 | 파일 트리 |
| 2 | gluefactory 에서 vendored 파일 복사 | `_vendored/` 채움 |
| 3 | 기본 master 복사 (`masters/default_master.png`) | 파일 존재 |
| 4 | FPFH ckpt 복사 (`checkpoints/iss_fpfh_v1_norm_0417.tar`) | 파일 존재 |
| 5 | `python -m ... precompute_default_master --descriptor fpfh` 로 FPFH pre-built 캐시 생성 | `cache/*_fpfh.npz` |
| 6 | `.gitignore`: `checkpoints/iss_shot_*.tar`, `cache/*_shot.npz` 제외 | git status 클린 |
| 7 | 본인 PC(Linux) 에서 smoke test 통과 확인 — FPFH end-to-end, SHOT 은 Linux 기준으로만 통과 | pytest green on Linux |
| 8 | git commit / push (내부 사전 공유 — 아직 "통합 릴리즈" 선언 아님) | 팀원 pull 가능 |
| 9 | 팀원에게 SHOT ckpt 별도 전달 (공유 드라이브 등) | 전달 완료 |
| 10 | 팀원 PC 에서 env 설치 + PCL 설치 + ckpt 배치 → FPFH smoke test | FPFH 테스트 green |
| 11 | 팀원 PC 에서 **Windows SHOT 검증 3 phase** (§15.3) 통과 확인 | Phase 1·2·3 모두 green |
| 12 | 11 실패 시: 로그·원인 기록 → 본인 측 패치 or 재빌드 → 10~11 재수행. 해결 전엔 **통합 릴리즈 미선언** | 실패 이슈 클로즈 |
| 13 | 11 통과 후 본인·팀원 합의로 **통합 릴리즈 태그** 부여 (`v0.1.0`) | git tag |

---

## 17. 범위 밖 / 향후 / 복구 경로

- ONNX / TensorRT 변환: 본 API 의 함수 시그니처를 유지한 채 `matcher.py` 내부 구현만 교체한다.
- Matching 결과 포맷 개선 (시각화, 로깅, 불확실성 추정 등): `viz.py` 확장 또는 `RegistrationResult` 필드 추가.
- 전처리 파라미터 튜닝 (floor band_fraction, bilateral sigma 등): 현재는 기존 값 고정.
- 다중 master 동시 비교, top-K 매칭 후보 반환: 필요 시 별도 함수 `register_multi(...)` 추가.
- 캐시 원자적 쓰기, 멀티프로세스 락: 현 버전은 단일 GUI 프로세스 가정.

### 17.1 Windows SHOT 재빌드 복구 경로

§15.3 Phase 1~2 가 팀원 PC 에서 실패하고 패치로도 해결되지 않으면, 팀원 PC 에서 `.pyd` 재빌드를 시도한다. 절차는 이미 레포에 포함된 `pybind_shot_window/` 가이드를 따른다:

1. Visual Studio 2022 Community + C++ workload 설치 (또는 Build Tools for VS 2022 만).
2. `vcvarsall.bat x64` 환경에서 `pybind_shot_window/build.bat` 실행 (PCL/Python/pybind11 경로는 팀원 PC 기준으로 편집).
3. 새 `shot_module.cp313-win_amd64.pyd` 가 생성되면 `§15.3` Phase 1~3 재시도.
4. 해결되면 팀원 PC 에서 빌드된 `.pyd` 를 레포에 커밋(팀원 PC 가 빌드 기준) 또는 별도 전달.

재빌드조차 실패할 정도의 환경 mismatch(예: Python 3.13 부재, PCL 1.15.1 설치 실패) 는 **SHOT 를 이번 릴리즈에서 제외**하고 FPFH only 로 축소 릴리즈를 재검토한다. 이는 §2 표의 원 결정(동시 릴리즈)과 충돌하므로 반드시 본인·팀원 합의로만 수행한다.

---

## 18. 결론

본 설계는 **"기존 동작 보존 + 단일 폴더 자급자족 + Python import 인터페이스"** 의 세 원칙으로 요약된다. 연구 레포의 검증된 코드를 `0417_shared/` 에 재배치하고, gluefactory 의존성을 vendored 로 흡수해 팀원 GUI 가 최소한의 설치로 바로 호출할 수 있게 한다. 구현 계획은 writing-plans 스킬로 이어서 작성한다.
