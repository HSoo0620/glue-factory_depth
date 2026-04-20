# 0417_shared — 코드 구조 지도

> **목적**: 어느 파일에 어떤 코드가 있는지, 데이터가 어떻게 흘러가는지 한눈에 본다.
> "이 동작은 어디를 고쳐야 하나?" 에 답할 수 있도록 설계된 문서.

- 사용법은 [USAGE.md](USAGE.md), 설치는 [../environment/setup_windows.md](../environment/setup_windows.md).
- 개발자용 요약·주의사항은 [../README.md](../README.md).

---

## 1. 디렉터리 트리 (한눈에)

```
0417_shared/
├── depth_registration/        # 패키지 본체 (import 대상)
│   ├── __init__.py            # public re-export: register_pair, preload_model, unload_model
│   ├── api.py                 ★ 메인 API 오케스트레이션 (scanned+master → T)
│   ├── params.py              ★ 고정 상수 (voxel, 좌표 스케일, ckpt 경로, PAD…)
│   ├── preprocessing.py       #  PNG/ndarray → mm PCD (bilateral, floor mask)
│   ├── iss.py                 #  ISS keypoint 검출 (Open3D compute_iss_keypoints)
│   ├── descriptors/
│   │   ├── fpfh.py            #  Open3D FPFH 33D + L2 normalize
│   │   └── shot.py            #  pybind shot_module 호출 (OS 별 바인딩 자동 경로)
│   ├── matcher.py             #  LightGlue ckpt 로드 + 추론 + mm↔uv 변환
│   ├── registration.py        #  Custom SVD RANSAC (1000 iter, inlier_th=5mm)
│   ├── cache.py               #  Master descriptor npz 캐시 (fingerprint)
│   ├── viz.py                 #  optional 시각화 (overlay/matches PNG)
│   ├── cli/
│   │   └── precompute_default_master.py   # 기본 master 캐시 사전 빌드 CLI
│   └── _vendored/             # 외부 모듈 스냅샷 (import path 고립 목적)
│       ├── lightglue.py                   #   gluefactory/models/matchers/lightglue.py
│       ├── iss_detection.py               #   precompute/iss_fpfh_norm.py 일부
│       ├── tensor.py                      #   gluefactory/utils/tensor.py
│       └── model_utils/{losses,metrics,misc}.py
├── tests/                     # pytest (19 tests)
│   ├── test_preprocessing.py
│   ├── test_iss.py
│   ├── test_fpfh.py
│   ├── test_shot.py           #  shot_module 있을 때만 실행 (skipif)
│   ├── test_cache.py
│   ├── test_registration.py
│   ├── test_smoke.py          #  register_pair end-to-end 3건
│   └── fixtures/
│       ├── sample_scanned.png             # 스모크용 scanned
│       ├── shot_master_linux_reference.npy (gitignored)
│       └── shot_master_linux_kp.npy        (gitignored)
├── examples/
│   ├── run_cli.py             #  argparse 스모크 (register_pair 호출)
│   ├── gui_integration_demo.py            # ★ GUI 통합 골격 (복붙용)
│   └── verify_shot_phase3.py              # Linux ↔ Windows SHOT 수치 일치 검증
├── masters/
│   └── default_master.png     # 기본 master (2501×2413 uint16)
├── checkpoints/
│   ├── iss_fpfh_v1_norm_0417.tar          # FPFH LightGlue (repo 포함, 3MB)
│   └── iss_shot_v1_dim352_0417.tar        # SHOT LightGlue (256MB, 심볼릭/수동 배치)
├── cache/                     # master descriptor npz (자동 생성, gitignored)
├── docs/
│   ├── USAGE.md               # 사용자 가이드 (입력 규격, 트러블슈팅)
│   └── STRUCTURE.md           # ← 이 문서
├── environment/
│   ├── environment.yml        # conda env (Python 3.13 Windows 기준)
│   ├── requirements.txt       # pip 대안
│   └── setup_windows.md       # Windows 설치 + 3-phase 검증
└── README.md
```

★ = 가장 자주 들여다보게 되는 파일.

---

## 2. 파일별 역할 & 라인 수

### 2.1 패키지 본체 `depth_registration/`

| 파일 | 줄 | 핵심 심볼 | 한줄 요약 |
|---|---:|---|---|
| `__init__.py` | 8 | `register_pair`, `preload_model`, `unload_model`, `RegistrationResult` | public 진입점 |
| `api.py` | 190 | `register_pair()`, `_get_model()` | scanned/master 를 ISS→FPFH/SHOT→LightGlue→RANSAC 로 묶는 오케스트레이터 |
| `params.py` | 50 | `LATERAL_MM`, `VOXEL_MM`, `FPFH_R_MM`, `PAD_H/W`, `CKPT_*` | 학습 설정과 동일해야 하는 모든 상수 |
| `preprocessing.py` | 95 | `load_depth_raw`, `preprocess_scanned`, `preprocess_master`, `mask_scanned_table` | PNG/ndarray → mm PCD |
| `iss.py` | 40 | `detect_iss_mm()` | Open3D ISS, max 512 랜덤 서브샘플 (seed=0) |
| `descriptors/fpfh.py` | 41 | `compute_fpfh()` | radius-only FPFH + **L2 normalize** (`_norm_0417` 학습 분포) |
| `descriptors/shot.py` | 106 | `compute_shot()`, `_setup_shot_import_path()` | OS 별 pybind → `extract_shot_at_keypoints`, PCL 내부 unit-sphere 정규화 그대로 |
| `matcher.py` | 99 | `load_lightglue()`, `run_lightglue()`, `mm_to_uv()` | ckpt 로드 + inference 배치 구성 (`view0/view1.image_size=PAD_H×PAD_W`) |
| `registration.py` | 53 | `ransac_rigid()`, `_svd_rigid()` | 3-point minimal solver × 1000 iter |
| `cache.py` | 46 | `master_cache_key()`, `load_master_cache()`, `save_master_cache()` | `SHA1(img+descriptor+params)` fingerprint npz |
| `viz.py` | 62 | `render_overlay_png()`, `render_matches_png()` | matplotlib Agg, optional. 핵심 경로와 분리 |
| `cli/precompute_default_master.py` | 68 | `main()` | `python -m depth_registration.cli.precompute_default_master --descriptor fpfh` |

### 2.2 Vendored `depth_registration/_vendored/`

| 파일 | 줄 | 원본 위치 | 비고 |
|---|---:|---|---|
| `lightglue.py` | 631 | `gluefactory/models/matchers/lightglue.py` | LightGlue 모델 정의 (수정 없이 복사) |
| `iss_detection.py` | 106 | `experiments/v1_20260415/precompute/iss_fpfh_norm.py` 일부 | `build_iss_pcd_uvd_scaled` 등. 현재 정합 경로에서는 직접 호출 안 함 (레거시) |
| `tensor.py` | 27 | `gluefactory/utils/tensor.py` | `batch_to_device`, `map_tensor` |
| `model_utils/losses.py` | 73 | `gluefactory/models/utils/losses.py` | LightGlue `NLLLoss` (추론용이지만 모델 생성 시 참조됨) |
| `model_utils/metrics.py` | 50 | 동상 | `matcher_metrics` |
| `model_utils/misc.py` | 70 | 동상 | 기타 유틸 |

**왜 vendored 인가:** teammate 가 배포 패키지만 받아도 `gluefactory` 전체 의존 없이 동작하도록, 필요한 파일만 스냅샷해 import path 를 `depth_registration._vendored` 로 고립시킴. 업스트림 변경과 drift 주의.

### 2.3 테스트 `tests/`

| 파일 | 줄 | 검증 대상 |
|---|---:|---|
| `test_preprocessing.py` | 62 | `load_depth_raw`, `mask_scanned_table`, bilateral, mm 변환, master/scanned 분기 |
| `test_iss.py` | 27 | 기본 검출 + empty input |
| `test_fpfh.py` | 16 | shape=(K,33), dtype=float32, **L2-norm ≈ 1 또는 0** |
| `test_shot.py` | 28 | shape=(K,352), unit-norm. `shot_module` 없으면 `skipif` |
| `test_cache.py` | 51 | key 결정성, roundtrip, miss, 손상 파일 처리 |
| `test_registration.py` | 37 | `ransac_rigid` 가 rigid transform 복원 + n<3 guard |
| `test_smoke.py` | 48 | `register_pair` 3건: FPFH 자가매칭 / 샘플 scanned / SHOT 자가매칭 |

### 2.4 예제 `examples/`

| 파일 | 줄 | 용도 |
|---|---:|---|
| `run_cli.py` | 33 | CLI 스모크. `--scanned --master --descriptor {fpfh,shot}` |
| `gui_integration_demo.py` | 166 | GUI 통합 골격. `on_app_start / on_register_clicked / on_app_exit` 3단 |
| `verify_shot_phase3.py` | 91 | Linux 에서 `--produce-reference`, Windows 에서 비교. mean L2 ≤ 1e-3 판정 |

---

## 3. 데이터 흐름 (`register_pair` 내부)

```
scanned PNG (uint16)                       master PNG (uint16)
       │                                          │
       │ preprocess_scanned                       │ preprocess_master
       │   • load_depth_raw                       │   • load_depth_raw
       │   • mask_scanned_table ← floor 제거      │   (bilateral 없음, mask 없음)
       │   • apply_bilateral                      │
       │   • zmap_to_pcd_mm (erode 5px)           │   • zmap_to_pcd_mm
       ▼                                          ▼
   scanned_pts  (N,3) float32 mm              master_pts  (M,3)
       │                                          │
       │ detect_iss_mm (max 512)                  │ ⎫ master 캐시 HIT → skip
       ▼                                          │ ⎬  (cache/<key>.npz 에 kp+desc)
   scanned_kp   (K,3) mm                           │ ⎭
       │                                          │ detect_iss_mm
       │ compute_fpfh / compute_shot              │ compute_fpfh / compute_shot
       ▼                                          ▼
   scanned_desc (K, 33 or 352) f32           master_desc   (K', …)
       │                                          │
       └──────────────────┬───────────────────────┘
                          │ mm_to_uv (÷LATERAL_MM, ÷TRANSPORT_MM)
                          │ run_lightglue
                          │   batch = {keypoints0/1, descriptors0/1,
                          │            keypoint_scores0/1,
                          │            view0/1.image_size=(PAD_H, PAD_W)}
                          ▼
                     matches0  (K,)  int (−1 이면 unmatched)
                          │
                          │ valid = matches0 ≥ 0
                          │ src_mm = scanned_kp[valid]
                          │ dst_mm = master_kp[matches0[valid]]
                          ▼
                     ransac_rigid (1000 iter, inlier_th=5 mm)
                          │
                          ▼
                 RegistrationResult { T, R, t, inlier_mask, elapsed_ms, … }
```

**핵심 포인트**:

1. **좌표계는 줄곧 mm.** LightGlue 에 넘길 때만 `mm_to_uv` 로 잠깐 픽셀화했다가, 결과는 다시 mm 매칭 쌍으로 복원.
2. **master 쪽은 캐시 우선.** 같은 master+descriptor+params 조합이면 디스크에서 바로 로드.
3. **scanned 쪽만 floor 제거/bilateral.** master 는 "깨끗한 입력" 가정. 이 비대칭은 `preprocess_scanned` vs `preprocess_master` 에서 결정됨.

---

## 4. 모듈 간 의존성 (간이 그래프)

```
                 ┌──────────────────┐
                 │   api.register_  │
                 │      pair        │
                 └─────────┬────────┘
           ┌─────────┬─────┼──────────┬────────────┐
           ▼         ▼     ▼          ▼            ▼
     preprocessing  iss  descriptors  matcher    cache
        │            │      │fpfh      │           │
        │            │      │shot ─────┼── _vendored.lightglue
        │            │      │          │           │ + model_utils/*
        └── params ──┴──────┴──────────┴───────────┘
```

- 모든 모듈은 `params.py` 만 공통 의존.
- `_vendored/` 는 `matcher.py` 를 통해서만 간접 사용됨.
- `viz.py` 는 위 그래프에 포함되지 않는 **선택 모듈** (matplotlib 의존성 격리).

---

## 5. 외부 의존성 (0417_shared 바깥)

0417_shared 디렉터리 자체에는 들어있지 않지만 런타임에 필요한 파일들:

| 대상 | 위치 (repo root 기준) | 용도 | 부재 시 증상 |
|---|---|---|---|
| SHOT Linux binding | `pybind_shot_linux/shot_module.cpython-310-x86_64-linux-gnu.so` | Linux SHOT 실행 | `test_shot*` SKIP, `compute_shot` 호출 시 `RuntimeError` |
| SHOT Windows binding | `pybind_shot_window/shot_module.cp313-win_amd64.pyd` | Windows SHOT 실행 | 동상 |
| PCL 1.15.1 (Windows) | `C:\Program Files\PCL 1.15.1\` (또는 `PCL_ROOT` 환경변수) | SHOT binding 이 로드하는 DLL | `ImportError: DLL load failed` |
| SHOT ckpt | `checkpoints/iss_shot_v1_dim352_0417.tar` | SHOT LightGlue 가중치 (256MB) | `FileNotFoundError` on `preload_model("shot")` |

바인딩 경로는 `depth_registration/descriptors/shot.py:_setup_shot_import_path()` 가 `_repo_root()/parents[5]` 기준으로 자동 탐색.

---

## 6. "이 동작은 어디를 고쳐야 하나?" 역인덱스

| 현상/요구 | 첫 suspect 파일 | 근거 |
|---|---|---|
| scanned 에 floor/table 이 남아 keypoint 가 거기 몰림 | `preprocessing.py : mask_scanned_table` | histogram-peak band 제거 로직 |
| keypoint 수가 항상 512 인 게 아쉬움 | `params.py : MAX_KEYPOINTS` + `iss.py` | random subsample 상한 |
| FPFH 결과가 L2-norm 이 0 에 몰려 매칭이 0 | `descriptors/fpfh.py` | radius 가 너무 작아 descriptor 가 0 벡터. `FPFH_R_MM` 확인 |
| SHOT 로드 시 DLL 에러 | `descriptors/shot.py : _setup_shot_import_path` | PCL 경로/OS 분기 |
| LightGlue 에서 매칭이 거의 없음 | `matcher.py : _default_matcher_conf(filter_threshold=0.1)` | threshold 낮추면 매칭 수↑(노이즈도↑) |
| RANSAC 이 자주 실패 | `registration.py : ransac_rigid(inlier_th)` | inlier_th 기본 5mm, `register_pair(inlier_th=...)` 로 조정 가능 |
| master 를 바꿨는데 캐시가 옛날 것 | `cache.py : master_cache_key` + `cache/` 디렉터리 | 이미지 바이트 해시 기반이라 자동 재계산되어야 정상. 안 되면 키 로직 확인 |
| preload 후 다른 descriptor 를 부르면 느림 | `api.py : _get_model` | descriptor/device 가 바뀌면 기존 모델 언로드 + 새로 로드 |
| 결과 T 방향(scanned↔master)이 헷갈림 | `api.py : src_mm/dst_mm` 조립부 | `src = scanned_kp, dst = master_kp[m]` → `T @ scanned = master` |
| scanned PNG 크기가 달라서 매칭 품질이 떨어짐 | `params.py : PAD_H/PAD_W` + `matcher.py : run_lightglue` | LightGlue `normalize_keypoints` 가 이 크기 기준 |

---

## 7. 수정하지 말아야 할 것 (locked)

- `params.py` 의 상수 — 학습 설정과 동일해야 분포가 맞음. 특히 `LATERAL_MM / VERTICAL_MM` 은 센서 보정값, `FPFH_R_MM / SHOT_R_MM` 은 학습 radius.
- `_vendored/lightglue.py` — 업스트림과 drift 시 ckpt state_dict 로드가 깨짐.
- `descriptors/fpfh.py` 의 **`SearchParamRadius` + L2 normalize** — 학습 `_norm_0417` 분포와 매칭.
- `descriptors/shot.py` 의 **추가 정규화 금지** — PCL 이 내부에서 unit-sphere 정규화, 이중 적용하면 분포 깨짐.

---

## 8. 학습/훈련 코드 위치 (참고용)

이 패키지는 배포 전용이라 훈련 코드를 포함하지 않는다. 원본은 repo 내 다음 경로:

- 학습 엔트리: `experiments/v1_20260415/train_*.sh`
- Precompute (FPFH/SHOT 캐시 빌드): `experiments/v1_20260415/precompute/iss_fpfh_norm.py`, `iss_shot.py`
- Config: `gluefactory/configs/iss_fpfh_v1_norm_0417.yaml`, `iss_shot_v1_dim352_0417.yaml`
- Dataset: `gluefactory/datasets/mitsubishi_resample2_iss_*.py`

배포 패키지의 동작이 학습과 다르게 보이면 위 경로의 precompute 스크립트를 먼저
확인하는 것이 정답 — 대부분 "학습은 이렇게 했는데 배포는 저렇게 함" 의 drift 가 원인.
