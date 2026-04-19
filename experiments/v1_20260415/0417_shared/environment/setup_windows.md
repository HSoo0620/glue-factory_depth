# Windows 설치 체크리스트

> 0417_shared 패키지를 Windows PC 에서 구동하기 위한 절차.
> Python 3.13 고정 (SHOT pybind 바인딩 `shot_module.cp313-win_amd64.pyd` 와 일치해야 함).

## 1. Python 환경

```powershell
# Miniconda 또는 Anaconda 설치 후
conda env create -f environment/environment.yml
conda activate depth_reg
```

## 2. CUDA 확인

```powershell
python -c "import torch; print(torch.cuda.is_available())"
```
`True` 가 아니면 `environment.yml` 의 `pytorch-cuda` 버전을 GPU 에 맞춰 조정 후 env 재생성.

## 3. PCL 1.15.1 (SHOT 사용 시만 필요)

https://github.com/PointCloudLibrary/pcl/releases
- `PCL-1.15.1-AllInOne-msvc2022-win64.exe` 다운로드 및 설치
- 기본 경로: `C:\Program Files\PCL 1.15.1\`
- 다른 경로 사용 시 환경변수 `PCL_ROOT` 설정

## 4. SHOT ckpt 수동 배치 (SHOT 사용 시)

별도 전달받은 `iss_shot_v1_dim352_0417.tar` 를 다음 경로에 복사:
```
experiments/v1_20260415/0417_shared/checkpoints/iss_shot_v1_dim352_0417.tar
```

## 5. Smoke test

```powershell
cd experiments/v1_20260415/0417_shared
pytest tests/test_smoke.py -v
```
- `test_fpfh_*` 모두 PASS.
- `test_shot_self_matching` 은 PCL 미설치 시 SKIP. 설치 후엔 PASS.

## 6. SHOT 3-phase 검증 (spec §15.3)

**Phase 1: import**
```powershell
python -c "import shot_module; print(dir(shot_module))"
```
Expected: `extract_shot_at_keypoints` 가 리스트에 포함. `ImportError: DLL load failed` 시 PCL 설치/DLL 경로 재점검.

**Phase 2: v1 API (`extract_shot_at_keypoints`) 검증**

`pybind_shot_window/extract_shot.py` 는 v5 파라미터 용이라 v1 배포 검증에는 부적합.
대신 배포 패키지가 실제로 사용하는 API 를 호출하는지 확인:

```powershell
python -c "import shot_module; print('functions:', [f for f in dir(shot_module) if not f.startswith('_')])"
```
Expected 출력에 `extract_shot_at_keypoints` 포함.

이어서 배포 코드 경로로 compute_shot 까지 돌려본다:

```powershell
python -c "
import sys; sys.path.insert(0, '.')
from depth_registration.preprocessing import load_depth_raw, preprocess_master
from depth_registration.iss import detect_iss_mm
from depth_registration.descriptors.shot import compute_shot
from depth_registration.params import DEFAULT_MASTER_PATH
import numpy as np
z = load_depth_raw(DEFAULT_MASTER_PATH)
pts = preprocess_master(z)
kp = detect_iss_mm(pts, seed=0)
desc = compute_shot(pts, kp)
norms = np.linalg.norm(desc, axis=1)
print(f'desc={desc.shape}, unit-norm rows={(np.isclose(norms, 1.0, atol=1e-3)).sum()}/512')
"
```
Expected: `desc=(512, 352)`, `unit-norm rows=512/512` (PCL 이 자동 unit-sphere 정규화).

**Phase 3: Linux 결과와 수치 일치**

Linux 에서 미리 생성된 `tests/fixtures/shot_master_linux_reference.npy` 와 비교.
(이 파일은 `.gitignore` 대상이라 repo 에 없음 — 본인에게서 별도 전달받아
동일 경로에 배치)

```powershell
python examples/verify_shot_phase3.py
```
Expected: `[PASS] mean L2 <= 1e-3`. 실패 시 ISS keypoint 순서 차이 가능성 존재
(v1 detect_iss_mm 은 seed=0 결정적이나 플랫폼 간 tie-break 로 미세 순서 차 가능).

Phase 1~3 모두 통과한 뒤에만 통합 릴리즈가 승인된다 (spec §16 단계 11~13).
