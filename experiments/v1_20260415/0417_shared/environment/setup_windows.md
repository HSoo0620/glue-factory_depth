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

**Phase 2: extract 호출**
```powershell
python pybind_shot_window/extract_shot.py
```
Expected: `points.shape = (M, 3)`, `descriptors.shape = (M, 352)`, `num_valid_desc > 0`.

**Phase 3: Linux 결과와 수치 일치**
Linux 측에서 동일 master PNG 로 계산한 `.npy` 를 전달받아 평균 L2 거리 비교. 허용 오차 `1e-3` (초기 가이드라인, 구현 시 조정).

Phase 1~3 모두 통과한 뒤에만 통합 릴리즈가 승인된다 (spec §16 단계 11~13).
