# 0417_shared — 사용자 가이드

> 이 문서는 `0417_shared` 를 **GUI/애플리케이션 측에서 import 해서 사용하는 사람**을 위한 것이다.
> 설치·환경 세팅은 `../README.md` 의 "설치" 절과 `../environment/setup_windows.md` 참고.

---

## 1. 30초 요약

```python
from depth_registration import preload_model, register_pair

preload_model("fpfh", device="cuda")             # 앱 시작 시 1회
result = register_pair("scan.png",               # 버튼 클릭 시마다 호출
                       descriptor="fpfh")
if result["success"]:
    T = result["T"]                              # (4,4) float64, mm, scanned→master
```

- **scanned/master PNG** → **4×4 rigid transform T** (`T @ scanned_xyz = master_xyz`)
- 좌표/단위: mm
- 성공 여부: `result["success"]` (`num_inliers ≥ 3` 이고 `inlier_ratio ≥ 0.1`)

---

## 2. 입력 규격 (중요)

### 2.1 PNG 파일

| 항목 | 요구값 |
|---|---|
| dtype | `uint16` (H, W) single-channel. uint8 도 받지만 자동 ×257 스케일 — 정확도 저하 가능 |
| 크기 | 학습 기준 **3008 × 2432** (H × W). 다른 크기도 동작하지만 정확도는 미보증 |
| 0 값 | **invalid/background** 로 취급됨 (point cloud 에서 제외) |
| 단위 | raw 값 자체. 내부에서 `Z = raw × 0.0085 mm` 로 변환 |

### 2.2 `np.ndarray` 로 직접 전달

`register_pair(scanned, master)` 의 `scanned` / `master` 는 문자열 경로 외에
`np.ndarray[uint16]` 도 받는다. GUI 가 depth 를 메모리에서 바로 갖고 있을 때 PNG
저장을 우회할 수 있다.

```python
scan_u16 = depth_sensor_read()            # np.ndarray[uint16], shape (H, W)
result = register_pair(scan_u16, descriptor="fpfh")
```

### 2.3 좌표/단위 변환 (내부)

```
X = u × 0.056 mm   (LATERAL_MM)
Y = v × 0.056 mm   (TRANSPORT_MM)
Z = raw × 0.0085 mm (VERTICAL_MM)
```

이 값들은 학습 설정과 동일해야 해서 **외부 override 불가**. 센서가 달라서 스케일
자체가 다르면 패키지 자체가 맞지 않는다 (담당자 상의 필요).

---

## 3. API 레퍼런스 (꼭 알아야 할 것만)

### 3.1 `register_pair(...)`

```python
register_pair(
    scanned: str | Path | np.ndarray,               # 필수
    master:  str | Path | np.ndarray | None = None, # None → 기본 master 자동 사용
    descriptor: "fpfh" | "shot" = "fpfh",
    *,
    cache_dir:   Path | None = None,                # None → <shared>/cache
    inlier_th:   float = 5.0,                       # RANSAC inlier 거리 임계 (mm)
    ransac_iter: int   = 1000,
    success_min_inlier_ratio: float = 0.1,
    device: str = "cuda",
) -> RegistrationResult
```

반환 dict:

| 키 | 타입 | 의미 |
|---|---|---|
| `T` | `ndarray(4,4) float64` | scanned → master rigid transform (mm) |
| `R`, `t` | `(3,3)`, `(3,)` float64 | `T` 의 회전·병진 분해본 |
| `success` | `bool` | GUI 가 최종 판정용으로 쓸 필드 |
| `num_matches` | `int` | LightGlue 가 반환한 매칭 수 (512 keypoint 중) |
| `num_inliers` | `int` | RANSAC 통과한 매칭 수 |
| `inlier_ratio` | `float` | `num_inliers / num_matches` |
| `matches_scanned_xyz` | `ndarray(M,3) float32` | **inlier 만**, scanned 좌표계 (mm) |
| `matches_master_xyz`  | `ndarray(M,3) float32` | **inlier 만**, master 좌표계 (mm) |
| `elapsed_ms` | `dict` | `preproc / descriptor_scanned / descriptor_master / lightglue / ransac / total` |

`T` 사용 예:

```python
import numpy as np
# scanned 좌표계의 한 점을 master 좌표계로 보내기
p_scanned = np.array([10.0, 20.0, 150.0, 1.0])   # (x, y, z, 1), mm
p_master  = result["T"] @ p_scanned              # (x', y', z', 1), mm
```

### 3.2 `preload_model(descriptor, device="cuda")`

- 해당 descriptor 의 LightGlue ckpt 를 로드해 GPU 에 상주시킨다.
- **한 번에 하나만 상주**: 이미 다른 descriptor 가 로드돼 있으면 자동으로 언로드 후 교체.
- 첫 `register_pair` 호출의 지연(약 1~2초)을 앱 시작 시간으로 옮기는 용도.

### 3.3 `unload_model(descriptor=None)`

- 현재 상주한 모델 해제 + `torch.cuda.empty_cache()`.
- 다른 CUDA 앱에 GPU 메모리를 양보해야 할 때 호출.

---

## 4. 전형적 GUI 흐름

아래 의사코드는 프레임워크 중립이다. PyQt / Tkinter / CLI 모두 동일 구조.

```python
# [앱 기동]
from depth_registration import preload_model, register_pair, unload_model
preload_model("fpfh", device="cuda")

# [스캔 완료 버튼 클릭]
def on_register_clicked(scanned_path: str):
    r = register_pair(scanned_path, descriptor="fpfh")
    if not r["success"]:
        show_warning(f"정합 실패 (inlier {r['num_inliers']}/{r['num_matches']})")
        return
    apply_transform(r["T"])                  # 3D 뷰어에 반영
    status_bar(f"{r['num_inliers']} inliers, "
               f"{r['elapsed_ms']['total']:.0f} ms")

# [앱 종료]
unload_model()
```

**실행 가능한 전체 예제**: `../examples/gui_integration_demo.py`

### 4.1 스레드 주의

- `register_pair` 는 **CPU + GPU 혼합 연산**이다. PyQt 메인 스레드에서 바로 호출하면
  UI 가 수초간 멈춘다. `QThread` / `concurrent.futures.ThreadPoolExecutor` 로 넘기는 걸 권장.
- `preload_model` / `register_pair` / `unload_model` 은 **동일 스레드에서** 호출할 것.
  LightGlue 모델이 GPU 에 상주해 있고 torch 는 thread-local CUDA stream 을 쓴다.

### 4.2 반복 호출 시 지연

- **첫 호출**은 LightGlue 모델 로드 + (SHOT 의 경우) PCL binding JIT 로 인해 느리다.
  `preload_model` 로 LightGlue 는 미리, master 캐시는 처음 한 번 만들어두면 됨.
- **2번째 이후**는 master 캐시 hit → `descriptor_master` 가 수십 ms 로 떨어진다.

---

## 5. descriptor 선택 가이드 (FPFH vs SHOT)

| 기준 | FPFH | SHOT |
|---|---|---|
| 속도 | **빠름** (~5-6s) | 느림 (~10s, 첫 호출) |
| 정확도 | 양호 | **더 높은 inlier_ratio** (self-match 기준 90%+ vs 75%+) |
| 설치 복잡도 | 기본 환경으로 즉시 동작 | PCL 1.15 + pybind 바인딩 필요 (Windows) |
| 권장 기본값 | ✅ | 성능이 부족할 때 전환 |

**실무 권장**: FPFH 로 시작하고, 특정 scanned 가 반복적으로 `success=False` 나면
그때 SHOT 으로 재시도하는 2단계 폴백. 단, 한 런타임에서 `preload_model` 을 자주
스위칭하면 로드 오버헤드(+1-2s)가 붙으니 프로파일에 맞게 고정 권장.

---

## 6. 트러블슈팅 (진단 플로우)

### 6.1 `success=False` / `inlier_ratio` 낮음

```
num_matches < 10?
├─ YES → scanned 가 master 와 다른 대상일 가능성 높음.
│        • scanned PNG 를 matplotlib 로 열어 실제 내용 확인
│        • 완전히 다른 물체를 찍었는지, 뒤집혔는지 육안 확인
│
└─ NO (수십~백+ matches) → 매칭은 됐지만 RANSAC 이 깨짐
         • scanned 전처리에서 floor 가 제대로 제거되지 않았을 가능성
           → mask_scanned_table 의 band 가 테이블 면을 놓쳤는지 확인
         • depth 에 심한 노이즈 → bilateral σ 이 안 맞음
         • 아주 큰 회전 (>90°): v1 학습 분포 밖
```

### 6.2 `ImportError: DLL load failed` (Windows, SHOT)

`environment/setup_windows.md` §3 (PCL 설치), §6 Phase 1 (import 검증) 참조.

### 6.3 `FileNotFoundError: ... iss_shot_v1_dim352_0417.tar`

SHOT ckpt (256MB) 는 repo 에 포함되지 않는다. 담당자에게서 별도 전달받아
`checkpoints/iss_shot_v1_dim352_0417.tar` 에 복사.

### 6.4 `CUDA out of memory`

```python
# 옵션 1: 다른 descriptor 가 상주 중일 수 있음 → 해제
unload_model()

# 옵션 2: CPU 로 폴백 (속도 저하 큼)
preload_model("fpfh", device="cpu")
result = register_pair(scan, descriptor="fpfh", device="cpu")
```

### 6.5 첫 호출만 극단적으로 느림 (>30s)

- LightGlue 모델이 cold 로드되는 중. `preload_model` 을 앱 기동 직후에 호출.
- (SHOT) PCL binding 의 첫 JIT 컴파일. 2번째 호출부터는 정상.

### 6.6 같은 scanned 인데 결과 T 가 매번 다름

- FPFH/SHOT 자체는 결정적이지만, 내부 ISS 검출에 부동소수점 tie-break 가 있어
  drone-fine 수준의 keypoint 순서 차이가 발생할 수 있다.
- `T` 는 mm 기준 sub-mm 오차 내에서 일관되어야 정상. 큰 변동 (>1mm) 이면 입력 자체가
  달라진 것으로 의심.

---

## 7. 캐시 관리

### 7.1 어디에 있나

- 기본 위치: `<shared>/cache/master_<img8>_<params8>_<desc>.npz`
- `register_pair(..., cache_dir=Path(...))` 로 오버라이드 가능

### 7.2 언제 재계산되나

캐시 키는 `SHA1(image bytes, descriptor, params)` 중 일부다. 즉:

- master PNG 가 바뀌면 → 자동 재계산
- descriptor 가 `"fpfh"` ↔ `"shot"` 바뀌면 → 각각 별개 캐시
- params (voxel, normal_r, fpfh_r 등) 는 고정이라 손댈 일 없음

### 7.3 수동 정리

```bash
rm cache/*.npz        # 다음 호출에서 자동 재생성
```

---

## 8. 성능 기준 (참고치)

Linux, RTX 3090, sample_scanned.png 기준:

| descriptor | total | preproc | desc(scan) | desc(master, cache hit) | LightGlue | RANSAC |
|---|---|---|---|---|---|---|
| FPFH | ~5.7 s | 660 ms | 3.7 s | 180 ms | 1.0 s | 150 ms |
| SHOT (1st) | ~10.3 s | 830 ms | 5.2 s | 250 ms | 3.9 s | 170 ms |
| SHOT (2nd+) | ~4.3 s | 830 ms | 5.2 s | 10 ms | 200 ms | 170 ms |

GPU VRAM 상주: FPFH ~1 MB, SHOT ~95 MB (descriptor_dim 차이).

---

## 9. 자주 묻는 질문

**Q. master 를 여러 개 쓰고 싶다 (부품마다 다른 master).**
A. `register_pair(scan, master=<path>)` 로 매번 명시. 각 master 는 첫 호출에 캐시
빌드(수 초) 후 이후는 hit. 여러 master 를 번갈아 써도 캐시는 공존.

**Q. 결과 `T` 를 역변환하고 싶다.**
A. `T_master_to_scanned = np.linalg.inv(result["T"])`. rigid transform 이므로 역행렬이
안전 (수치적으로도 안정).

**Q. `matches_*_xyz` 배열은 어디에 쓰나?**
A. 디버깅·시각화용. inlier 쌍만 필터되어 들어있어 quality 검증이나 overlay 렌더링에 유용.

**Q. ransac_iter/inlier_th 를 조정해도 되나?**
A. 가능. 다만 `inlier_th` 는 mm 단위라 5.0 mm 가 학습 설정에 튜닝된 값. 크게 올리면
false positive 증가, 내리면 success=False 빈발. 극단적으로 바꾸지 말 것.

**Q. scanned PNG 해상도가 다르면?**
A. 3008×2432 가 아니면 LightGlue 내부 normalize_keypoints 가 out-of-distribution
좌표를 받게 되어 성능이 저하될 수 있음. 가능하면 센서 쪽에서 3008×2432 로 맞추길.

---

## 10. 관련 문서

- `../README.md` — 개발자용 요약 (파이프라인 내부 파라미터, warning)
- `STRUCTURE.md` — 코드 구조 지도 (어느 파일이 어떤 역할, 데이터 흐름, "이거 어디 고치지?" 역인덱스)
- `../environment/setup_windows.md` — Windows 설치 + 3-phase 수치 검증
- `../examples/run_cli.py` — CLI 스모크
- `../examples/gui_integration_demo.py` — GUI 통합 참고 코드
- `../examples/verify_shot_phase3.py` — Linux/Windows SHOT 수치 일치 검증
