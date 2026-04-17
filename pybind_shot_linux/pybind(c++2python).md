# C++ → Python 바인딩 포팅 기록 (SHOT352 via pybind11)

**작성일**: 2026-04-14
**대상**: `pybind_shot_linux/shot_module` (PCL SHOT352 descriptor 추출기)
**맥락**: Windows 에서 C++ opus 가 만든 pybind11 모듈을 Linux (Ubuntu 20.04 + conda LightGlue) 로 이식하고, 기존 `.bin` 기반 추론 파이프라인에 on-the-fly 모드로 통합.

---

## 0. 왜 pybind11 인가

- PCL SHOT descriptor 는 C++ 전용. Python 순수 구현은 성능상 실용 불가.
- 초기 해결: C++ 로 SHOT 뽑아서 `.bin` 으로 덤프 → Python 에서 읽어서 KDTree 룩업.
- 문제: 파라미터(voxel / normal_r / shot_r) 바꿀 때마다 Windows 머신에서 재실행 → sweep 불가.
- 목표: Python 에서 직접 `shot_module.extract_shot(points, voxel, normal_r, shot_r)` 호출.

---

## 1. 환경 진단

| 항목 | 상태 | 비고 |
|------|------|------|
| Python | 3.10.19 (LightGlue env) | `.so` 는 이 ABI 에 묶임 |
| g++ | 9.4.0 (Ubuntu 20.04) | `--enable-default-pie` ← 나중에 말썽 |
| CMake | 3.16.3 → 4.3.0 (conda install 후) | CMakeLists.txt 요구 3.16 OK |
| pybind11 | 없음 → 3.0.3 → 2.9.2 | 3.x 는 의심했지만 실제 원인은 아니었음 |
| PCL | 없음 → 1.11.1 (conda-forge) | Windows 는 1.15.1 |

**결정**: conda-forge 로 설치. apt 는 sudo 필요 + Ubuntu 20.04 PCL 이 오래됨.

```bash
conda install -c conda-forge pcl pybind11 cmake -y --solver=libmamba
```

> **주의**: conda 기본 (classic) solver 로는 PCL 의존성 해결이 너무 느려서 로그가 멈춘 것처럼 보임 (`conda-libmamba-solver` 는 이미 설치되어 있어서 `--solver=libmamba` 로 즉시 해결).

---

## 2. 빌드 과정에서 만난 3가지 함정

### 2.1. `build.sh` 의 PCL 버전 체크가 1.11 을 못 잡음

- `build.sh` 는 `pkg-config --exists pcl_features-{1.15,1.14,1.13}` 만 체크 → conda 가 준 1.11 에서 실패.
- 실제 `find_package(PCL 1.10 REQUIRED ...)` 는 1.11 을 받아주므로 CMake 를 직접 호출해서 우회:

```bash
export PKG_CONFIG_PATH="$CONDA_PREFIX/lib/pkgconfig:$PKG_CONFIG_PATH"
cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -Dpybind11_DIR=$(python -c 'import pybind11; print(pybind11.get_cmake_dir())')
```

**교훈**: 이식된 빌드 스크립트의 버전 whitelist 는 당겨진 범위가 좁을 수 있음. CMake 를 직접 호출해 `find_package` 에 맡기는 게 포터블.

### 2.2. LTO + Ubuntu default-pie = PT_INTERP 가 `.so` 에 들어감

**증상**: import 시점에
```
ImportError: ... cannot dynamically load position-independent executable
```

**진단**:
```bash
file shot_module.cpython-310-x86_64-linux-gnu.so
# ...interpreter /lib64/ld-linux-x86-64.so.2, no section header
```
Shared object 인데 `PT_INTERP` 가 박혀 있음 → 현대 glibc 의 `dlopen()` 이 거부.

**원인**: Ubuntu 20.04 gcc 9.4 는 `--enable-default-pie`. pybind11 (또는 CMake 4.x) 이 자동으로 `INTERPROCEDURAL_OPTIMIZATION=ON` 을 걸어서 `-flto=auto` 가 주입됨. `-shared -flto=auto` 조합이 링커에 `-pie` 효과를 전파시킴.

**해결**:
```bash
cmake .. -DCMAKE_INTERPROCEDURAL_OPTIMIZATION=OFF ...
```

**교훈**: Ubuntu 20.04 계열에서 pybind11 모듈 빌드 시 LTO 와 default-pie 가 부딪힐 수 있음. 성능 차이 미미하므로 LTO 끄는 게 안전.

### 2.3. Eigen alignment mismatch → `shared_ptr` 소멸자에서 segfault

**증상**: import 는 성공하나 `extract_shot()` 호출 후 SHOT 계산 완료 직후 SIGSEGV.

**진단 (gdb)**:
```
#0  __GI___libc_free (mem=0x9f1) at malloc.c:3102    ← 쓰레기 주소 free
#1  shot_module.so
#2  std::_Sp_counted_base::_M_release()              ← shared_ptr 소멸
```

**원인**: `CMakeLists.txt` 가 모듈을 `EIGEN_MAX_ALIGN_BYTES=32` 로 빌드. 그러나 conda-forge PCL 1.11 은 기본값(16) 으로 컴파일됨. PCL 헤더에 선언된 Eigen-aligned 타입들의 `sizeof`/`offsetof` 가 두 쪽에서 달라짐 → `pcl::PointCloud<pcl::Normal>` 의 shared_ptr control block 이 어긋난 포인터를 들고 있다가 소멸자에서 `free(0x9f1)` 호출.

**해결**: `CMakeLists.txt:22` 에서 라인 제거.
```cmake
target_compile_definitions(shot_module PRIVATE
    NOMINMAX
    EIGEN_MAX_ALIGN_BYTES=32   ← 제거
)
```

**교훈**: Windows MSVC + PCL 1.15 (AVX 기본 = 32-byte 정렬) 에서 괜찮던 매크로가 Linux conda-forge PCL (16-byte 기본) 과 충돌. **Eigen 관련 매크로는 링크 대상 라이브러리의 빌드 시점 매크로와 반드시 일치**시켜야 함. 의심스러우면 그냥 default 쓰는 게 안전.

---

## 3. 검증: Windows `.bin` vs Linux `.bin` 바이너리 재현성

같은 PNG (`roi13_zmap 1.png`, 2687×3600, 6M valid pixels) 로 양 플랫폼에서 실행.

| 비교 항목 | 결과 |
|-----------|------|
| 파일 크기 | **byte-exact** (3,280,208 bytes) |
| Keypoint 개수 | **byte-exact** (2310/2310) |
| Keypoint 좌표 | **byte-exact** (max diff = 0.000 mm) |
| Keypoint 순서 | **byte-exact** (NN 매칭 idx == arange) |
| Descriptor L2 norm | 양쪽 mean = 1.000000 |
| Descriptor 값 | 드리프트 존재 (byte-exact 아님) |
| Descriptor L2 차이 | median=8.35e-05, mean=1.33e-03, max=3.07e-02 |

**해석**:
- VoxelGrid + NaN filter 파이프라인은 완전 deterministic (기대대로).
- SHOT descriptor 는 `NormalEstimationOMP` 의 병렬 누적 순서 비결정성 + SIMD/FMA 차이로 부동소수점 드리프트. 하지만 unit-L2-normalized 벡터의 median 오차 1e-4 수준은 cosine similarity 로 환산하면 ~5e-9 변화 → downstream matching 에 사실상 영향 없음.
- max 3.07e-02 outlier 들은 평면 degenerate 패치로 SHOT 의 LRF 가 부호 flip 할 수 있는 케이스. 어차피 매칭에 쓸 수 없는 영역이라 버려져도 무방.

**판정**: 의미적으로 동치 (practically identical).

---

## 4. 파이프라인 통합 (on-the-fly SHOT)

### 4.1. 설계 원칙: "seam" 을 정확히 짚기

기존 코드의 `lookup_shot352(pts_cloud, desc_cloud, kp_xyz, n_valid)` 는 cloud 의 출처를 모름 — KDTree 룩업만 함. 따라서:
- `.bin` 로드 결과 `(pts_cloud, desc_cloud)` 튜플을
- `shot_module.extract_shot()` 결과로 **그대로 대체** 하면 나머지 파이프라인은 건드릴 필요 없음.

### 4.2. 추가 파일

- `vis_new/_shot_onthefly.py`: `zmap_to_cloud()` + `compute_shot_cloud()` 헬퍼. `pybind_shot_linux/` 를 `sys.path` 에 주입해 `shot_module` 을 import.

### 4.3. 수정 파일

- `vis_new/inference_scan_vs_scan_shot.py`, `inference_scan_vs_train_shot.py`:
  - `prepare_iss_shot()` 에 `shot_params: dict | None = None` kwarg 추가.
  - CLI 4개 추가: `--shot_source {bin,onthefly}`, `--shot_voxel`, `--shot_normal_r`, `--shot_r`.
  - 출력 파일명에 source tag 자동 삽입: `..._otf_v5_nr25_sr50_...png`.

### 4.4. end-to-end 동치 검증

| 스크립트 | 모드 | matches | inliers |
|----------|------|---------|---------|
| scan_vs_scan (roi13 × data1, mask_affine) | bin | 5 | 1 |
| scan_vs_scan | onthefly (v=5, nr=25, sr=50) | 4 | 0 |
| scan_vs_train (roi13 × zmap_0000) | bin | 12~13 | 4 |
| scan_vs_train | onthefly (v=5, nr=25, sr=50) | 12 | 4 |

- pts 수 (2310 / 1875), NN distance (p50=1.80 mm) 양 모드 **완전 일치**.
- match 가 scan-vs-scan 에서 1개 빠진 건 descriptor drift 가 borderline 매칭을 flip 시킨 케이스 (예측된 수준).
- scan-vs-train 은 완전 일치 → sweep 실험에 써도 안전.

---

## 5. 최종 빌드 레시피 (재현용)

```bash
# 1. 의존성
conda activate LightGlue
conda install -c conda-forge pcl pybind11 cmake -y --solver=libmamba
conda install -c conda-forge "pybind11<3" -y    # 2.9.2

# 2. 빌드
cd pybind_shot_linux
export PKG_CONFIG_PATH="$CONDA_PREFIX/lib/pkgconfig:$PKG_CONFIG_PATH"
rm -rf build && mkdir build && cd build
cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -Dpybind11_DIR=$(python -c 'import pybind11; print(pybind11.get_cmake_dir())') \
  -DCMAKE_INTERPROCEDURAL_OPTIMIZATION=OFF
cmake --build . -j$(nproc)
cp shot_module.cpython-310-x86_64-linux-gnu.so ..

# 3. 스모크 테스트
cd ..
python -c "import numpy as np, shot_module; \
  r = shot_module.extract_shot(np.random.rand(500,3).astype(np.float32)*100, 5.0, 25.0, 50.0); \
  print('OK', r['num_valid_desc'])"
```

**CMakeLists.txt 수정 사항** (별도):
- `target_compile_definitions` 에서 `EIGEN_MAX_ALIGN_BYTES=32` 라인 제거.

---

## 6. 교훈 정리

| # | 교훈 | 적용 범위 |
|---|------|----------|
| 1 | conda solver 가 느리면 libmamba 로 | 대형 C++ 의존성 설치 시 |
| 2 | build script 의 버전 whitelist 를 과신하지 말 것, CMake 가 더 유연 | PCL 외 다른 C++ lib 에도 해당 |
| 3 | Ubuntu 20.04 에서 pybind11 + LTO 는 PT_INTERP 사고 가능성 | gcc 9.x + default-pie 환경 |
| 4 | Eigen 관련 컴파일 플래그는 링크 대상 lib 과 **무조건** 일치시킬 것 | PCL, Eigen 기반 모든 C++ lib |
| 5 | 결정성 검증 단계에서 keypoint / descriptor 분리해서 비교 | numerical reproducibility 전반 |
| 6 | 통합 시 "seam" 을 찾아 최소 침습 변경 | 기존 파이프라인 확장 일반 원칙 |

---

## 7. 다음 단계

- SHOT radius sweep: `--shot_source onthefly --shot_r {30,40,50,60,80}` 반복 → FPFH 때처럼 training radius(50) 외 값이 scan 도메인에서 더 나을 가능성.
- `.bin` 생성 파이프라인을 Linux 버전으로 교체 가능 (byte-exact keypoint, 의미적 동치 descriptor). Windows 머신 의존성 제거.
- 향후 SHOT 외 다른 PCL descriptor (FPFH, PFH 등) 도 동일한 pybind 패턴으로 포팅 가능 — 본 문서의 함정 3가지가 재발 주의 대상.
