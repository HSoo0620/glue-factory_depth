# pybind11 SHOT352 Descriptor 모듈 구축 가이드

## 개요

PCL의 SHOT352 descriptor 추출을 Python에서 사용할 수 있도록 pybind11로 C++ 바인딩한 모듈.

```
Python (이미지 로딩, 좌표변환)
  ↓ numpy array (N, 3) float32
pybind11 shot_module (C++ PCL)
  ↓ VoxelGrid → Normal → SHOT352
Python (결과 numpy array 수신, 저장)
```

**핵심 포인트**: `pcl_io`를 링크하지 않으므로 OpenNI2 의존성 없음.

---

## 환경 요구사항

| 항목 | 로컬 검증 버전 | 비고 |
|------|---------------|------|
| OS | Windows 11 | Linux도 동일 구조 (빌드 명령만 다름) |
| Python | 3.13 | 3.8+ 가능 |
| PCL | 1.15.1 | include/lib 경로 필요 |
| Eigen3 | PCL 번들 | PCL 3rdParty에 포함 |
| Boost | 1.87 (PCL 번들) | PCL 3rdParty에 포함 |
| FLANN | PCL 번들 | PCL 3rdParty에 포함 |
| pybind11 | 3.0.3 | `pip install pybind11` |
| numpy | 2.3+ | 이미지→포인트클라우드 변환 |
| Pillow | 11.3+ | 16-bit PNG 로딩 |
| MSVC | v143 (VS 2022) | Windows 빌드 시. Linux는 g++ |

---

## 파일 구조

```
pybind_shot/
├── shot_module.cpp          # pybind11 C++ 모듈 소스
├── setup.py                 # setuptools 빌드 (Linux용)
├── build.bat                # Windows 직접 빌드 스크립트
├── extract_shot.py          # Python 실행 스크립트
└── shot_module.cp313-win_amd64.pyd  # 빌드 결과물
```

---

## 빌드 방법

### Windows (cl.exe 직접 컴파일)

setuptools가 VS를 못 찾는 경우가 많으므로, `vcvarsall.bat` + `cl.exe` 직접 호출이 안정적.

```bat
@echo off
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvarsall.bat" x64

cl.exe /O2 /EHsc /std:c++17 /openmp /MD ^
    /DNOMINMAX /D_CRT_SECURE_NO_WARNINGS /DEIGEN_MAX_ALIGN_BYTES=32 ^
    /I"<PYTHON_INCLUDE>" /I"<PYBIND11_INCLUDE>" ^
    /I"<PCL>/include/pcl-1.15" ^
    /I"<PCL>/3rdParty/Eigen3/include/eigen3" ^
    /I"<PCL>/3rdParty/Boost/include/boost-1_87" ^
    /I"<PCL>/3rdParty/FLANN/include" ^
    /LD shot_module.cpp ^
    /Fe:shot_module.cp313-win_amd64.pyd ^
    /link ^
    /LIBPATH:"<PYTHON_LIBS>" ^
    /LIBPATH:"<PCL>/lib" ^
    /LIBPATH:"<PCL>/3rdParty/Boost/lib" ^
    /LIBPATH:"<PCL>/3rdParty/FLANN/lib" ^
    python313.lib pcl_common.lib pcl_features.lib pcl_filters.lib ^
    pcl_kdtree.lib pcl_search.lib pcl_octree.lib flann_cpp.lib
```

**경로 확인 명령**:
```bash
# Python include/lib 경로
python -c "import sysconfig; print(sysconfig.get_path('include')); print(sysconfig.get_config_var('LIBDIR'))"

# pybind11 include 경로
python -c "import pybind11; print(pybind11.get_include())"

# 빌드 결과물 확장자
python -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))"
# 예: .cp313-win_amd64.pyd (Windows), .cpython-313-x86_64-linux-gnu.so (Linux)
```

### Linux (g++ 직접 컴파일)

```bash
PYTHON_INC=$(python3 -c "import sysconfig; print(sysconfig.get_path('include'))")
PYBIND_INC=$(python3 -c "import pybind11; print(pybind11.get_include())")
EXT_SUFFIX=$(python3 -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))")
PCL_PREFIX=/usr  # or wherever PCL is installed

g++ -O2 -std=c++17 -shared -fPIC -fopenmp \
    -DNOMINMAX -DEIGEN_MAX_ALIGN_BYTES=32 \
    -I"$PYTHON_INC" -I"$PYBIND_INC" \
    $(pkg-config --cflags pcl_features-1.15 pcl_filters-1.15) \
    shot_module.cpp \
    -o shot_module${EXT_SUFFIX} \
    $(pkg-config --libs pcl_features-1.15 pcl_filters-1.15) \
    -lpcl_common -lpcl_features -lpcl_filters \
    -lpcl_kdtree -lpcl_search -lpcl_octree -lflann_cpp
```

`pkg-config`이 없으면 수동 지정:
```bash
g++ -O2 -std=c++17 -shared -fPIC -fopenmp \
    -DNOMINMAX -DEIGEN_MAX_ALIGN_BYTES=32 \
    -I"$PYTHON_INC" -I"$PYBIND_INC" \
    -I/usr/include/pcl-1.15 \
    -I/usr/include/eigen3 \
    shot_module.cpp \
    -o shot_module${EXT_SUFFIX} \
    -lpcl_common -lpcl_features -lpcl_filters \
    -lpcl_kdtree -lpcl_search -lpcl_octree -lflann_cpp
```

### setup.py (Linux에서 잘 동작)

```bash
pip install pybind11
python setup.py build_ext --inplace
```

> Windows에서는 setuptools가 VS를 못 찾는 경우가 많아 `build.bat` 사용 권장.

---

## 링크 라이브러리 (7개만)

| 라이브러리 | 용도 |
|-----------|------|
| `pcl_common` | 기본 자료구조 (PointCloud, PointXYZ) |
| `pcl_features` | SHOT352, Normal estimation |
| `pcl_filters` | VoxelGrid downsampling |
| `pcl_kdtree` | KdTree (검색 구조) |
| `pcl_search` | 검색 인터페이스 |
| `pcl_octree` | Octree (pcl_search 의존) |
| `flann_cpp` | FLANN (KdTree 백엔드) |

**의도적으로 제외한 것**: `pcl_io`, `pcl_io_ply` → 이것들이 OpenNI2, VTK에 의존하므로 제외. PCD 파일 저장이 필요하면 Python 측에서 처리.

---

## Python 사용법

### 기본 사용

```python
import os
# Windows: DLL 경로 등록 (import 전에 반드시 호출)
os.add_dll_directory(r"C:\Program Files\PCL 1.15.1\bin")
os.add_dll_directory(r"C:\Program Files\PCL 1.15.1\3rdParty\FLANN\bin")
os.add_dll_directory(r"C:\Program Files\PCL 1.15.1\3rdParty\VTK\bin")
# Linux에서는 LD_LIBRARY_PATH로 대체하거나 불필요

import numpy as np
from PIL import Image
import shot_module

# 1. 이미지 로딩 → 포인트 클라우드 (mm 단위)
img = np.array(Image.open("scanned_data1.png"))  # uint16
mask = img > 0
ys, xs = np.where(mask)
points = np.column_stack([
    xs.astype(np.float32) * 0.056,    # lateral mm/px
    ys.astype(np.float32) * 0.056,    # transport mm/px
    img[mask].astype(np.float32) * 0.0085  # vertical mm/raw
]).astype(np.float32)

# 2. SHOT descriptor 추출
result = shot_module.extract_shot(
    points,
    voxel_size=5.0,      # mm
    normal_radius=25.0,   # mm
    shot_radius=50.0      # mm
)

# 3. 결과
pts  = result["points"]       # (M, 3) float32 - 다운샘플된 keypoint 좌표
desc = result["descriptors"]  # (M, 352) float32 - SHOT352 descriptor
print(f"Keypoints: {result['num_keypoints']}")
print(f"Valid descriptors: {result['num_valid_desc']}")
```

### API 명세

```python
shot_module.extract_shot(
    points: np.ndarray,      # (N, 3) float32, mm 단위 포인트 클라우드
    voxel_size: float = 5.0,     # VoxelGrid leaf size (mm)
    normal_radius: float = 25.0, # Normal estimation search radius (mm)
    shot_radius: float = 50.0    # SHOT descriptor search radius (mm)
) -> dict
```

**반환값 dict**:

| 키 | 타입 | 설명 |
|----|------|------|
| `points` | `np.ndarray (M, 3)` | 다운샘플+NaN제거 후 keypoint 좌표 |
| `descriptors` | `np.ndarray (M, 352)` | SHOT352 descriptor |
| `num_input` | `int` | 입력 포인트 수 |
| `num_downsampled` | `int` | VoxelGrid 후 포인트 수 |
| `num_keypoints` | `int` | NaN normal 제거 후 최종 keypoint 수 |
| `num_valid_desc` | `int` | 유효한 descriptor 수 |

---

## v5 파라미터 (검증 완료)

| 파라미터 | 값 | 단위 | 비고 |
|---------|-----|------|------|
| lateralMm | 0.056 | mm/pixel | 픽셀당 가로 간격 |
| transportMm | 0.056 | mm/pixel | 픽셀당 세로 간격 |
| verticalMm | 0.0085 | mm/raw | raw값 1당 높이 간격 |
| voxelSize | 5.0 | mm | 다운샘플 leaf size |
| normalRadius | 25.0 | mm | 법선 추정 반경 |
| shotRadius | 50.0 | mm | SHOT descriptor 반경 |

비율: `voxel : normal : shot = 1 : 5 : 10`

---

## 출력 바이너리 포맷 (`*_shot352.bin`)

C++ 버전과 동일한 포맷:

```
[4 bytes] uint32 numPoints (M)
[4 bytes] uint32 descDim   (352)
-- per point (M회 반복) --
[4 bytes] float32 x
[4 bytes] float32 y
[4 bytes] float32 z
[1408 bytes] float32 × 352  (descriptor)
```

Python에서 읽기:
```python
import struct, numpy as np

with open("result_shot352.bin", "rb") as f:
    n, dim = struct.unpack("II", f.read(8))
    points = np.zeros((n, 3), dtype=np.float32)
    descs  = np.zeros((n, dim), dtype=np.float32)
    for i in range(n):
        points[i] = struct.unpack("fff", f.read(12))
        descs[i]  = struct.unpack(f"{dim}f", f.read(dim * 4))
```

또는 `.npy` 파일 사용 (더 간편):
```python
points = np.load("result_points.npy")      # (M, 3)
descs  = np.load("result_desc.npy")        # (M, 352)
```

---

## 검증 결과 (2026-04-14)

| 이미지 | 크기 | 원본 포인트 | 다운샘플 | 유효 descriptor | 시간 |
|--------|------|-----------|---------|---------------|------|
| roi13_zmap 1.png | 2687×3600 | 6,052,818 | 2,310 | 2,310 (100%) | 0.7s |
| scanned_data1.png | 2687×3515 | 6,099,912 | 1,875 | 1,875 (100%) | 0.7s |

---

## 트러블슈팅

### Windows: `ImportError: DLL load failed`

Python 3.8+에서 DLL 검색 경로가 변경됨. `import shot_module` 전에 반드시:
```python
import os
os.add_dll_directory(r"C:\Program Files\PCL 1.15.1\bin")
os.add_dll_directory(r"C:\Program Files\PCL 1.15.1\3rdParty\FLANN\bin")
```

### Windows: setuptools가 VS를 못 찾음

`vswhere.exe`가 PATH에 없으면 발생. `build.bat`으로 `cl.exe` 직접 호출하는 것이 안정적.

### Linux: PCL 설치

```bash
# Ubuntu/Debian
sudo apt install libpcl-dev

# 또는 conda
conda install -c conda-forge pcl
```

### 빌드 결과물 확장자 확인

Python 버전/OS에 따라 `.pyd` (Windows) 또는 `.so` (Linux) 확장자가 다름:
```bash
python -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))"
```
`/Fe:` (Windows) 또는 `-o` (Linux) 옵션에 이 확장자 사용.
