# SHOT352 pybind11 모듈 (Linux 서버용)

## Quick Start

```bash
# 1. 의존성 설치
sudo apt install libpcl-dev cmake g++    # PCL + 빌드 도구
pip install pybind11 numpy Pillow        # Python 패키지

# 2. 빌드 (둘 중 하나 선택)
# 방법 A: build.sh (CMake)
chmod +x build.sh
./build.sh

# 방법 B: setup.py
pip install -e .

# 3. 실행
python extract_shot.py -i /path/to/scanned -o /path/to/output
```

## 파일 구조

```
pybind_shot_linux/
├── shot_module.cpp      # pybind11 C++ 모듈 (PCL SHOT352)
├── CMakeLists.txt       # CMake 빌드 설정
├── build.sh             # 원클릭 빌드 스크립트
├── setup.py             # pip install 빌드 (대안)
├── extract_shot.py      # Python 실행 스크립트
└── README.md
```

## 빌드 방법 상세

### 방법 A: build.sh (권장)

```bash
./build.sh
```

내부적으로:
1. pybind11, numpy, Pillow, PCL 존재 확인
2. CMake configure + build
3. `.so` 파일을 현재 디렉토리에 복사

### 방법 B: setup.py

```bash
pip install -e .
# 또는
python setup.py build_ext --inplace
```

`pkg-config`로 PCL 경로를 자동 탐색. 없으면 fallback 경로 사용.

### 방법 C: g++ 직접 컴파일

```bash
PYBIND_INC=$(python3 -c "import pybind11; print(pybind11.get_include())")
PYTHON_INC=$(python3 -c "import sysconfig; print(sysconfig.get_path('include'))")
EXT=$(python3 -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))")

g++ -O2 -std=c++17 -shared -fPIC -fopenmp \
    -DNOMINMAX -DEIGEN_MAX_ALIGN_BYTES=32 \
    -I"$PYTHON_INC" -I"$PYBIND_INC" \
    $(pkg-config --cflags pcl_features-1.14) \
    shot_module.cpp \
    -o shot_module${EXT} \
    $(pkg-config --libs pcl_features-1.14 pcl_filters-1.14 pcl_search-1.14) \
    -lflann_cpp
```

> PCL 버전에 따라 `1.14`를 `1.13`, `1.12` 등으로 변경.

## 사용법

### CLI

```bash
# 디렉토리 내 모든 PNG 처리
python extract_shot.py -i ./scanned -o ./output

# 단일 파일
python extract_shot.py -i ./scanned/data1.png -o ./output

# 파라미터 커스텀
python extract_shot.py -i ./scanned -o ./output \
    --lateral 0.056 --transport 0.056 --vertical 0.0085 \
    --voxel 5.0 --normal_r 25.0 --shot_r 50.0
```

### Python API

```python
import numpy as np
from PIL import Image
import shot_module

# 이미지 → 포인트 클라우드
img = np.array(Image.open("depth.png"))
mask = img > 0
ys, xs = np.where(mask)
points = np.column_stack([
    xs.astype(np.float32) * 0.056,
    ys.astype(np.float32) * 0.056,
    img[mask].astype(np.float32) * 0.0085
]).astype(np.float32)

# SHOT 추출
result = shot_module.extract_shot(points, voxel_size=5.0, normal_radius=25.0, shot_radius=50.0)

pts  = result["points"]       # (M, 3)   keypoint 좌표
desc = result["descriptors"]  # (M, 352)  SHOT descriptor
```

### 반환값

| 키 | 타입 | 설명 |
|----|------|------|
| `points` | `ndarray (M, 3)` | keypoint 좌표 (mm) |
| `descriptors` | `ndarray (M, 352)` | SHOT352 descriptor |
| `num_input` | `int` | 입력 포인트 수 |
| `num_downsampled` | `int` | VoxelGrid 후 |
| `num_keypoints` | `int` | 최종 keypoint 수 |
| `num_valid_desc` | `int` | 유효 descriptor 수 |

## v5 파라미터

| 파라미터 | 값 | 의미 |
|---------|-----|------|
| lateral | 0.056 mm/px | x 해상도 |
| transport | 0.056 mm/px | y 해상도 |
| vertical | 0.0085 mm/raw | z 해상도 |
| voxel | 5.0 mm | 다운샘플 크기 |
| normal_r | 25.0 mm | 법선 추정 반경 |
| shot_r | 50.0 mm | SHOT 반경 |

비율: `voxel : normal : shot = 1 : 5 : 10`

## 출력 파일

| 파일 | 용도 |
|------|------|
| `*_shot352.bin` | C++ 호환 바이너리 (좌표+descriptor) |
| `*_points.npy` | keypoint 좌표, `np.load()` |
| `*_desc.npy` | SHOT descriptor, `np.load()` |

## 링크 라이브러리 (pcl_io 제외)

```
pcl_common, pcl_features, pcl_filters,
pcl_kdtree, pcl_search, pcl_octree, flann_cpp
```

`pcl_io`를 링크하지 않으므로 OpenNI2/VTK 없이도 동작.

## 트러블슈팅

### PCL 못 찾음 (CMake)

```bash
# PCL 설치 확인
dpkg -l | grep libpcl-dev
pkg-config --modversion pcl_common-1.14

# conda 환경이면
conda install -c conda-forge pcl
```

### pybind11 못 찾음 (CMake)

```bash
# CMake에 pybind11 경로 전달
cmake .. -Dpybind11_DIR=$(python3 -c "import pybind11; print(pybind11.get_cmake_dir())")
```

### Eigen 충돌

PCL 번들 Eigen과 시스템 Eigen이 충돌할 수 있음:
```bash
# 시스템 Eigen 사용 강제
cmake .. -DEIGEN3_INCLUDE_DIR=/usr/include/eigen3
```

### import 시 undefined symbol

OpenMP 링크 누락:
```bash
# g++ 빌드 시 -fopenmp 확인
# CMake는 자동 처리됨
```
