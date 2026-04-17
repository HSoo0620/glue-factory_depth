#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BUILD_DIR="$SCRIPT_DIR/build"

echo "============================================"
echo " Building shot_module (pybind11 + PCL)"
echo "============================================"

# ---- Check dependencies ----
echo ""
echo "[1/4] Checking dependencies..."

python3 -c "import pybind11; print(f'  pybind11: {pybind11.__version__}')" || {
    echo "  pybind11 not found. Installing..."
    pip install pybind11
}

python3 -c "import numpy; print(f'  numpy: {numpy.__version__}')"
python3 -c "from PIL import Image; import PIL; print(f'  Pillow: {PIL.__version__}')"

# Check PCL
if pkg-config --exists pcl_features-1.15 2>/dev/null; then
    echo "  PCL: $(pkg-config --modversion pcl_features-1.15)"
    PCL_VER="1.15"
elif pkg-config --exists pcl_features-1.14 2>/dev/null; then
    echo "  PCL: $(pkg-config --modversion pcl_features-1.14)"
    PCL_VER="1.14"
elif pkg-config --exists pcl_features-1.13 2>/dev/null; then
    echo "  PCL: $(pkg-config --modversion pcl_features-1.13)"
    PCL_VER="1.13"
elif pkg-config --exists pcl_features-1.11 2>/dev/null; then
    echo "  PCL: $(pkg-config --modversion pcl_features-1.11)"
    PCL_VER="1.11"
elif dpkg -l | grep -q libpcl-dev 2>/dev/null; then
    echo "  PCL: installed (via apt)"
else
    echo "  PCL not found! Install with:"
    echo "    sudo apt install libpcl-dev"
    echo "    # or: conda install -c conda-forge pcl"
    exit 1
fi

# ---- CMake build ----
echo ""
echo "[2/4] Configuring (CMake)..."
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

cmake "$SCRIPT_DIR" \
    -DCMAKE_BUILD_TYPE=Release \
    -Dpybind11_DIR="$(python3 -c 'import pybind11; print(pybind11.get_cmake_dir())')"

echo ""
echo "[3/4] Building..."
cmake --build . --config Release -j$(nproc)

# ---- Copy result ----
echo ""
echo "[4/4] Installing..."
EXT_SUFFIX=$(python3 -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))")
SO_FILE=$(find "$BUILD_DIR" -name "shot_module*${EXT_SUFFIX}" -o -name "shot_module*.so" | head -1)

if [ -n "$SO_FILE" ]; then
    cp "$SO_FILE" "$SCRIPT_DIR/"
    echo ""
    echo "============================================"
    echo " BUILD SUCCESS"
    echo " Output: $SCRIPT_DIR/$(basename $SO_FILE)"
    ls -lh "$SCRIPT_DIR/$(basename $SO_FILE)"
    echo "============================================"
else
    echo "ERROR: Build output not found!"
    exit 1
fi
