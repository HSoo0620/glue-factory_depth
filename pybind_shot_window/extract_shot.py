"""
SHOT352 Descriptor Extractor for Scanned Z-map Images
Uses pybind11 shot_module (PCL backend, no OpenNI2 dependency)

v5 Parameters:
  lateral  = 0.056 mm/px
  transport= 0.056 mm/px
  vertical = 0.0085 mm/raw
  voxel    = 5.0 mm
  normalR  = 25.0 mm
  shotR    = 50.0 mm
"""

import os
# PCL DLL directories must be added BEFORE importing shot_module
os.add_dll_directory(r"C:\Program Files\PCL 1.15.1\bin")
os.add_dll_directory(r"C:\Program Files\PCL 1.15.1\3rdParty\FLANN\bin")
os.add_dll_directory(r"C:\Program Files\PCL 1.15.1\3rdParty\VTK\bin")

import numpy as np
from PIL import Image
from pathlib import Path
import struct
import time

import shot_module


# ---- v5 Parameters ----
LATERAL_MM   = 0.056
TRANSPORT_MM = 0.056
VERTICAL_MM  = 0.0085
VOXEL_SIZE   = 5.0
NORMAL_R     = 25.0
SHOT_R       = 50.0

SCANNED_DIR = Path(__file__).parent.parent / "scanned"
OUTPUT_DIR  = Path(__file__).parent.parent / "output_shot_scanned_v5"


def load_zmap_to_cloud(image_path: Path) -> np.ndarray:
    """Load 16-bit depth PNG -> point cloud (N, 3) float32 in mm."""
    img = np.array(Image.open(image_path))

    # 8-bit -> scale to 16-bit range (same as stbi_load_16)
    if img.dtype == np.uint8:
        img = img.astype(np.uint16) * 257

    print(f"  Image: {img.shape[1]} x {img.shape[0]}, "
          f"dtype={img.dtype}, range=[{img[img > 0].min()}, {img.max()}]")

    mask = img > 0
    ys, xs = np.where(mask)
    zs = img[mask].astype(np.float32)

    points = np.column_stack([
        xs.astype(np.float32) * LATERAL_MM,
        ys.astype(np.float32) * TRANSPORT_MM,
        zs * VERTICAL_MM,
    ]).astype(np.float32)

    print(f"  Valid pixels: {len(points)}")
    print(f"  Physical size: {img.shape[1]*LATERAL_MM:.1f} x {img.shape[0]*TRANSPORT_MM:.1f} mm")
    return points


def save_shot_binary(path: Path, points: np.ndarray, descriptors: np.ndarray):
    """Save in same binary format as C++ version.
    Format: [numPoints:u32][descDim:u32] then per point: [x,y,z:f32][desc[352]:f32]
    """
    n = len(points)
    with open(path, "wb") as f:
        f.write(struct.pack("II", n, 352))
        for i in range(n):
            f.write(struct.pack("fff", points[i, 0], points[i, 1], points[i, 2]))
            f.write(descriptors[i].tobytes())
    print(f"  Saved: {path} ({path.stat().st_size / 1024:.0f} KB)")


def process_image(image_path: Path, output_dir: Path):
    """Process a single z-map image."""
    stem = image_path.stem
    print(f"\n{'='*50}")
    print(f" Processing: {image_path.name}")
    print(f"{'='*50}")

    t0 = time.time()

    # 1. Load image -> point cloud (mm)
    points = load_zmap_to_cloud(image_path)

    # 2. SHOT descriptor extraction (via PCL pybind11 module)
    result = shot_module.extract_shot(
        points,
        voxel_size=VOXEL_SIZE,
        normal_radius=NORMAL_R,
        shot_radius=SHOT_R,
    )

    out_points = result["points"]       # (M, 3)
    out_desc   = result["descriptors"]  # (M, 352)

    # 3. Save binary
    output_dir.mkdir(parents=True, exist_ok=True)
    bin_path = output_dir / f"{stem}_shot352.bin"
    save_shot_binary(bin_path, out_points, out_desc)

    # 4. Save points as npy (bonus, easy to load in Python later)
    npy_path = output_dir / f"{stem}_points.npy"
    np.save(npy_path, out_points)
    desc_npy_path = output_dir / f"{stem}_desc.npy"
    np.save(desc_npy_path, out_desc)
    print(f"  Saved: {npy_path}")
    print(f"  Saved: {desc_npy_path}")

    elapsed = time.time() - t0
    print(f"\n  Stats: {result['num_input']} -> {result['num_downsampled']} "
          f"-> {result['num_keypoints']} keypoints, "
          f"{result['num_valid_desc']} valid descriptors")
    print(f"  Time: {elapsed:.1f}s")

    return result


def main():
    print("=" * 50)
    print(" SHOT352 Extractor (Python + PCL pybind11, v5)")
    print("=" * 50)
    print(f" Input : {SCANNED_DIR}")
    print(f" Output: {OUTPUT_DIR}")
    print(f" Params: voxel={VOXEL_SIZE}, normalR={NORMAL_R}, shotR={SHOT_R}")

    images = sorted(SCANNED_DIR.glob("*.png"))
    print(f"\n Found {len(images)} image(s).")

    total_start = time.time()
    for i, img_path in enumerate(images):
        print(f"\n[{i+1}/{len(images)}]", end="")
        process_image(img_path, OUTPUT_DIR)

    total_elapsed = time.time() - total_start
    print(f"\n{'='*50}")
    print(f" All done! {len(images)} images in {total_elapsed:.1f}s")
    print(f" Output: {OUTPUT_DIR}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
