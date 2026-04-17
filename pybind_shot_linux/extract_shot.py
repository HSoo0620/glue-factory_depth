"""
SHOT352 Descriptor Extractor for Scanned Z-map Images
Cross-platform: Windows + Linux

v5 Parameters:
  lateral  = 0.056 mm/px
  transport= 0.056 mm/px
  vertical = 0.0085 mm/raw
  voxel    = 5.0 mm
  normalR  = 25.0 mm
  shotR    = 50.0 mm
"""

import os
import sys
import platform

# Windows: register PCL DLL directories before import
if platform.system() == "Windows":
    pcl_root = r"C:\Program Files\PCL 1.15.1"
    for subdir in ["bin", r"3rdParty\FLANN\bin", r"3rdParty\VTK\bin"]:
        dll_dir = os.path.join(pcl_root, subdir)
        if os.path.isdir(dll_dir):
            os.add_dll_directory(dll_dir)

import numpy as np
from PIL import Image
from pathlib import Path
import struct
import time
import argparse

import shot_module


# ---- Default v5 Parameters ----
DEFAULT_LATERAL_MM   = 0.056
DEFAULT_TRANSPORT_MM = 0.056
DEFAULT_VERTICAL_MM  = 0.0085
DEFAULT_VOXEL_SIZE   = 5.0
DEFAULT_NORMAL_R     = 25.0
DEFAULT_SHOT_R       = 50.0


def load_zmap_to_cloud(image_path, lateral_mm, transport_mm, vertical_mm):
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
        xs.astype(np.float32) * lateral_mm,
        ys.astype(np.float32) * transport_mm,
        zs * vertical_mm,
    ]).astype(np.float32)

    print(f"  Valid pixels: {len(points)}")
    print(f"  Physical size: {img.shape[1]*lateral_mm:.1f} x "
          f"{img.shape[0]*transport_mm:.1f} mm")
    return points


def save_shot_binary(path, points, descriptors):
    """Save in C++ compatible binary format.
    [numPoints:u32][descDim:u32] per point: [x,y,z:f32][desc[352]:f32]
    """
    n = len(points)
    with open(path, "wb") as f:
        f.write(struct.pack("II", n, 352))
        for i in range(n):
            f.write(struct.pack("fff", points[i, 0], points[i, 1], points[i, 2]))
            f.write(descriptors[i].tobytes())
    size_kb = os.path.getsize(path) / 1024
    print(f"  Saved: {path} ({size_kb:.0f} KB)")


def process_image(image_path, output_dir, args):
    """Process a single z-map image."""
    image_path = Path(image_path)
    output_dir = Path(output_dir)
    stem = image_path.stem

    print(f"\n{'='*50}")
    print(f" Processing: {image_path.name}")
    print(f"{'='*50}")

    t0 = time.time()

    # 1. Load image -> point cloud (mm)
    points = load_zmap_to_cloud(
        image_path, args.lateral, args.transport, args.vertical)

    # 2. SHOT descriptor extraction
    result = shot_module.extract_shot(
        points,
        voxel_size=args.voxel,
        normal_radius=args.normal_r,
        shot_radius=args.shot_r,
    )

    out_points = result["points"]       # (M, 3)
    out_desc   = result["descriptors"]  # (M, 352)

    # 3. Save
    output_dir.mkdir(parents=True, exist_ok=True)

    bin_path = output_dir / f"{stem}_shot352.bin"
    save_shot_binary(bin_path, out_points, out_desc)

    npy_pts  = output_dir / f"{stem}_points.npy"
    npy_desc = output_dir / f"{stem}_desc.npy"
    np.save(npy_pts, out_points)
    np.save(npy_desc, out_desc)
    print(f"  Saved: {npy_pts}")
    print(f"  Saved: {npy_desc}")

    elapsed = time.time() - t0
    print(f"\n  Stats: {result['num_input']} -> {result['num_downsampled']} "
          f"-> {result['num_keypoints']} keypoints, "
          f"{result['num_valid_desc']} valid descriptors")
    print(f"  Time: {elapsed:.1f}s")

    return result


def main():
    parser = argparse.ArgumentParser(
        description="SHOT352 Descriptor Extractor (PCL + pybind11)")
    parser.add_argument("--input", "-i", type=str, default=None,
                        help="Input directory or single PNG file")
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="Output directory")
    parser.add_argument("--lateral",   type=float, default=DEFAULT_LATERAL_MM)
    parser.add_argument("--transport", type=float, default=DEFAULT_TRANSPORT_MM)
    parser.add_argument("--vertical",  type=float, default=DEFAULT_VERTICAL_MM)
    parser.add_argument("--voxel",     type=float, default=DEFAULT_VOXEL_SIZE)
    parser.add_argument("--normal_r",  type=float, default=DEFAULT_NORMAL_R)
    parser.add_argument("--shot_r",    type=float, default=DEFAULT_SHOT_R)
    args = parser.parse_args()

    # Defaults relative to script location
    script_dir = Path(__file__).parent.resolve()
    if args.input is None:
        args.input = str(script_dir.parent / "scanned")
    if args.output is None:
        args.output = str(script_dir.parent / "output_shot_scanned_v5")

    input_path = Path(args.input)

    # Collect images
    if input_path.is_file():
        images = [input_path]
    elif input_path.is_dir():
        images = sorted(input_path.glob("*.png"))
    else:
        print(f"ERROR: {input_path} not found")
        sys.exit(1)

    print("=" * 50)
    print(" SHOT352 Extractor (Python + PCL pybind11, v5)")
    print("=" * 50)
    print(f" Input : {args.input}")
    print(f" Output: {args.output}")
    print(f" Params: voxel={args.voxel}, normalR={args.normal_r}, shotR={args.shot_r}")
    print(f" Resolution: lateral={args.lateral}, transport={args.transport}, "
          f"vertical={args.vertical}")
    print(f"\n Found {len(images)} image(s).")

    total_start = time.time()
    for i, img_path in enumerate(images):
        print(f"\n[{i+1}/{len(images)}]", end="")
        process_image(img_path, args.output, args)

    total_elapsed = time.time() - total_start
    print(f"\n{'='*50}")
    print(f" All done! {len(images)} images in {total_elapsed:.1f}s")
    print(f" Output: {args.output}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
