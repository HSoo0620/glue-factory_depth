# RoPS135 Descriptor - Output Specification

## Overview

| Item | Value |
|------|-------|
| Dataset | `dataset_resample_2` (641 depth maps, 5761x5761 16-bit PNG) |
| Descriptor | RoPS135 (Rotational Projection Statistics) |
| Dimension | **135** floats per point |
| Frames | 641 |
| Points per frame | ~14,000 ~ 22,000 (after VoxelGrid downsampling) |

## Algorithm Summary

RoPS (Rotational Projection Statistics) is a mesh-based 3D local feature descriptor. Unlike SHOT which operates on unorganized point clouds with normals, RoPS requires **triangle mesh connectivity** to compute a Local Reference Frame (LRF) and project local surface geometry onto distribution matrices.

### Pipeline

```
16-bit depth PNG
  -> Organized triangle mesh (grid adjacency + depth discontinuity filter)
  -> VoxelGrid downsample (keypoints)
  -> RoPS descriptor per keypoint (mesh-based LRF + rotational projections)
  -> Binary output
```

### Descriptor Construction (per keypoint)

1. **Local surface extraction**: Find all mesh triangles within the support radius via KD-tree search + point-to-triangle index
2. **LRF computation**: Weighted scatter matrix of local triangle vertices -> eigendecomposition -> 3 principal axes
3. **Rotation + projection**: For each of 3 axes (X, Y, Z) x 3 rotation angles -> rotate local surface -> project onto 3 planes (XY, XZ, YZ) -> compute 5x5 distribution matrix
4. **Central moments**: Extract 5 statistics per distribution matrix (4 central moments + Shannon entropy)
5. **L1 normalization**: Normalize the full feature vector

**Feature dimension** = `rotations(3) x axes(3) x projections(3) x moments(5)` = **135**

## Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `voxelSize` | 1.0 | VoxelGrid downsample leaf size |
| `ropsRadius` | 10.0 | RoPS support radius (local surface crop) |
| `maxEdgeLen` | 5.0 | Max triangle edge length in 3D (depth discontinuity filter) |
| `meshStride` | 8 | Depth map grid subsampling stride (every Nth pixel) |
| `numBins` | 5 | Distribution matrix partition bins (compile-time) |
| `numRotations` | 3 | Number of rotation steps per axis (compile-time) |

### Mesh Generation Details

- **Method**: Organized fast mesh from depth map grid structure
- **Stride=8**: Samples every 8th pixel -> ~720x720 grid -> ~50K-80K surface points, ~100K-160K triangles
- **Depth discontinuity filter**: Triangles with any edge > `maxEdgeLen` (5.0) are discarded to avoid connecting foreground/background
- Depth scale: `depth = raw_uint16 * (clip_end - clip_start) / 65535 + clip_start`
- Camera intrinsics: `fx=fy=8001.389, cx=cy=2880.5` (from `calib_XXXX.ini`)
- Point cloud coordinate system: camera coordinates (X=right, Y=down, Z=forward)

### Parallelization

PCL's `ROPSEstimation` is single-threaded. This implementation parallelizes via OpenMP by splitting keypoints across threads, each with its own `ROPSEstimation` instance sharing a read-only surface mesh.

## Output Files

Each frame produces two files in `output_rops_10_omp/`:

```
depth_raw_XXXX_cloud.pcd     # Downsampled keypoints (PCL binary PCD)
depth_raw_XXXX_rops135.bin   # RoPS135 descriptors (custom binary)
```

## Binary Format (`_rops135.bin`)

```
Offset  Type        Description
---------------------------------------------
0       uint32      N = number of points
4       uint32      D = descriptor dimension (135)
8       ...         Per-point data (N entries):

Per-point entry (552 bytes each):
  +0    float32     x coordinate
  +4    float32     y coordinate
  +8    float32     z coordinate
  +12   float32[D]  RoPS135 descriptor (135 floats)
```

**Total per-point size** = 4*(3+135) = **552 bytes**

## Loading Example (Python)

```python
import numpy as np

def load_rops135(path):
    with open(path, 'rb') as f:
        header = np.fromfile(f, dtype=np.uint32, count=2)
        N, D = int(header[0]), int(header[1])  # N points, D=135

        points = np.zeros((N, 3), dtype=np.float32)
        descriptors = np.zeros((N, D), dtype=np.float32)

        for i in range(N):
            points[i] = np.fromfile(f, dtype=np.float32, count=3)
            descriptors[i] = np.fromfile(f, dtype=np.float32, count=D)

    return points, descriptors  # (N,3), (N,135)

# Usage
pts, desc = load_rops135("output_rops_10_omp/depth_raw_0000_rops135.bin")
```

### Vectorized Loading (faster)

```python
import numpy as np

def load_rops135_fast(path):
    with open(path, 'rb') as f:
        N, D = np.fromfile(f, dtype=np.uint32, count=2)
        data = np.fromfile(f, dtype=np.float32).reshape(N, 3 + D)
    return data[:, :3], data[:, 3:]  # points (N,3), descriptors (N,135)
```

## Loading Example (C++)

```cpp
#include <fstream>
#include <vector>

struct RoPSFeature {
    float x, y, z;
    float descriptor[135];
};

std::vector<RoPSFeature> loadRoPS135(const std::string& path) {
    std::ifstream fin(path, std::ios::binary);
    uint32_t N, D;
    fin.read(reinterpret_cast<char*>(&N), 4);
    fin.read(reinterpret_cast<char*>(&D), 4);  // D = 135

    std::vector<RoPSFeature> features(N);
    for (uint32_t i = 0; i < N; i++) {
        fin.read(reinterpret_cast<char*>(&features[i]), sizeof(RoPSFeature));
    }
    return features;
}
```

## Feature Matching Guide

### 1. Nearest Neighbor Matching

```python
from scipy.spatial import cKDTree

pts_a, desc_a = load_rops135_fast("output_rops_10_omp/depth_raw_0000_rops135.bin")
pts_b, desc_b = load_rops135_fast("output_rops_10_omp/depth_raw_0001_rops135.bin")

# Build KD-tree on descriptor space
tree = cKDTree(desc_b)
dists, indices = tree.query(desc_a, k=2)  # k=2 for ratio test

# Lowe's ratio test
ratio = dists[:, 0] / (dists[:, 1] + 1e-8)
good = ratio < 0.8
matched_a = pts_a[good]
matched_b = pts_b[indices[good, 0]]
```

### 2. FLANN-based Matching (C++ / PCL)

```cpp
#include <pcl/registration/correspondence_estimation.h>
#include <pcl/registration/correspondence_rejection_sample_consensus.h>

// After loading RoPS descriptors into pcl::PointCloud<pcl::Histogram<135>>:
pcl::registration::CorrespondenceEstimation<pcl::Histogram<135>, pcl::Histogram<135>> est;
est.setInputSource(desc_source);
est.setInputTarget(desc_target);

pcl::Correspondences correspondences;
est.determineReciprocalCorrespondences(correspondences, 0.25f);  // max_distance threshold

// RANSAC-based outlier rejection
pcl::registration::CorrespondenceRejectorSampleConsensus<pcl::PointXYZ> rejector;
rejector.setInputSource(cloud_source);
rejector.setInputTarget(cloud_target);
rejector.setInlierThreshold(2.0);
rejector.getRemainingCorrespondences(correspondences, inliers);
```

### 3. Tips

- **Dense descriptor**: Unlike SHOT352 (sparse, ~100/352 non-zero), RoPS135 is dense — most dimensions are non-zero
- **L1 normalized**: Descriptors are L1-normalized by the PCL implementation. Use **L1 or L2 distance** for matching
- **Zero descriptors**: Some keypoints at mesh boundaries may produce all-zero descriptors. Filter before matching: `valid = np.any(desc != 0, axis=1)`
- **Reciprocal matching** (A->B and B->A agree) significantly reduces false matches
- **RANSAC** on 3D point correspondences removes geometric outliers

## Comparison with SHOT352

| Property | SHOT352 | RoPS135 |
|----------|---------|---------|
| Dimension | 352 | 135 |
| Input requirement | Point cloud + normals | Triangle mesh |
| Sparsity | Sparse (~30% non-zero) | Dense (~100% non-zero) |
| Per-point size | 1,420 bytes | 552 bytes |
| PCL OMP support | Native (`SHOTEstimationOMP`) | Manual (keypoint splitting) |
| Speed (24 threads, stride=8) | ~0.8s/frame | ~12s/frame |
| Rotation invariance | Yes (via LRF from normals) | Yes (via LRF from mesh scatter matrix) |
| Robustness to noise | Moderate | Higher (mesh-based averaging) |

## Command-line Usage

```bash
# Default (stride=8 recommended)
pcl_rops.exe <datasetDir> <outputDir> <voxelSize> <ropsRadius> <maxEdgeLen> <meshStride>

# Example
pcl_rops.exe "../../dataset_resample_2/dataset_resample_2" "../../output_rops_10_omp" 1.0 10.0 5.0 8
```

| Argument | Default | Description |
|----------|---------|-------------|
| `datasetDir` | `../../dataset_resample_2/dataset_resample_2` | Input depth maps |
| `outputDir` | `../../output_rops_10` | Output directory |
| `voxelSize` | 1.0 | VoxelGrid leaf size |
| `ropsRadius` | 10.0 | RoPS support radius |
| `maxEdgeLen` | 5.0 | Max triangle edge length |
| `meshStride` | 1 | Grid subsampling (recommend 8) |
