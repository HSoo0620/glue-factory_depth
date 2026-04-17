# SHOT352 Descriptor - Output Specification

## Overview

| Item | Value |
|------|-------|
| Dataset | `dataset_resample_2` (641 depth maps, 5761x5761 16-bit PNG) |
| Descriptor | SHOT352 (Signature of Histograms of Orientations) |
| Dimension | **352** floats per point |
| Frames | 641 |
| Points per frame | ~18,000 ~ 35,000 (after VoxelGrid downsampling) |

## Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `voxelSize` | 1.0 | VoxelGrid downsample leaf size |
| `normalRadius` | 5.0 | Normal estimation search radius |
| `shotRadius` | 10.0 | SHOT descriptor support radius |

- Depth scale: `depth = raw_uint16 * (clip_end - clip_start) / 65535 + clip_start`
- Camera intrinsics: `fx=fy=8001.389, cx=cy=2880.5` (from `calib_XXXX.ini`)
- Point cloud coordinate system: camera coordinates (X=right, Y=down, Z=forward)

## Output Files

Each frame produces two files in `output_shot/`:

```
depth_raw_XXXX_cloud.pcd    # Downsampled point cloud (PCL binary PCD)
depth_raw_XXXX_shot352.bin  # SHOT352 descriptors (custom binary)
```

## Binary Format (`_shot352.bin`)

```
Offset  Type        Description
─────────────────────────────────────────────
0       uint32      N = number of points
4       uint32      D = descriptor dimension (352)
8       ...         Per-point data (N entries):

Per-point entry (1420 bytes each):
  +0    float32     x coordinate
  +4    float32     y coordinate
  +8    float32     z coordinate
  +12   float32[D]  SHOT352 descriptor (352 floats)
```

**Total per-point size** = 4*(3+352) = **1,420 bytes**

## Loading Example (Python)

```python
import numpy as np

def load_shot352(path):
    with open(path, 'rb') as f:
        header = np.fromfile(f, dtype=np.uint32, count=2)
        N, D = int(header[0]), int(header[1])  # N points, D=352

        points = np.zeros((N, 3), dtype=np.float32)
        descriptors = np.zeros((N, D), dtype=np.float32)

        for i in range(N):
            points[i] = np.fromfile(f, dtype=np.float32, count=3)
            descriptors[i] = np.fromfile(f, dtype=np.float32, count=D)

    return points, descriptors  # (N,3), (N,352)

# Usage
pts, desc = load_shot352("output_shot/depth_raw_0000_shot352.bin")
```

### Vectorized Loading (faster)

```python
import numpy as np

def load_shot352_fast(path):
    with open(path, 'rb') as f:
        N, D = np.fromfile(f, dtype=np.uint32, count=2)
        data = np.fromfile(f, dtype=np.float32).reshape(N, 3 + D)
    return data[:, :3], data[:, 3:]  # points (N,3), descriptors (N,352)
```

## Loading Example (C++ / PCL)

```cpp
#include <fstream>
#include <vector>

struct SHOTFeature {
    float x, y, z;
    float descriptor[352];
};

std::vector<SHOTFeature> loadSHOT352(const std::string& path) {
    std::ifstream fin(path, std::ios::binary);
    uint32_t N, D;
    fin.read(reinterpret_cast<char*>(&N), 4);
    fin.read(reinterpret_cast<char*>(&D), 4);  // D = 352

    std::vector<SHOTFeature> features(N);
    for (uint32_t i = 0; i < N; i++) {
        fin.read(reinterpret_cast<char*>(&features[i]), sizeof(SHOTFeature));
    }
    return features;
}
```

## Feature Matching Guide

### 1. Nearest Neighbor Matching

```python
from scipy.spatial import cKDTree

pts_a, desc_a = load_shot352_fast("output_shot/depth_raw_0000_shot352.bin")
pts_b, desc_b = load_shot352_fast("output_shot/depth_raw_0001_shot352.bin")

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

// After loading SHOT descriptors into pcl::PointCloud<pcl::SHOT352>:
pcl::registration::CorrespondenceEstimation<pcl::SHOT352, pcl::SHOT352> est;
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

- **NaN descriptors**: Some points may have all-zero descriptors (invalid local reference frame at boundaries). Filter these before matching: `valid = np.any(desc != 0, axis=1)`
- **L2 distance** is the standard metric for SHOT352 matching
- **Reciprocal matching** (A->B and B->A agree) significantly reduces false matches
- **RANSAC** on 3D point correspondences removes geometric outliers
- Typical non-zero bins: ~100-113 out of 352 dimensions (sparse descriptor)
