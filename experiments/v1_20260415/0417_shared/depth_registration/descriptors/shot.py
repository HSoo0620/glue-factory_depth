"""SHOT352 descriptor via pybind shot_module.

Training (`experiments/v1_20260415/precompute/iss_shot.py`) 은 Linux 용 바인딩
`pybind_shot_linux/shot_module.cpython-310-x86_64-linux-gnu.so` 를 사용해
`extract_shot_at_keypoints(vox_pts, kp_xyz, ...)` 로 계산한 raw SHOT352 를
그대로 저장했다 (L2 정규화 없음). 추론에서도 동일 호출/정규화를 유지한다.

Windows:  `pybind_shot_window/shot_module.cp313-win_amd64.pyd` + PCL 1.15.1 DLL
Linux:    `pybind_shot_linux/shot_module.cpython-310-x86_64-linux-gnu.so`

Import 시점에 모듈이 없으면 에러를 지연시켜, FPFH 경로만 쓰는 사용자는
PCL 없이도 정상 동작한다.
"""
from __future__ import annotations
import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import open3d as o3d

from .. import params as P

_SHOT_IMPORT_ERROR: Optional[Exception] = None
_shot_module = None


def _repo_root() -> Path:
    # __file__ = <repo>/experiments/v1_20260415/0417_shared/depth_registration/descriptors/shot.py
    # parents: 0=descriptors, 1=depth_registration, 2=0417_shared,
    #          3=v1_20260415, 4=experiments, 5=<repo_root>
    return Path(__file__).resolve().parents[5]


def _setup_shot_import_path() -> None:
    """OS 별 pybind 디렉터리를 sys.path 에 등록.

    Windows 에서는 PCL 1.15.1 의 DLL 경로도 미리 등록한다 (PCL_ROOT override 허용).
    """
    repo_root = _repo_root()
    if sys.platform == "win32":
        pcl_root = Path(os.environ.get("PCL_ROOT", r"C:\Program Files\PCL 1.15.1"))
        for sub in ("bin", r"3rdParty\FLANN\bin", r"3rdParty\VTK\bin"):
            p = pcl_root / sub
            if p.exists():
                os.add_dll_directory(str(p))
        pybind_dir = repo_root / "pybind_shot_window"
    else:
        pybind_dir = repo_root / "pybind_shot_linux"
    if pybind_dir.exists():
        sys.path.insert(0, str(pybind_dir))


def _lazy_import_shot():
    global _shot_module, _SHOT_IMPORT_ERROR
    if _shot_module is not None:
        return _shot_module
    if _SHOT_IMPORT_ERROR is not None:
        raise _SHOT_IMPORT_ERROR
    try:
        _setup_shot_import_path()
        import shot_module as _m
        _shot_module = _m
        return _m
    except Exception as e:
        _SHOT_IMPORT_ERROR = RuntimeError(
            "SHOT requires shot_module (Windows: PCL 1.15.1 + .pyd, "
            "Linux: prebuilt .so in pybind_shot_linux). "
            "See environment/setup_windows.md. "
            f"Original error: {e!r}"
        )
        raise _SHOT_IMPORT_ERROR


def compute_shot(pts_mm: np.ndarray, kp_mm: np.ndarray,
                 voxel: float = P.VOXEL_MM,
                 normal_r: float = P.NORMAL_R_MM,
                 shot_r: float = P.SHOT_R_MM) -> np.ndarray:
    """pybind `extract_shot_at_keypoints` 로 keypoint-only SHOT352 추출.

    훈련과 동일하게 dense → voxel downsample → `extract_shot_at_keypoints` 를
    호출하며, `valid_mask=False` 행은 0-벡터 그대로 유지한다 (훈련 cache 규약과 일치).
    """
    if len(kp_mm) == 0:
        return np.zeros((0, 352), dtype=np.float32)

    sm = _lazy_import_shot()

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts_mm.astype(np.float64))
    if voxel > 0:
        pcd = pcd.voxel_down_sample(voxel)
    vox_pts = np.asarray(pcd.points, dtype=np.float32)

    result = sm.extract_shot_at_keypoints(
        vox_pts,
        kp_mm.astype(np.float32),
        voxel_size=float(voxel),
        normal_radius=float(normal_r),
        shot_radius=float(shot_r),
    )
    desc = np.nan_to_num(
        np.asarray(result["descriptors"], dtype=np.float32), nan=0.0
    )
    return desc
