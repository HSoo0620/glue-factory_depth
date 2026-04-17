"""v1_20260415 학습 설정과 일치하는 전역 상수.

값 변경은 성능 저하를 의미하므로 register_pair API 에서는 외부 override 를
허용하지 않는다 (RANSAC 제외). spec §6 참조.
"""
from __future__ import annotations
from pathlib import Path

# 좌표/단위
LATERAL_MM: float   = 0.056   # X = u * LATERAL_MM
TRANSPORT_MM: float = 0.056   # Y = v * TRANSPORT_MM
VERTICAL_MM: float  = 0.0085  # Z = raw * VERTICAL_MM

# Descriptor
VOXEL_MM: float    = 1.0
NORMAL_R_MM: float = 20.0
FPFH_R_MM: float   = 20.0
SHOT_R_MM: float   = 40.0

# ISS
ERODE_BOUNDARY_PX: int   = 5
ISS_SALIENT_MULT: float  = 6.0
ISS_NONMAX_MULT: float   = 2.0
ISS_GAMMA_21: float      = 0.5
ISS_GAMMA_32: float      = 0.5
ISS_MIN_NEIGHBORS: int   = 5
MAX_KEYPOINTS: int       = 512

# Scanned 전처리
BILAT_D: int               = 5
BILAT_SIGMA_C: float       = 100.0
BILAT_SIGMA_S: float       = 3.0
FLOOR_BAND_FRACTION: float = 0.05
FLOOR_BIN_WIDTH: int       = 50

# LightGlue 입력 zero-pad (학습 이미지 규격)
PAD_H: int = 3008
PAD_W: int = 2432

# RANSAC
DEFAULT_RANSAC_ITER: int = 1000
DEFAULT_INLIER_TH: float = 5.0

# 기본 경로
PKG_DIR: Path   = Path(__file__).resolve().parent              # .../depth_registration
SHARED_DIR: Path = PKG_DIR.parent                              # .../0417_shared
DEFAULT_MASTER_PATH: Path = SHARED_DIR / "masters" / "default_master.png"
DEFAULT_CACHE_DIR: Path   = SHARED_DIR / "cache"
CKPT_FPFH: Path = SHARED_DIR / "checkpoints" / "iss_fpfh_v1_norm_0417.tar"
CKPT_SHOT: Path = SHARED_DIR / "checkpoints" / "iss_shot_v1_dim352_0417.tar"
