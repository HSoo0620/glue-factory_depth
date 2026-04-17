"""Inference pipeline for ISS + FPFH/SHOT + LightGlue on the new dataset.

Design goals:
  - Single module entry point for the GUI viewer team.
  - Independent of training code path; does not touch DataLoader/Dataset.
  - Optional: SHOT descriptors come from an external C++ .bin file (per zmap).
  - Transform frame defaults to camera-frame mm. A per-view world override
    (R_cam, t_cam) may be supplied to return the transform in world frame.

Typical flow:
    pipeline = InferencePipeline(
        checkpoint="outputs/training/<exp>/checkpoint_best.tar",
        descriptor_type="fpfh",
        device="cuda",
    )
    feats_m = pipeline.extract_features(zmap_master_u16)
    feats_i = pipeline.extract_features(zmap_input_u16)
    result = pipeline.match(feats_m, feats_i)
    # result.T_est: 4x4 rigid transform (master → input), camera-frame mm
    # result.kp0_matched_uv, result.kp1_matched_uv: pixel coords (original res)

SHOT variant: feats = pipeline.extract_features(zmap, shot_bin_path=".../zmap_0042.bin")
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import open3d as o3d
import torch

from gluefactory.datasets.new_dataset import constants as C
from gluefactory.datasets.new_dataset.coords import (
    build_camera_frame_pcd,
    pixel_to_cam_xyz,
)
from gluefactory.datasets.new_dataset.iss_detection import (
    build_iss_pcd_uvd_scaled,
    detect_iss_keypoints,
    select_keypoints,
)


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class CameraCalib:
    """Per-camera mm/pixel calibration. Defaults match the training dataset."""
    lat_mm: float = C.LAT_MM
    trans_mm: float = C.TRANS_MM
    vert_mm: float = C.VERT_MM

    def pixel_to_cam(self, u, v, raw) -> np.ndarray:
        u = np.asarray(u, dtype=np.float64)
        v = np.asarray(v, dtype=np.float64)
        raw = np.asarray(raw, dtype=np.float64)
        return np.stack(
            [u * self.lat_mm, v * self.trans_mm, raw * self.vert_mm], axis=-1
        )


@dataclass
class ViewFeatures:
    """Per-view precomputed features, ready for the matcher."""
    kp_uv_orig: np.ndarray        # (N, 2) float64, original-resolution pixels
    kp_uv_resized: np.ndarray     # (N, 2) float32, fed to LG (after resize_factor)
    kp_xyz_cam: np.ndarray        # (N, 3) float64, camera-frame mm
    kp_valid_mask: np.ndarray     # (N,) bool — raw > 0 at kp location
    descriptors: np.ndarray       # (N, D) float32, L2-normalized
    scores: np.ndarray            # (N,) float32
    zmap_resized: np.ndarray      # (H, W) float32 in [0, 1] — LG input
    zmap_shape_orig: tuple[int, int]  # (H, W)
    n_valid: int                  # kp slots filled (<= N)


@dataclass
class InferenceResult:
    kp0_matched_uv: np.ndarray    # (M, 2) original-res pixels
    kp1_matched_uv: np.ndarray
    kp0_matched_xyz: np.ndarray   # (M, 3) camera-frame mm
    kp1_matched_xyz: np.ndarray
    match_scores: np.ndarray      # (M,)
    T_est: np.ndarray             # (4, 4) master → input
    inlier_mask: np.ndarray       # (M,) bool
    frame: str = "camera"         # "camera" | "world"
    extra: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# SHOT .bin loader (C++ external descriptor)
# ---------------------------------------------------------------------------


def load_shot_bin_descriptors(bin_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Parse a C++ SHOT352 binary file.

    Layout (float32 little-endian):
        N uint32
        points   (N, 3) float32   # camera-frame mm
        descs    (N, 352) float32
    Returns (points_mm, desc). NaNs in desc are filtered.
    """
    bin_path = Path(bin_path)
    raw = np.fromfile(bin_path, dtype=np.uint8)
    # File may be stored as header (int32) + interleaved floats, or simply a
    # flat (points | desc) concatenation. The existing precompute script
    # handles multiple layouts — TODO: copy its exact parsing when wiring the
    # real C++ format here.
    raise NotImplementedError(
        "load_shot_bin_descriptors: port the parsing from "
        "precompute_new_iss_shot.py before enabling SHOT inference."
    )


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


class InferencePipeline:
    """End-to-end matcher for two depth zmaps."""

    def __init__(
        self,
        checkpoint: str | Path | None = None,
        descriptor_type: str = "fpfh",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        calib: Optional[CameraCalib] = None,
        *,
        resize_factor: float = C.RESIZE_FACTOR,
        max_num_keypoints: int = 512,
        # FPFH params (used if descriptor_type == "fpfh"; match training defaults)
        voxel_size: float = 5.0,
        fpfh_radius: float = 50.0,
        fpfh_normal_radius: float = 25.0,
        anomaly_dist_mm: float = 10.0,
        # ISS params
        iss_gamma_21: float = 0.5,
        iss_gamma_32: float = 0.5,
        iss_min_neighbors: int = 5,
        iss_erode_boundary: int = 5,
        # Registration
        ransac_iter: int = 1000,
        ransac_inlier_th_mm: float = 5.0,
    ):
        if descriptor_type not in ("fpfh", "shot"):
            raise ValueError(f"descriptor_type must be 'fpfh' or 'shot', got {descriptor_type!r}")

        self.descriptor_type = descriptor_type
        self.device = device
        self.calib = calib or CameraCalib()
        self.resize_factor = float(resize_factor)
        self.max_num_keypoints = int(max_num_keypoints)
        self.voxel_size = float(voxel_size)
        self.fpfh_radius = float(fpfh_radius)
        self.fpfh_normal_radius = float(fpfh_normal_radius)
        self.anomaly_dist_mm = float(anomaly_dist_mm)
        self.iss_params = dict(
            gamma_21=iss_gamma_21,
            gamma_32=iss_gamma_32,
            min_neighbors=iss_min_neighbors,
            erode_boundary=iss_erode_boundary,
        )
        self.ransac_iter = int(ransac_iter)
        self.ransac_inlier_th_mm = float(ransac_inlier_th_mm)
        self._rng = np.random.default_rng(0)

        self.model = None
        self.checkpoint_path = Path(checkpoint) if checkpoint else None
        if self.checkpoint_path is not None:
            self._load_model(self.checkpoint_path)

    # ------------------------------------------------------------------ model

    def _load_model(self, ckpt_path: Path) -> None:
        from omegaconf import OmegaConf

        from gluefactory.models import get_model

        if not ckpt_path.exists():
            raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")
        cp = torch.load(ckpt_path, map_location="cpu")
        conf = OmegaConf.create(cp["conf"])
        model = get_model(conf.model.name)(conf.model).to(self.device)
        model.load_state_dict(cp["model"], strict=False)
        model.eval()
        self.model = model

    # ------------------------------------------------------------- extraction

    def extract_features(
        self,
        zmap_u16: np.ndarray,
        *,
        shot_bin_path: str | Path | None = None,
    ) -> ViewFeatures:
        """ISS keypoints + per-keypoint descriptor (FPFH or SHOT)."""
        if zmap_u16.dtype != np.uint16:
            raise TypeError(f"zmap must be uint16, got {zmap_u16.dtype}")

        H, W = zmap_u16.shape
        new_h = int(round(H * self.resize_factor))
        new_w = int(round(W * self.resize_factor))
        zmap_resized = cv2.resize(
            zmap_u16, (new_w, new_h), interpolation=cv2.INTER_NEAREST
        ).astype(np.float32) / 65535.0

        # 1. ISS detection in (u, v, depth_scaled)
        pcd_iss, erode_mask, _, _ = build_iss_pcd_uvd_scaled(
            zmap_u16, erode_boundary=self.iss_params["erode_boundary"]
        )
        iss_kp_3d = detect_iss_keypoints(
            pcd_iss,
            gamma_21=self.iss_params["gamma_21"],
            gamma_32=self.iss_params["gamma_32"],
            min_neighbors=self.iss_params["min_neighbors"],
        )
        kp_resized, scores, n_valid, kp_uv_orig = select_keypoints(
            iss_kp_3d, zmap_u16,
            max_num_keypoints=self.max_num_keypoints,
            erode_mask=erode_mask,
            resize_factor=self.resize_factor,
            rng=self._rng,
        )

        # 2. Keypoint XYZ (camera-frame mm) at original-resolution (u, v, raw)
        kp_xyz = np.zeros((self.max_num_keypoints, 3), dtype=np.float64)
        valid_mask = np.zeros(self.max_num_keypoints, dtype=bool)
        for i in range(n_valid):
            u = int(round(float(kp_uv_orig[i, 0])))
            v = int(round(float(kp_uv_orig[i, 1])))
            u = max(0, min(u, W - 1))
            v = max(0, min(v, H - 1))
            raw = zmap_u16[v, u]
            if raw == 0:
                continue
            kp_xyz[i] = self.calib.pixel_to_cam(u, v, raw)
            valid_mask[i] = True

        # 3. Descriptor
        if self.descriptor_type == "fpfh":
            desc = self._compute_fpfh_descriptors(
                zmap_u16, kp_xyz, n_valid, valid_mask,
            )
        else:
            if shot_bin_path is None:
                raise ValueError(
                    "descriptor_type='shot' requires shot_bin_path "
                    "(pre-computed C++ output per zmap)."
                )
            desc = self._lookup_shot_descriptors(
                shot_bin_path, kp_xyz, n_valid, valid_mask,
            )

        return ViewFeatures(
            kp_uv_orig=kp_uv_orig,
            kp_uv_resized=kp_resized,
            kp_xyz_cam=kp_xyz,
            kp_valid_mask=valid_mask,
            descriptors=desc,
            scores=scores,
            zmap_resized=zmap_resized,
            zmap_shape_orig=(H, W),
            n_valid=int(n_valid),
        )

    def _compute_fpfh_descriptors(
        self,
        zmap_u16: np.ndarray,
        kp_xyz: np.ndarray,
        n_valid: int,
        valid_mask: np.ndarray,
    ) -> np.ndarray:
        out = np.zeros((self.max_num_keypoints, 33), dtype=np.float32)
        if n_valid < 3:
            return out

        pcd_xyz, _ = build_camera_frame_pcd(
            zmap_u16, erode_boundary=self.iss_params["erode_boundary"]
        )
        if len(pcd_xyz.points) == 0:
            return out

        if self.voxel_size > 0:
            pcd_xyz = pcd_xyz.voxel_down_sample(self.voxel_size)
        pcd_xyz.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=self.fpfh_normal_radius, max_nn=30
            )
        )
        fpfh = o3d.pipelines.registration.compute_fpfh_feature(
            pcd_xyz,
            o3d.geometry.KDTreeSearchParamHybrid(
                radius=self.fpfh_radius, max_nn=100
            ),
        )
        fpfh_dense = np.asarray(fpfh.data).T.astype(np.float32)  # (M, 33)
        kdtree = o3d.geometry.KDTreeFlann(pcd_xyz)
        pts = np.asarray(pcd_xyz.points)

        for i in range(n_valid):
            if not valid_mask[i]:
                continue
            _, idx, _ = kdtree.search_knn_vector_3d(kp_xyz[i], 1)
            j = idx[0]
            if float(np.linalg.norm(pts[j] - kp_xyz[i])) > self.anomaly_dist_mm:
                continue
            out[i] = fpfh_dense[j]

        n = np.linalg.norm(out[:n_valid], axis=1, keepdims=True)
        out[:n_valid] = out[:n_valid] / (n + 1e-8)
        return out

    def _lookup_shot_descriptors(
        self,
        shot_bin_path: str | Path,
        kp_xyz: np.ndarray,
        n_valid: int,
        valid_mask: np.ndarray,
    ) -> np.ndarray:
        pts_mm, desc_all = load_shot_bin_descriptors(Path(shot_bin_path))
        # Build KDTree on camera-frame mm and look up nearest descriptor
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts_mm.astype(np.float64))
        kdtree = o3d.geometry.KDTreeFlann(pcd)

        out = np.zeros((self.max_num_keypoints, 352), dtype=np.float32)
        for i in range(n_valid):
            if not valid_mask[i]:
                continue
            _, idx, _ = kdtree.search_knn_vector_3d(kp_xyz[i], 1)
            out[i] = desc_all[idx[0]]
        n = np.linalg.norm(out[:n_valid], axis=1, keepdims=True)
        out[:n_valid] = out[:n_valid] / (n + 1e-8)
        return out

    # ---------------------------------------------------------------- matcher

    @torch.no_grad()
    def match(
        self,
        feats_m: ViewFeatures,
        feats_i: ViewFeatures,
        *,
        world_rt_master: tuple[np.ndarray, np.ndarray] | None = None,
        world_rt_input: tuple[np.ndarray, np.ndarray] | None = None,
    ) -> InferenceResult:
        """Run LightGlue + RANSAC+SVD. Optionally return transform in world frame.

        world_rt_* = (R (3,3), t (3,)) to convert camera-frame mm → world.
        """
        if self.model is None:
            raise RuntimeError(
                "No checkpoint loaded. Pass `checkpoint=...` to InferencePipeline."
            )

        batch = self._pack_batch(feats_m, feats_i)
        pred = self.model(batch)

        kp0 = pred["keypoints0"][0].cpu().numpy()          # (N, 2) resized pixels
        kp1 = pred["keypoints1"][0].cpu().numpy()
        m0 = pred["matches0"][0].cpu().numpy()
        scores = pred.get("matching_scores0", None)
        if scores is not None:
            scores = scores[0].cpu().numpy()

        valid = m0 > -1
        kp0_valid_res = kp0[valid]
        kp1_valid_res = kp1[m0[valid]]
        scores_m = scores[valid] if scores is not None else np.ones(valid.sum())

        # Convert resized-LG pixels back to original pixels
        kp0_uv = kp0_valid_res / self.resize_factor
        kp1_uv = kp1_valid_res / self.resize_factor

        # Pull the precomputed camera-frame XYZ for matched indices
        idx0 = np.flatnonzero(valid)
        idx1 = m0[valid]
        xyz0 = feats_m.kp_xyz_cam[idx0]
        xyz1 = feats_i.kp_xyz_cam[idx1]
        keep = feats_m.kp_valid_mask[idx0] & feats_i.kp_valid_mask[idx1]
        if keep.sum() < 3:
            return InferenceResult(
                kp0_matched_uv=kp0_uv, kp1_matched_uv=kp1_uv,
                kp0_matched_xyz=xyz0, kp1_matched_xyz=xyz1,
                match_scores=scores_m, T_est=np.eye(4),
                inlier_mask=np.zeros(len(kp0_uv), dtype=bool),
                frame="camera",
                extra={"reason": "too few valid kp XYZ (<3)"},
            )

        xyz0_k = xyz0[keep]; xyz1_k = xyz1[keep]

        # Optional world-frame conversion before RANSAC so T_est is in world
        frame = "camera"
        if world_rt_master is not None and world_rt_input is not None:
            Rm, tm = world_rt_master
            Ri, ti = world_rt_input
            xyz0_k = xyz0_k @ Rm.T + tm
            xyz1_k = xyz1_k @ Ri.T + ti
            frame = "world"

        T_est, inl_keep = _ransac_rigid_svd(
            xyz0_k, xyz1_k, n_iter=self.ransac_iter,
            inlier_th=self.ransac_inlier_th_mm, rng=self._rng,
        )

        inlier_mask = np.zeros(len(kp0_uv), dtype=bool)
        inlier_mask[np.flatnonzero(keep)[inl_keep]] = True

        return InferenceResult(
            kp0_matched_uv=kp0_uv, kp1_matched_uv=kp1_uv,
            kp0_matched_xyz=xyz0, kp1_matched_xyz=xyz1,
            match_scores=scores_m, T_est=T_est,
            inlier_mask=inlier_mask, frame=frame,
        )

    def run(
        self,
        zmap_master: np.ndarray,
        zmap_input: np.ndarray,
        *,
        shot_bin_master: str | Path | None = None,
        shot_bin_input: str | Path | None = None,
        world_rt_master: tuple[np.ndarray, np.ndarray] | None = None,
        world_rt_input: tuple[np.ndarray, np.ndarray] | None = None,
    ) -> InferenceResult:
        """Convenience wrapper: extract both views then match."""
        feats_m = self.extract_features(zmap_master, shot_bin_path=shot_bin_master)
        feats_i = self.extract_features(zmap_input, shot_bin_path=shot_bin_input)
        return self.match(
            feats_m, feats_i,
            world_rt_master=world_rt_master, world_rt_input=world_rt_input,
        )

    # ------------------------------------------------------------- batching

    def _pack_batch(self, feats_m: ViewFeatures, feats_i: ViewFeatures) -> dict:
        def view(f: ViewFeatures) -> dict:
            img = torch.from_numpy(f.zmap_resized).unsqueeze(0).unsqueeze(0)
            H, W = f.zmap_resized.shape
            return {
                "image": img.to(self.device),
                "image_size": torch.tensor([[H, W]], dtype=torch.long, device=self.device),
                "keypoints": torch.from_numpy(f.kp_uv_resized)
                    .float().unsqueeze(0).to(self.device),
                "keypoint_scores": torch.from_numpy(f.scores)
                    .float().unsqueeze(0).to(self.device),
                "descriptors": torch.from_numpy(f.descriptors)
                    .float().unsqueeze(0).to(self.device),
            }
        return {"view0": view(feats_m), "view1": view(feats_i)}


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------


def _rigid_svd(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    sc = src.mean(0); dc = dst.mean(0)
    H = (src - sc).T @ (dst - dc)
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1] *= -1
        R = Vt.T @ U.T
    T = np.eye(4); T[:3, :3] = R; T[:3, 3] = dc - R @ sc
    return T


def _ransac_rigid_svd(src, dst, n_iter, inlier_th, rng, min_samples=3):
    N = src.shape[0]
    if N < min_samples:
        return np.eye(4), np.zeros(N, dtype=bool)
    best_T, best_inl, best_cnt = np.eye(4), np.zeros(N, dtype=bool), -1
    for _ in range(n_iter):
        idx = rng.choice(N, min_samples, replace=False)
        T = _rigid_svd(src[idx], dst[idx])
        proj = src @ T[:3, :3].T + T[:3, 3]
        inl = np.linalg.norm(proj - dst, axis=1) < inlier_th
        if inl.sum() > best_cnt:
            best_T, best_inl, best_cnt = T, inl, int(inl.sum())
    if best_cnt >= min_samples:
        best_T = _rigid_svd(src[best_inl], dst[best_inl])
    return best_T, best_inl
