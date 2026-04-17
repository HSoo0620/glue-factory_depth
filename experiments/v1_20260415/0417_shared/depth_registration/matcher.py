"""LightGlue 로드/추론 래퍼.

입력 batch 구조는 gluefactory.two_view_pipeline 이 LightGlue 로 넘기는 형태와
동일하다: top-level 에 `keypoints{0,1}`, `descriptors{0,1}`, `keypoint_scores{0,1}`
+ `view0/view1` 서브딕트에 `image_size` 가 있어야 normalize_keypoints 가 동작한다.
원본 참조: experiments/v1_20260415/infer_scanned_vs_master_shot352.py
"""
from __future__ import annotations
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from omegaconf import OmegaConf

from . import params as P
from ._vendored.lightglue import LightGlue
from ._vendored.tensor import batch_to_device

Descriptor = Literal["fpfh", "shot"]


def _ckpt_path(descriptor: Descriptor) -> Path:
    return P.CKPT_FPFH if descriptor == "fpfh" else P.CKPT_SHOT


def _default_matcher_conf(descriptor: Descriptor) -> dict:
    """ckpt 에 conf 가 빠져 있을 때 쓰는 학습 기본값 (configs/iss_{fpfh,shot}_v1*.yaml 참조)."""
    if descriptor == "fpfh":
        return dict(input_dim=33, descriptor_dim=36, num_heads=3,
                    filter_threshold=0.1, flash=False, checkpointed=True)
    return dict(input_dim=352, descriptor_dim=352, num_heads=4,
                filter_threshold=0.1, flash=False, checkpointed=True)


def load_lightglue(descriptor: Descriptor, device: str = "cuda") -> torch.nn.Module:
    ckpt_path = _ckpt_path(descriptor)
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"LightGlue checkpoint not found: {ckpt_path}. "
            f"SHOT ckpt 의 경우 별도 전달된 파일을 이 경로로 복사해야 함."
        )
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    cfg_from_ckpt = {}
    try:
        cfg_from_ckpt = dict(ckpt.get("conf", {}).get("model", {}).get("matcher", {}))
    except Exception:
        cfg_from_ckpt = {}
    cfg = {**_default_matcher_conf(descriptor), **cfg_from_ckpt}
    cfg.pop("name", None)

    model = LightGlue(OmegaConf.create(cfg))

    sd = ckpt.get("model", ckpt)
    matcher_sd = {
        k[len("matcher."):]: v for k, v in sd.items() if k.startswith("matcher.")
    }
    if not matcher_sd:     # state_dict 이 이미 flat 상태이면 그대로 사용
        matcher_sd = sd
    model.load_state_dict(matcher_sd, strict=False)
    model.eval().to(device)
    return model


def mm_to_uv(kp_mm: np.ndarray) -> np.ndarray:
    """(K,3) mm → (K,2) uv (X,Y 만 사용)."""
    u = kp_mm[:, 0] / P.LATERAL_MM
    v = kp_mm[:, 1] / P.TRANSPORT_MM
    return np.stack([u, v], axis=1).astype(np.float32)


@torch.no_grad()
def run_lightglue(model: torch.nn.Module,
                  kp0_uv: np.ndarray, desc0: np.ndarray,
                  kp1_uv: np.ndarray, desc1: np.ndarray,
                  device: str = "cuda") -> dict:
    """LightGlue 한번 호출. 반환: {'matches0': (K0,), 'matching_scores0': (K0,)}"""
    size = torch.tensor([[P.PAD_H, P.PAD_W]], dtype=torch.long, device=device)

    def _kp(a):   return torch.from_numpy(a[None]).float()
    def _desc(a): return torch.from_numpy(a[None]).float()

    batch = {
        "view0": {"image_size": size},
        "view1": {"image_size": size},
        "keypoints0":       _kp(kp0_uv),
        "keypoints1":       _kp(kp1_uv),
        "descriptors0":     _desc(desc0),
        "descriptors1":     _desc(desc1),
        "keypoint_scores0": torch.ones((1, len(kp0_uv)), dtype=torch.float32),
        "keypoint_scores1": torch.ones((1, len(kp1_uv)), dtype=torch.float32),
    }
    batch = batch_to_device(batch, device)
    pred = model(batch)
    return {
        "matches0":         pred["matches0"][0].cpu().numpy(),
        "matching_scores0": pred["matching_scores0"][0].cpu().numpy(),
    }
