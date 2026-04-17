"""
LightGlue with 3D Positional Encoding.

기존 LightGlue의 2D PE (x, y) 대신 depth map을 이용하여
3D 좌표 (X, Y, Z)로 변환한 뒤 PE를 생성합니다.

기존 lightglue.py를 수정하지 않고 독립적으로 동작합니다.
"""

import warnings
from pathlib import Path
from typing import Callable, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch import nn

from ...settings import DATA_PATH
from ..utils.losses import NLLLoss
from ..utils.metrics import matcher_metrics

FLASH_AVAILABLE = hasattr(F, "scaled_dot_product_attention")

torch.backends.cudnn.deterministic = True

AMP_CUSTOM_FWD_F32 = (
    torch.amp.custom_fwd(cast_inputs=torch.float32, device_type="cuda")
    if hasattr(torch.amp, "custom_fwd")
    else torch.cuda.amp.custom_fwd(cast_inputs=torch.float32)
)


# ──────────────────────────────────────────────
# 3D keypoint utilities
# ──────────────────────────────────────────────

@AMP_CUSTOM_FWD_F32
def sample_depth_at_kpts(kpts, depth_map):
    """Keypoint 위치에서 depth 값을 nearest neighbor로 샘플링.

    Args:
        kpts: (B, N, 2) pixel coordinates (x, y)
        depth_map: (B, 1, H, W) normalized depth [0, 1]
    Returns:
        depth_vals: (B, N) depth values
        valid: (B, N) bool mask (depth > 0)
    """
    B, N, _ = kpts.shape
    H, W = depth_map.shape[-2:]

    # (x, y) → grid_sample expects [-1, 1]
    grid = kpts.clone()
    grid[..., 0] = grid[..., 0] / (W - 1) * 2 - 1  # x
    grid[..., 1] = grid[..., 1] / (H - 1) * 2 - 1  # y
    grid = grid[:, None, :, :]  # (B, 1, N, 2)

    sampled = F.grid_sample(
        depth_map, grid, mode="nearest", align_corners=True
    )  # (B, 1, 1, N)
    depth_vals = sampled[:, 0, 0, :]  # (B, N)
    valid = depth_vals > 0
    return depth_vals, valid


@AMP_CUSTOM_FWD_F32
def kpts_to_3d(kpts, depth_vals, valid, fx, fy, cx, cy,
               clip_start, clip_end, image_size, orig_image_size):
    """2D keypoints + depth를 3D 좌표로 변환.

    Args:
        kpts: (B, N, 2) pixel coordinates (x, y) at current resolution
        depth_vals: (B, N) normalized depth [0, 1]
        valid: (B, N) validity mask
        fx, fy, cx, cy: camera intrinsics at orig_image_size
        clip_start, clip_end: depth conversion range
        image_size: current image resolution (scalar)
        orig_image_size: original image resolution (scalar, e.g. 5761)
    Returns:
        kpts_3d: (B, N, 3) 3D coordinates (X, Y, Z)
    """
    # Scale intrinsics to current resolution
    scale = image_size / orig_image_size
    fx_s = fx * scale
    fy_s = fy * scale
    cx_s = cx * scale
    cy_s = cy * scale

    # Convert normalized depth to real depth
    Z = clip_start + depth_vals * (clip_end - clip_start)  # (B, N)
    Z = torch.where(valid, Z, torch.zeros_like(Z))

    # Backproject to 3D
    x = kpts[..., 0]  # (B, N)
    y = kpts[..., 1]  # (B, N)
    X = (x - cx_s) / fx_s * Z
    Y = (y - cy_s) / fy_s * Z

    kpts_3d = torch.stack([X, Y, Z], dim=-1)  # (B, N, 3)
    return kpts_3d


@AMP_CUSTOM_FWD_F32
def normalize_keypoints_3d(kpts_3d, valid):
    """3D 좌표를 [-1, 1] 범위로 정규화.

    유효한 점들의 median과 max extent 기준으로 정규화합니다.
    depth=0 (invalid) 점은 (0, 0, 0)으로 유지됩니다.

    Args:
        kpts_3d: (B, N, 3) 3D coordinates
        valid: (B, N) validity mask
    Returns:
        kpts_3d_norm: (B, N, 3) normalized 3D coordinates
    """
    B, N, _ = kpts_3d.shape
    kpts_3d_norm = torch.zeros_like(kpts_3d)

    for b in range(B):
        mask = valid[b]  # (N,)
        if mask.sum() == 0:
            continue
        pts = kpts_3d[b, mask]  # (M, 3)
        center = pts.median(dim=0).values  # (3,)
        centered = pts - center
        scale = centered.abs().max() + 1e-6
        kpts_3d_norm[b, mask] = centered / scale

    return kpts_3d_norm


# ──────────────────────────────────────────────
# LightGlue building blocks (동일)
# ──────────────────────────────────────────────

def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x = x.unflatten(-1, (-1, 2))
    x1, x2 = x.unbind(dim=-1)
    return torch.stack((-x2, x1), dim=-1).flatten(start_dim=-2)


def apply_cached_rotary_emb(freqs: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    return (t * freqs[0]) + (rotate_half(t) * freqs[1])


class LearnableFourierPositionalEncoding(nn.Module):
    def __init__(self, M: int, dim: int, F_dim: int = None,
                 gamma: float = 1.0) -> None:
        super().__init__()
        F_dim = F_dim if F_dim is not None else dim
        self.gamma = gamma
        self.Wr = nn.Linear(M, F_dim // 2, bias=False)
        nn.init.normal_(self.Wr.weight.data, mean=0, std=self.gamma ** -2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        projected = self.Wr(x)
        cosines, sines = torch.cos(projected), torch.sin(projected)
        emb = torch.stack([cosines, sines], 0).unsqueeze(-3)
        return emb.repeat_interleave(2, dim=-1)


class TokenConfidence(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.token = nn.Sequential(nn.Linear(dim, 1), nn.Sigmoid())
        self.loss_fn = nn.BCEWithLogitsLoss(reduction="none")

    def forward(self, desc0: torch.Tensor, desc1: torch.Tensor):
        return (
            self.token(desc0.detach()).squeeze(-1),
            self.token(desc1.detach()).squeeze(-1),
        )

    def loss(self, desc0, desc1, la_now, la_final):
        logit0 = self.token[0](desc0.detach()).squeeze(-1)
        logit1 = self.token[0](desc1.detach()).squeeze(-1)
        la_now, la_final = la_now.detach(), la_final.detach()
        correct0 = (
            la_final[:, :-1, :].max(-1).indices == la_now[:, :-1, :].max(-1).indices
        )
        correct1 = (
            la_final[:, :, :-1].max(-2).indices == la_now[:, :, :-1].max(-2).indices
        )
        return (
            self.loss_fn(logit0, correct0.float()).mean(-1)
            + self.loss_fn(logit1, correct1.float()).mean(-1)
        ) / 2.0


class Attention(nn.Module):
    def __init__(self, allow_flash: bool) -> None:
        super().__init__()
        if allow_flash and not FLASH_AVAILABLE:
            warnings.warn(
                "FlashAttention is not available. For optimal speed, "
                "consider installing torch >= 2.0 or flash-attn.",
                stacklevel=2,
            )
        self.enable_flash = allow_flash and FLASH_AVAILABLE
        if FLASH_AVAILABLE:
            torch.backends.cuda.enable_flash_sdp(allow_flash)

    def forward(self, q, k, v, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.enable_flash and q.device.type == "cuda":
            if FLASH_AVAILABLE:
                args = [x.half().contiguous() for x in [q, k, v]]
                v = F.scaled_dot_product_attention(*args, attn_mask=mask).to(q.dtype)
                return v if mask is None else v.nan_to_num()
        elif FLASH_AVAILABLE:
            args = [x.contiguous() for x in [q, k, v]]
            v = F.scaled_dot_product_attention(*args, attn_mask=mask)
            return v if mask is None else v.nan_to_num()
        else:
            s = q.shape[-1] ** -0.5
            sim = torch.einsum("...id,...jd->...ij", q, k) * s
            if mask is not None:
                sim.masked_fill(~mask, -float("inf"))
            attn = F.softmax(sim, -1)
            return torch.einsum("...ij,...jd->...id", attn, v)


class SelfBlock(nn.Module):
    def __init__(
        self, embed_dim: int, num_heads: int, flash: bool = False, bias: bool = True
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        assert self.embed_dim % num_heads == 0
        self.head_dim = self.embed_dim // num_heads
        self.Wqkv = nn.Linear(embed_dim, 3 * embed_dim, bias=bias)
        self.inner_attn = Attention(flash)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.ffn = nn.Sequential(
            nn.Linear(2 * embed_dim, 2 * embed_dim),
            nn.LayerNorm(2 * embed_dim, elementwise_affine=True),
            nn.GELU(),
            nn.Linear(2 * embed_dim, embed_dim),
        )

    def forward(
        self,
        x: torch.Tensor,
        encoding: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        qkv = self.Wqkv(x)
        qkv = qkv.unflatten(-1, (self.num_heads, -1, 3)).transpose(1, 2)
        q, k, v = qkv[..., 0], qkv[..., 1], qkv[..., 2]
        q = apply_cached_rotary_emb(encoding, q)
        k = apply_cached_rotary_emb(encoding, k)
        context = self.inner_attn(q, k, v, mask=mask)
        message = self.out_proj(context.transpose(1, 2).flatten(start_dim=-2))
        return x + self.ffn(torch.cat([x, message], -1))


class CrossBlock(nn.Module):
    def __init__(
        self, embed_dim: int, num_heads: int, flash: bool = False, bias: bool = True
    ) -> None:
        super().__init__()
        self.heads = num_heads
        dim_head = embed_dim // num_heads
        self.scale = dim_head ** -0.5
        inner_dim = dim_head * num_heads
        self.to_qk = nn.Linear(embed_dim, inner_dim, bias=bias)
        self.to_v = nn.Linear(embed_dim, inner_dim, bias=bias)
        self.to_out = nn.Linear(inner_dim, embed_dim, bias=bias)
        self.ffn = nn.Sequential(
            nn.Linear(2 * embed_dim, 2 * embed_dim),
            nn.LayerNorm(2 * embed_dim, elementwise_affine=True),
            nn.GELU(),
            nn.Linear(2 * embed_dim, embed_dim),
        )
        if flash and FLASH_AVAILABLE:
            self.flash = Attention(True)
        else:
            self.flash = None

    def map_(self, func: Callable, x0: torch.Tensor, x1: torch.Tensor):
        return func(x0), func(x1)

    def forward(
        self, x0: torch.Tensor, x1: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> List[torch.Tensor]:
        qk0, qk1 = self.map_(self.to_qk, x0, x1)
        v0, v1 = self.map_(self.to_v, x0, x1)
        qk0, qk1, v0, v1 = map(
            lambda t: t.unflatten(-1, (self.heads, -1)).transpose(1, 2),
            (qk0, qk1, v0, v1),
        )
        if self.flash is not None and qk0.device.type == "cuda":
            m0 = self.flash(qk0, qk1, v1, mask)
            m1 = self.flash(
                qk1, qk0, v0,
                mask.transpose(-1, -2) if mask is not None else None,
            )
        else:
            qk0, qk1 = qk0 * self.scale ** 0.5, qk1 * self.scale ** 0.5
            sim = torch.einsum("bhid, bhjd -> bhij", qk0, qk1)
            if mask is not None:
                sim = sim.masked_fill(~mask, -float("inf"))
            attn01 = F.softmax(sim, dim=-1)
            attn10 = F.softmax(sim.transpose(-2, -1).contiguous(), dim=-1)
            m0 = torch.einsum("bhij, bhjd -> bhid", attn01, v1)
            m1 = torch.einsum("bhji, bhjd -> bhid", attn10.transpose(-2, -1), v0)
            if mask is not None:
                m0, m1 = m0.nan_to_num(), m1.nan_to_num()
        m0, m1 = self.map_(
            lambda t: t.transpose(1, 2).flatten(start_dim=-2), m0, m1
        )
        m0, m1 = self.map_(self.to_out, m0, m1)
        x0 = x0 + self.ffn(torch.cat([x0, m0], -1))
        x1 = x1 + self.ffn(torch.cat([x1, m1], -1))
        return x0, x1


class TransformerLayer(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.self_attn = SelfBlock(*args, **kwargs)
        self.cross_attn = CrossBlock(*args, **kwargs)

    def forward(
        self, desc0, desc1, encoding0, encoding1,
        mask0: Optional[torch.Tensor] = None,
        mask1: Optional[torch.Tensor] = None,
    ):
        if mask0 is not None and mask1 is not None:
            return self.masked_forward(
                desc0, desc1, encoding0, encoding1, mask0, mask1
            )
        else:
            desc0 = self.self_attn(desc0, encoding0)
            desc1 = self.self_attn(desc1, encoding1)
            return self.cross_attn(desc0, desc1)

    def masked_forward(self, desc0, desc1, encoding0, encoding1, mask0, mask1):
        mask = mask0 & mask1.transpose(-1, -2)
        mask0 = mask0 & mask0.transpose(-1, -2)
        mask1 = mask1 & mask1.transpose(-1, -2)
        desc0 = self.self_attn(desc0, encoding0, mask0)
        desc1 = self.self_attn(desc1, encoding1, mask1)
        return self.cross_attn(desc0, desc1, mask)


def sigmoid_log_double_softmax(
    sim: torch.Tensor, z0: torch.Tensor, z1: torch.Tensor
) -> torch.Tensor:
    b, m, n = sim.shape
    certainties = F.logsigmoid(z0) + F.logsigmoid(z1).transpose(1, 2)
    scores0 = F.log_softmax(sim, 2)
    scores1 = F.log_softmax(
        sim.transpose(-1, -2).contiguous(), 2
    ).transpose(-1, -2)
    scores = sim.new_full((b, m + 1, n + 1), 0)
    scores[:, :m, :n] = scores0 + scores1 + certainties
    scores[:, :-1, -1] = F.logsigmoid(-z0.squeeze(-1))
    scores[:, -1, :-1] = F.logsigmoid(-z1.squeeze(-1))
    return scores


class MatchAssignment(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim
        self.matchability = nn.Linear(dim, 1, bias=True)
        self.final_proj = nn.Linear(dim, dim, bias=True)

    def forward(self, desc0: torch.Tensor, desc1: torch.Tensor):
        mdesc0, mdesc1 = self.final_proj(desc0), self.final_proj(desc1)
        _, _, d = mdesc0.shape
        mdesc0, mdesc1 = mdesc0 / d ** 0.25, mdesc1 / d ** 0.25
        sim = torch.einsum("bmd,bnd->bmn", mdesc0, mdesc1)
        z0 = self.matchability(desc0)
        z1 = self.matchability(desc1)
        scores = sigmoid_log_double_softmax(sim, z0, z1)
        return scores, sim

    def get_matchability(self, desc: torch.Tensor):
        return torch.sigmoid(self.matchability(desc)).squeeze(-1)


def filter_matches(scores: torch.Tensor, th: float):
    max0, max1 = scores[:, :-1, :-1].max(2), scores[:, :-1, :-1].max(1)
    m0, m1 = max0.indices, max1.indices
    indices0 = torch.arange(m0.shape[1], device=m0.device)[None]
    indices1 = torch.arange(m1.shape[1], device=m1.device)[None]
    mutual0 = indices0 == m1.gather(1, m0)
    mutual1 = indices1 == m0.gather(1, m1)
    max0_exp = max0.values.exp()
    zero = max0_exp.new_tensor(0)
    mscores0 = torch.where(mutual0, max0_exp, zero)
    mscores1 = torch.where(mutual1, mscores0.gather(1, m1), zero)
    valid0 = mutual0 & (mscores0 > th)
    valid1 = mutual1 & valid0.gather(1, m1)
    m0 = torch.where(valid0, m0, -1)
    m1 = torch.where(valid1, m1, -1)
    return m0, m1, mscores0, mscores1


# ──────────────────────────────────────────────
# LightGlue3DPE
# ──────────────────────────────────────────────

class LightGlue3DPE(nn.Module):
    """LightGlue with 3D Positional Encoding.

    depth map에서 keypoint의 실제 3D 좌표를 계산하여
    Fourier positional encoding에 (X, Y, Z)를 입력합니다.
    """

    default_conf = {
        "name": "lightglue_3dpe",
        "input_dim": 256,
        "add_scale_ori": False,
        "descriptor_dim": 256,
        "n_layers": 9,
        "num_heads": 4,
        "flash": False,
        "mp": False,
        "depth_confidence": -1,
        "width_confidence": -1,
        "filter_threshold": 0.0,
        "checkpointed": False,
        "weights": None,
        "weights_from_version": "v0.1_arxiv",
        "loss": {
            "gamma": 1.0,
            "fn": "nll",
            "nll_balancing": 0.5,
        },
        # ── 3D PE 전용 설정 ──
        "clip_start": 0.1,
        "clip_end": 1000.0,
        "fx": 8001.39,
        "fy": 8001.39,
        "cx": 2880.5,
        "cy": 2880.5,
        "orig_image_size": 5761,
    }

    required_data_keys = [
        "keypoints0", "keypoints1", "descriptors0", "descriptors1",
    ]

    url = "https://github.com/cvg/LightGlue/releases/download/{}/{}_lightglue.pth"

    def __init__(self, conf) -> None:
        super().__init__()
        self.conf = conf = OmegaConf.merge(self.default_conf, conf)
        if conf.input_dim != conf.descriptor_dim:
            self.input_proj = nn.Linear(
                conf.input_dim, conf.descriptor_dim, bias=True
            )
        else:
            self.input_proj = nn.Identity()

        head_dim = conf.descriptor_dim // conf.num_heads
        # ── 핵심 변경: M=3 (3D 좌표) ──
        self.posenc = LearnableFourierPositionalEncoding(
            3 + 2 * conf.add_scale_ori, head_dim, head_dim
        )

        h, n, d = conf.num_heads, conf.n_layers, conf.descriptor_dim

        self.transformers = nn.ModuleList(
            [TransformerLayer(d, h, conf.flash) for _ in range(n)]
        )
        self.log_assignment = nn.ModuleList(
            [MatchAssignment(d) for _ in range(n)]
        )
        self.token_confidence = nn.ModuleList(
            [TokenConfidence(d) for _ in range(n - 1)]
        )

        self.loss_fn = NLLLoss(conf.loss)

        # Load pretrained weights (posenc은 shape 불일치로 skip됨)
        state_dict = None
        if conf.weights is not None:
            if Path(conf.weights).exists():
                state_dict = torch.load(conf.weights, map_location="cpu")
            elif (Path(DATA_PATH) / conf.weights).exists():
                state_dict = torch.load(
                    str(Path(DATA_PATH) / conf.weights), map_location="cpu"
                )
            else:
                fname = (
                    f"{conf.weights}_{conf.weights_from_version}".replace(
                        ".", "-"
                    )
                    + ".pth"
                )
                state_dict = torch.hub.load_state_dict_from_url(
                    self.url.format(conf.weights_from_version, conf.weights),
                    file_name=fname,
                )

        if state_dict:
            for i in range(self.conf.n_layers):
                pattern = f"self_attn.{i}", f"transformers.{i}.self_attn"
                state_dict = {
                    k.replace(*pattern): v for k, v in state_dict.items()
                }
                pattern = f"cross_attn.{i}", f"transformers.{i}.cross_attn"
                state_dict = {
                    k.replace(*pattern): v for k, v in state_dict.items()
                }
            self.load_state_dict(state_dict, strict=False)

        self.register_buffer(
            "confidence_thresholds",
            torch.Tensor(
                [
                    self.confidence_threshold(i)
                    for i in range(self.conf.n_layers)
                ]
            ),
        )

    def _get_3d_kpts(self, kpts, depth_map, image_size):
        """keypoints를 3D 좌표로 변환 후 정규화.

        Args:
            kpts: (B, N, 2) pixel coordinates
            depth_map: (B, 1, H, W)
            image_size: (B, 2) — (H, W)
        Returns:
            kpts_3d_norm: (B, N, 3) normalized 3D coordinates
        """
        # 현재 이미지 해상도 (정사각형 가정)
        cur_size = image_size[0, 0].item()

        depth_vals, valid = sample_depth_at_kpts(kpts, depth_map)

        kpts_3d = kpts_to_3d(
            kpts, depth_vals, valid,
            fx=self.conf.fx,
            fy=self.conf.fy,
            cx=self.conf.cx,
            cy=self.conf.cy,
            clip_start=self.conf.clip_start,
            clip_end=self.conf.clip_end,
            image_size=cur_size,
            orig_image_size=self.conf.orig_image_size,
        )

        kpts_3d_norm = normalize_keypoints_3d(kpts_3d, valid)
        return kpts_3d_norm

    def compile(self, mode="reduce-overhead"):
        if self.conf.width_confidence != -1:
            warnings.warn(
                "Point pruning is partially disabled for compiled forward.",
                stacklevel=2,
            )
        for i in range(self.conf.n_layers):
            self.transformers[i] = torch.compile(
                self.transformers[i], mode=mode, fullgraph=True
            )

    def forward(self, data: dict) -> dict:
        for key in self.required_data_keys:
            assert key in data, f"Missing key {key} in data"

        kpts0, kpts1 = data["keypoints0"], data["keypoints1"]
        b, m, _ = kpts0.shape
        b, n, _ = kpts1.shape
        device = kpts0.device

        # ── 3D Position Encoding ──
        assert "view0" in data and "view1" in data, \
            "LightGlue3DPE requires view0/view1 with depth maps"
        size0 = data["view0"].get("image_size")
        size1 = data["view1"].get("image_size")
        depth0 = data["view0"]["image"]  # (B, 1, H, W)
        depth1 = data["view1"]["image"]  # (B, 1, H, W)

        kpts0_3d = self._get_3d_kpts(kpts0, depth0, size0)  # (B, N, 3)
        kpts1_3d = self._get_3d_kpts(kpts1, depth1, size1)  # (B, N, 3)

        if self.conf.add_scale_ori:
            sc0, o0 = data["scales0"], data["oris0"]
            sc1, o1 = data["scales1"], data["oris1"]
            kpts0_3d = torch.cat(
                [kpts0_3d,
                 sc0 if sc0.dim() == 3 else sc0[..., None],
                 o0 if o0.dim() == 3 else o0[..., None]],
                -1,
            )
            kpts1_3d = torch.cat(
                [kpts1_3d,
                 sc1 if sc1.dim() == 3 else sc1[..., None],
                 o1 if o1.dim() == 3 else o1[..., None]],
                -1,
            )

        desc0 = data["descriptors0"].contiguous()
        desc1 = data["descriptors1"].contiguous()

        assert desc0.shape[-1] == self.conf.input_dim
        assert desc1.shape[-1] == self.conf.input_dim
        if torch.is_autocast_enabled():
            desc0 = desc0.half()
            desc1 = desc1.half()
        desc0 = self.input_proj(desc0)
        desc1 = self.input_proj(desc1)

        # cache positional embeddings (3D)
        encoding0 = self.posenc(kpts0_3d)
        encoding1 = self.posenc(kpts1_3d)

        # GNN + final_proj + assignment
        do_early_stop = self.conf.depth_confidence > 0 and not self.training
        do_point_pruning = self.conf.width_confidence > 0 and not self.training

        all_desc0, all_desc1 = [], []

        if do_point_pruning:
            ind0 = torch.arange(0, m, device=device)[None]
            ind1 = torch.arange(0, n, device=device)[None]
            prune0 = torch.ones_like(ind0)
            prune1 = torch.ones_like(ind1)
        token0, token1 = None, None

        for i in range(self.conf.n_layers):
            if self.conf.checkpointed and self.training:
                desc0, desc1 = torch.utils.checkpoint.checkpoint(
                    self.transformers[i],
                    desc0, desc1, encoding0, encoding1,
                    use_reentrant=False,
                )
            else:
                desc0, desc1 = self.transformers[i](
                    desc0, desc1, encoding0, encoding1
                )
            if self.training or i == self.conf.n_layers - 1:
                all_desc0.append(desc0)
                all_desc1.append(desc1)
                continue

            if do_early_stop:
                assert b == 1
                token0, token1 = self.token_confidence[i](desc0, desc1)
                if self.check_if_stop(
                    token0[..., :m, :], token1[..., :n, :], i, m + n
                ):
                    break
            if do_point_pruning:
                assert b == 1
                scores0 = self.log_assignment[i].get_matchability(desc0)
                prunemask0 = self.get_pruning_mask(token0, scores0, i)
                keep0 = torch.where(prunemask0)[1]
                ind0 = ind0.index_select(1, keep0)
                desc0 = desc0.index_select(1, keep0)
                encoding0 = encoding0.index_select(-2, keep0)
                prune0[:, ind0] += 1
                scores1 = self.log_assignment[i].get_matchability(desc1)
                prunemask1 = self.get_pruning_mask(token1, scores1, i)
                keep1 = torch.where(prunemask1)[1]
                ind1 = ind1.index_select(1, keep1)
                desc1 = desc1.index_select(1, keep1)
                encoding1 = encoding1.index_select(-2, keep1)
                prune1[:, ind1] += 1

        desc0, desc1 = desc0[..., :m, :], desc1[..., :n, :]
        scores, _ = self.log_assignment[i](desc0, desc1)
        m0, m1, mscores0, mscores1 = filter_matches(
            scores, self.conf.filter_threshold
        )

        if do_point_pruning:
            m0_ = torch.full((b, m), -1, device=m0.device, dtype=m0.dtype)
            m1_ = torch.full((b, n), -1, device=m1.device, dtype=m1.dtype)
            m0_[:, ind0] = torch.where(
                m0 == -1, -1, ind1.gather(1, m0.clamp(min=0))
            )
            m1_[:, ind1] = torch.where(
                m1 == -1, -1, ind0.gather(1, m1.clamp(min=0))
            )
            mscores0_ = torch.zeros((b, m), device=mscores0.device)
            mscores1_ = torch.zeros((b, n), device=mscores1.device)
            mscores0_[:, ind0] = mscores0
            mscores1_[:, ind1] = mscores1
            m0, m1, mscores0, mscores1 = m0_, m1_, mscores0_, mscores1_
        else:
            prune0 = torch.ones_like(mscores0) * self.conf.n_layers
            prune1 = torch.ones_like(mscores1) * self.conf.n_layers

        pred = {
            "matches0": m0,
            "matches1": m1,
            "matching_scores0": mscores0,
            "matching_scores1": mscores1,
            "ref_descriptors0": torch.stack(all_desc0, 1),
            "ref_descriptors1": torch.stack(all_desc1, 1),
            "log_assignment": scores,
            "prune0": prune0,
            "prune1": prune1,
        }
        return pred

    def confidence_threshold(self, layer_index: int) -> float:
        threshold = 0.8 + 0.1 * np.exp(
            -4.0 * layer_index / self.conf.n_layers
        )
        return np.clip(threshold, 0, 1)

    def get_pruning_mask(
        self, confidences: torch.Tensor, scores: torch.Tensor,
        layer_index: int
    ) -> torch.Tensor:
        keep = scores > (1 - self.conf.width_confidence)
        if confidences is not None:
            keep |= confidences <= self.confidence_thresholds[layer_index]
        return keep

    def check_if_stop(
        self, confidences0, confidences1, layer_index, num_points
    ) -> torch.Tensor:
        confidences = torch.cat([confidences0, confidences1], -1)
        threshold = self.confidence_thresholds[layer_index]
        ratio_confident = (
            1.0 - (confidences < threshold).float().sum() / num_points
        )
        return ratio_confident > self.conf.depth_confidence

    def loss(self, pred, data):
        def loss_params(pred, i):
            la, _ = self.log_assignment[i](
                pred["ref_descriptors0"][:, i],
                pred["ref_descriptors1"][:, i],
            )
            return {"log_assignment": la}

        sum_weights = 1.0
        nll, gt_weights, loss_metrics = self.loss_fn(
            loss_params(pred, -1), data
        )
        N = pred["ref_descriptors0"].shape[1]
        losses = {
            "total": nll,
            "last": nll.clone().detach(),
            **loss_metrics,
        }

        if self.training:
            losses["confidence"] = 0.0

        losses["row_norm"] = (
            pred["log_assignment"].exp()[:, :-1].sum(2).mean(1)
        )
        for i in range(N - 1):
            params_i = loss_params(pred, i)
            nll, _, _ = self.loss_fn(params_i, data, weights=gt_weights)

            if self.conf.loss.gamma > 0.0:
                weight = self.conf.loss.gamma ** (N - i - 1)
            else:
                weight = i + 1
            sum_weights += weight
            losses["total"] = losses["total"] + nll * weight

            losses["confidence"] += self.token_confidence[i].loss(
                pred["ref_descriptors0"][:, i],
                pred["ref_descriptors1"][:, i],
                params_i["log_assignment"],
                pred["log_assignment"],
            ) / (N - 1)

            del params_i
        losses["total"] /= sum_weights

        if self.training:
            losses["total"] = losses["total"] + losses["confidence"]

        if not self.training:
            metrics = matcher_metrics(pred, data)
        else:
            metrics = {}
        return losses, metrics


__main_model__ = LightGlue3DPE
