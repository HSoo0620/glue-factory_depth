import torch
from ..base_model import BaseModel


class DepthGTMatcher(BaseModel):

    default_conf = {
        "gt_radius": 3,
    }

    required_data_keys = ["keypoints0", "keypoints1", "gt_matches"]

    def _init(self, conf):
        self.radius = conf.gt_radius

    def _forward(self, data):

        sp_kpts0 = data["keypoints0"]  # (B,N0,2)
        sp_kpts1 = data["keypoints1"]  # (B,N1,2)
        gt = data["gt_matches"]  # (B,M,4)

        B, N0, _ = sp_kpts0.shape
        N1 = sp_kpts1.shape[1]

        assignment = torch.zeros(
            (B, N0, N1),
            device=sp_kpts0.device
        )

        # matches0[b, i] = j means keypoint i in image0 matches keypoint j in image1
        # -1 = unmatched, -2 = ignore
        UNMATCHED = -1
        IGNORE = -2

        matches0 = torch.full((B, N0), UNMATCHED, device=sp_kpts0.device, dtype=torch.long)
        matches1 = torch.full((B, N1), UNMATCHED, device=sp_kpts0.device, dtype=torch.long)

        for b in range(B):

            gt0 = gt[b][:, :2]  # master
            gt1 = gt[b][:, 2:]  # input

            # Filter out padding rows (all zeros from pad_sequence)
            valid_mask = (gt0.abs().sum(-1) > 0) | (gt1.abs().sum(-1) > 0)
            gt0 = gt0[valid_mask]
            gt1 = gt1[valid_mask]

            k0 = sp_kpts0[b]
            k1 = sp_kpts1[b]

            if gt0.shape[0] == 0:
                # No GT matches: all keypoints are unmatched
                matches0[b] = UNMATCHED
                matches1[b] = UNMATCHED
                continue

            # distance from GT points to SuperPoint keypoints
            d0 = torch.cdist(gt0, k0)  # (M, N0)
            d1 = torch.cdist(gt1, k1)  # (M, N1)

            idx0 = d0.argmin(dim=1)  # nearest SP keypoint for each GT in view0
            idx1 = d1.argmin(dim=1)  # nearest SP keypoint for each GT in view1

            min0 = d0.min(dim=1).values
            min1 = d1.min(dim=1).values

            valid = (min0 < self.radius) & (min1 < self.radius)

            idx0_v = idx0[valid]
            idx1_v = idx1[valid]

            assignment[b, idx0_v, idx1_v] = 1

            # Build matches0/matches1
            matches0[b, idx0_v] = idx1_v
            matches1[b, idx1_v] = idx0_v

            # Keypoints not matched by any GT pair are "unmatched" (negative)
            # Compute which SP keypoints have no nearby GT point at all -> unmatched
            d0_min_per_kp = d0.min(dim=0).values  # (N0,) min dist to any GT
            d1_min_per_kp = d1.min(dim=0).values  # (N1,)

            neg_th = self.radius * 2  # keypoints far from any GT are unmatched
            neg0 = d0_min_per_kp > neg_th
            neg1 = d1_min_per_kp > neg_th

            matches0[b][neg0 & (matches0[b] == IGNORE)] = UNMATCHED
            matches1[b][neg1 & (matches1[b] == IGNORE)] = UNMATCHED

        # Keys WITHOUT gt_ prefix - two_view_pipeline adds the gt_ prefix
        return {
            "assignment": assignment,
            "matches0": matches0,
            "matches1": matches1,
            "matching_scores0": (matches0 > -1).float(),
            "matching_scores1": (matches1 > -1).float(),
        }

    def loss(self, pred, data):
        raise NotImplementedError
