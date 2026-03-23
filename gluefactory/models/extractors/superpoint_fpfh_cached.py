"""
Precomputed FPFH descriptor를 pass-through하는 extractor.
데이터셋에서 이미 keypoints + descriptors가 로드되어 있으므로 그대로 반환.
"""

from ..base_model import BaseModel


class SuperPointFPFHCached(BaseModel):
    default_conf = {
        "descriptor_dim": 33,  # FPFH is 33-dim
        "trainable": False,
    }

    required_data_keys = ["keypoints", "keypoint_scores", "descriptors"]

    def _init(self, conf):
        pass  # no parameters

    def _forward(self, data):
        return {
            "keypoints": data["keypoints"],
            "keypoint_scores": data["keypoint_scores"],
            "descriptors": data["descriptors"],
        }

    def loss(self, pred, data):
        raise NotImplementedError
