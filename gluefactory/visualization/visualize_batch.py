import numpy as np
import pandas as pd
import torch
from pathlib import Path

from ..utils.tensor import batch_to_device
from .viz2d import cm_RdGn, plot_heatmaps, plot_image_grid, plot_keypoints, plot_matches


def make_match_figures(pred_, data_, n_pairs=2):
    # print first n pairs in batch
    if "0to1" in pred_.keys():
        pred_ = pred_["0to1"]
    images, kpts, matches, mcolors = [], [], [], []
    heatmaps = []
    pred = batch_to_device(pred_, "cpu", non_blocking=False)
    data = batch_to_device(data_, "cpu", non_blocking=False)

    view0, view1 = data["view0"], data["view1"]

    n_pairs = min(n_pairs, view0["image"].shape[0])
    assert view0["image"].shape[0] >= n_pairs

    kp0, kp1 = pred["keypoints0"], pred["keypoints1"]
    m0 = pred["matches0"]
    gtm0 = pred["gt_matches0"]

    for i in range(n_pairs):
        valid = (m0[i] > -1) & (gtm0[i] >= -1)
        kpm0, kpm1 = kp0[i][valid].numpy(), kp1[i][m0[i][valid]].numpy()
        images.append(
            [view0["image"][i].permute(1, 2, 0), view1["image"][i].permute(1, 2, 0)]
        )
        kpts.append([kp0[i], kp1[i]])
        matches.append((kpm0, kpm1))

        correct = gtm0[i][valid] == m0[i][valid]

        if "heatmap0" in pred.keys():
            heatmaps.append(
                [
                    torch.sigmoid(pred["heatmap0"][i, 0]),
                    torch.sigmoid(pred["heatmap1"][i, 0]),
                ]
            )
        elif "depth" in view0.keys() and view0["depth"] is not None:
            heatmaps.append([view0["depth"][i], view1["depth"][i]])

        mcolors.append(cm_RdGn(correct).tolist())

    fig, axes = plot_image_grid(images, return_fig=True, set_lim=True)
    if len(heatmaps) > 0:
        [plot_heatmaps(heatmaps[i], axes=axes[i], a=1.0) for i in range(n_pairs)]
    [plot_keypoints(kpts[i], axes=axes[i], colors="royalblue") for i in range(n_pairs)]
    [
        plot_matches(*matches[i], color=mcolors[i], axes=axes[i], a=0.5, lw=1.0, ps=0.0)
        for i in range(n_pairs)
    ]

    return {"matching": fig}


def _classify_match_colors(kpm0, kpm1, csv_path, gt_radius=3):
    """
    매칭 결과를 4가지 색상으로 분류.
        하늘색(skyblue):  정답 pair (GT 정의 + 비가림 + 정확 매칭)
        보라색(purple):   GT에 정의됐지만 매칭 틀림
        녹색(limegreen):  정답 매칭이지만 GT에서 가려짐(occluded)
        빨간색(red):      GT에 포인트 정의 없는데 매칭됨
    """
    n = len(kpm0)
    colors = []

    if csv_path is None or not Path(csv_path).exists():
        return [[1.0, 0.0, 0.0]] * n

    corr = pd.read_csv(csv_path)
    all_master_xy = corr[["master_x", "master_y"]].values.astype(np.float32)
    all_input_xy = corr[["input_x", "input_y"]].values.astype(np.float32)
    all_occluded = corr["occluded"].values.astype(bool)

    for i in range(n):
        kp0_pt = kpm0[i]
        kp1_pt = kpm1[i]

        dists_master = np.linalg.norm(all_master_xy - kp0_pt, axis=1)
        nearest_idx = np.argmin(dists_master)
        nearest_dist = dists_master[nearest_idx]

        if nearest_dist < gt_radius:
            gt_input_pt = all_input_xy[nearest_idx]
            is_occluded = all_occluded[nearest_idx]
            match_dist = np.linalg.norm(kp1_pt - gt_input_pt)

            if match_dist < gt_radius:
                if is_occluded:
                    colors.append([0.2, 0.8, 0.2])   # 녹색
                else:
                    colors.append([0.53, 0.81, 0.98]) # 하늘색
            else:
                colors.append([0.5, 0.0, 0.5])        # 보라색
        else:
            colors.append([1.0, 0.0, 0.0])            # 빨간색

    return colors


def make_match_figures_depth(pred_, data_, n_pairs=2):
    """4색 시각화: 하늘색(정답) / 보라색(오답) / 녹색(가림정답) / 빨간색(GT미정의)"""
    if "0to1" in pred_.keys():
        pred_ = pred_["0to1"]
    images, kpts, matches, mcolors = [], [], [], []
    pred = batch_to_device(pred_, "cpu", non_blocking=False)
    data = batch_to_device(data_, "cpu", non_blocking=False)

    view0, view1 = data["view0"], data["view1"]
    csv_paths = data.get("csv_path", [None] * view0["image"].shape[0])

    n_pairs = min(n_pairs, view0["image"].shape[0])

    kp0, kp1 = pred["keypoints0"], pred["keypoints1"]
    m0 = pred["matches0"]
    gtm0 = pred["gt_matches0"]

    for i in range(n_pairs):
        valid = (m0[i] > -1) & (gtm0[i] >= -1)
        kpm0, kpm1 = kp0[i][valid].numpy(), kp1[i][m0[i][valid]].numpy()
        images.append(
            [view0["image"][i].permute(1, 2, 0), view1["image"][i].permute(1, 2, 0)]
        )
        kpts.append([kp0[i], kp1[i]])
        matches.append((kpm0, kpm1))
        mcolors.append(_classify_match_colors(kpm0, kpm1, csv_paths[i]))

    fig, axes = plot_image_grid(images, return_fig=True, set_lim=True)
    [plot_keypoints(kpts[i], axes=axes[i], colors="royalblue") for i in range(n_pairs)]
    [
        plot_matches(*matches[i], color=mcolors[i], axes=axes[i], a=0.5, lw=1.0, ps=0.0)
        for i in range(n_pairs)
    ]

    return {"matching": fig}
