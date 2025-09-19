from typing import Tuple

import torch
import torch.nn.functional as F


def resize_heatmap(heatmap: torch.Tensor, target_size: Tuple[int, int]) -> torch.Tensor:
    """
    Resize a 2D heatmap (H, W) to target_size (H_out, W_out) using bilinear interpolation
    with align_corners=False, then normalizes to [0, 1]. Raises on invalid inputs.
    """
    resized_heatmap = F.interpolate(
        heatmap.unsqueeze(0).unsqueeze(0),
        size=target_size,
        mode="bilinear",
        align_corners=False,
    ).squeeze(0).squeeze(0)
    if not isinstance(heatmap, torch.Tensor):
        raise TypeError(f"Heatmap must be a torch.Tensor, got {type(heatmap)}")
    if heatmap.ndim != 2:
        raise ValueError(f"Heatmap must be 2D (H, W), got {heatmap.ndim} dimensions")
    if (
        not isinstance(target_size, (tuple, list)) or
        len(target_size) != 2 or
        not all(isinstance(x, int) and x > 0 for x in target_size)
    ):
        raise ValueError(
            f"Target size must be a tuple or list of two positive ints, got {target_size}"
        )
    if not torch.is_floating_point(heatmap):
        heatmap = heatmap.float()

    hmin, hmax = resized_heatmap.min(), resized_heatmap.max()
    norm_heatmap = (resized_heatmap - hmin) / (hmax - hmin + 1e-6)
    return norm_heatmap