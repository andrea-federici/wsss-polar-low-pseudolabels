import os
from typing import Tuple

import cv2
import numpy as np
from tqdm import tqdm

from src.train.helpers.adv_er_helper import load_accumulated_heatmap


def generate_masks_from_heatmaps(
    base_heatmaps_dir: str,
    mask_dir: str,
    mask_size: Tuple[int, int],  # (width, height)
    threshold: float,  # this must be the same as the one used during training
    iteration: int = 0,
) -> None:
    """
    Generates binary masks from accumulated heatmaps by applying a threshold and
    resizing them.

    Heatmaps from all iterations up to `iteration` are loaded from
    `base_heatmaps_dir/iteration_<iteration>`, accumulated, resized to `mask_size`, and
    thresholded to produce binary masks, which are saved in `mask_dir`.

    Args:
        base_heatmaps_dir (str): Directory containing heatmaps.
        mask_dir (str): Directory to save generated masks.
        mask_size (Tuple[int, int]): Output mask size (width, height).
        threshold (float): Threshold for binarization.
        iteration (int, optional): The last iteration to accumulate heatmaps from.
            Defaults to 0.

    Returns:
        None: The function saves the generated masks as image files.
    """
    os.makedirs(mask_dir, exist_ok=True)
    heatmap_filenames = [
        f
        for f in os.listdir(os.path.join(base_heatmaps_dir, f"iteration_{iteration}"))
        if f.endswith(".pt")
    ]

    print(f"Loaded {len(heatmap_filenames)} heatmaps.")

    for filename in tqdm(heatmap_filenames, desc="Processing heatmaps"):
        img_path = filename.replace(
            ".pt", ""
        )  # TODO: This works now, but when I will fix the
        # naming of the heatmaps (.png.pt -> .pt), this will need to be changed and it
        # should be .replace(".pt", ".png").
        mask_path = os.path.join(mask_dir, filename.replace(".pt", ".png"))

        heatmap = load_accumulated_heatmap(
            base_heatmaps_dir, img_path, label=1, current_iteration=iteration + 1
        )

        if len(heatmap.shape) > 2:  # TODO: is this needed?
            heatmap = heatmap.squeeze(0)
        heatmap = heatmap.numpy()

        heatmap = cv2.resize(
            heatmap,
            mask_size,  # cv2.resize takes (width, height)
            interpolation=cv2.INTER_LINEAR,
        )

        binary_mask = (heatmap > threshold).astype(np.uint8)

        cv2.imwrite(mask_path, binary_mask * 255)


# In order to run this script, run the following command from the root directory of the
# project:
# PYTHONPATH=$(pwd) python3 tools/generate_masks.py
if __name__ == "__main__":
    base_heatmaps_dir = "out/heatmaps"
    mask_dir = "out/masks"
    mask_size = (500, 500)
    threshold = 0.75
    iteration = 5

    generate_masks_from_heatmaps(
        base_heatmaps_dir, mask_dir, mask_size, threshold, iteration
    )
