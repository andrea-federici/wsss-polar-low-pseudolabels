import os
from typing import Tuple

import cv2
import numpy as np
from tqdm import tqdm

from src.train.helpers.adv_er_helper import load_accumulated_heatmap


# TODO: at the moment this function is only able to generate masks from the
# given heatmaps. It does not accumulate the heatmaps. So, it is currently
# only usable for the iteration 0 heatmaps.
def generate_masks_from_heatmaps(
    base_heatmaps_dir: str,
    mask_dir: str,
    mask_size: Tuple[int, int],  # (width, height)
    threshold: float,  # this must be the same as the one used during training
    iteration: int = 0,
):
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

        if len(heatmap.shape) > 2:
            heatmap = heatmap.squeeze(0)
        heatmap = heatmap.numpy()

        heatmap = cv2.resize(
            heatmap,
            mask_size,  # cv2.resize takes (width, height)
            interpolation=cv2.INTER_LINEAR,
        )

        # print(f'Path: {heatmap_path}')
        # if not heatmap:
        #     print('Heatmap is None')

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
