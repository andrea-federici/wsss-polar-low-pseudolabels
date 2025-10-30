import os
from collections import deque
from typing import Tuple, Union

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from src.data.adversarial_erasing_io import \
    generate_multiclass_mask_from_heatmaps
from src.utils.constants import PYTORCH_EXTENSION


# TODO: this does not work really well if there are black pixels next to black patch (hurricane center usually)
def _get_blind_spot_mask(orig_img_path: str, low_thresh: int = 10) -> np.ndarray:
    """
    Reads an RGB (or RGBA) image from `orig_img_path` and returns a 2D boolean array
    of shape (H, W), where True = pixel belongs to a border-connected black patch
    (“blind spot”), and False = pixel is “good data.”

    A pixel is considered “black” if all channels R,G,B are <= low_thresh.
    We then flood-fill from any black pixel on the image's boundary, collecting
    all 8-connected black pixels. This way, any black patch that does *not* touch
    the border is ignored (so small dark cloud holes won't be removed).

    Args:
        orig_img_path (str):
            Path to the PNG/JPEG satellite image. If RGBA-encoded, the alpha is dropped.
        low_thresh (int, optional):
            All pixels whose R, G, and B channels are ≤ low_thresh will be considered
            “black.” Defaults to 10.

    Returns:
        blind_mask (np.ndarray of dtype bool):
            A 2D array of shape (H, W). True = that pixel is part of the blind spot.
    """
    # 1) Load the image (RGB or RGBA)
    orig = cv2.imread(orig_img_path, cv2.IMREAD_UNCHANGED)
    if orig is None:
        raise FileNotFoundError(f"Could not load image at '{orig_img_path}'")

    # If RGBA, drop alpha
    if orig.ndim == 3 and orig.shape[2] == 4:
        orig = orig[..., :3]

    # 2) Build a binary “black‐pixel” mask: True where (R, G, B) <= low_thresh
    #    This picks up pure‐black or nearly black areas.
    black_pixels = np.all(orig <= low_thresh, axis=2)  # shape = (H, W), dtype=bool

    # --- Morphological opening to remove thin bridges ---
    bin_img = (black_pixels.astype(np.uint8)) * 255
    k = 21  # kernel size (odd number); increase to 5 or 7 for thicker bridge removal
    kernel = np.ones((k, k), dtype=np.uint8)
    opened = cv2.morphologyEx(bin_img, cv2.MORPH_OPEN, kernel)
    black_pixels = opened > 0
    # ---------------------------------------------------

    H, W = black_pixels.shape

    # 3) Flood‐fill from any black pixel on the boundary
    #    We'll do a simple BFS: start from every border location that is black,
    #    then propagate to any 8-connected black neighbor.
    blind_mask = np.zeros((H, W), dtype=bool)
    visited = np.zeros((H, W), dtype=bool)
    q = deque()

    # Enqueue all boundary black‐pixel locations:
    #   top row, bottom row, left column, right column
    for x in range(W):
        if black_pixels[0, x] and not visited[0, x]:
            visited[0, x] = True
            blind_mask[0, x] = True
            q.append((0, x))
        if black_pixels[H - 1, x] and not visited[H - 1, x]:
            visited[H - 1, x] = True
            blind_mask[H - 1, x] = True
            q.append((H - 1, x))

    for y in range(H):
        if black_pixels[y, 0] and not visited[y, 0]:
            visited[y, 0] = True
            blind_mask[y, 0] = True
            q.append((y, 0))
        if black_pixels[y, W - 1] and not visited[y, W - 1]:
            visited[y, W - 1] = True
            blind_mask[y, W - 1] = True
            q.append((y, W - 1))

    # 4) BFS over 8‐connected neighbors
    directions = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

    while q:
        y, x = q.popleft()
        for dy, dx in directions:
            ny, nx = y + dy, x + dx
            if ny < 0 or ny >= H or nx < 0 or nx >= W:
                continue
            if visited[ny, nx]:
                continue
            # Only propagate into black pixels
            if not black_pixels[ny, nx]:
                continue

            visited[ny, nx] = True
            blind_mask[ny, nx] = True
            q.append((ny, nx))

    return blind_mask


# TODO: update docstring
def generate_masks(
    *,
    base_heatmaps_dir: str,
    mask_dir: str,
    mask_size: Tuple[int, int],  # (width, height)
    threshold: float,  # Must match the one used in load_multilabel_mask
    type: str,  # "binary" or "multiclass"
    iteration: int = 0,
    remove_background: bool = False,
    vis: bool = False,
    envelope_start: int = 2,
    envelope_scale: float = 0.1,
) -> None:
    """
    Generates multi-label masks from accumulated heatmaps.

    For each image, heatmaps from all iterations up to `iteration` are accumulated
    (using load_multilabel_mask) to produce a multi-label mask where each pixel is
    assigned the iteration index (starting from 1) when it first became active.

    Args:
        base_heatmaps_dir (str): Directory containing heatmaps.
        mask_dir (str): Directory to save generated masks.
        mask_size (Tuple[int, int]): Output mask size (width, height).
        threshold (float): Threshold for activation.
        iteration (int, optional): Last iteration to accumulate heatmaps from.
            Defaults to 0.
        envelope_start (int, optional): Iteration at which the area-targeted
            envelope begins restricting new pixels. Defaults to 2.
        envelope_scale (float, optional): Scale factor used to dilate the
            envelope relative to the current mask area. Defaults to 0.1.

    Returns:
        None. The function saves the generated masks as image files.
    """
    os.makedirs(mask_dir, exist_ok=True)
    iteration_dir = os.path.join(base_heatmaps_dir, f"iteration_{iteration}")
    heatmap_filenames = [
        f for f in sorted(os.listdir(iteration_dir)) if f.endswith(PYTORCH_EXTENSION)
    ]
    print(f"Loaded {len(heatmap_filenames)} heatmaps.")

    for filename in tqdm(
        heatmap_filenames,
        desc=f"Processing {type} heatmaps, iteration {iteration}, vis={vis}",
    ):
        # Assume the image name is encoded in the filename (without extension)
        img_name = os.path.splitext(os.path.basename(filename))[0]
        mask_path = os.path.join(mask_dir, f"{img_name}.png")

        if type == "binary":
            # Generate mask iteratively while restricting new activations with the
            # area-targeted envelope.
            multi_label_mask = generate_multiclass_mask_from_heatmaps(
                base_heatmaps_dir,
                img_name,
                label=1,
                iteration=iteration,
                threshold=threshold,
                envelope_start=envelope_start,
                envelope_scale=envelope_scale,
            )
            heatmap = (
                multi_label_mask.float().unsqueeze(0).unsqueeze(0)
            )  # shape: [1, 1, H, W]
            resized_heatmap = F.interpolate(
                heatmap,
                size=mask_size,
                mode="nearest",
            )
            mask = (resized_heatmap.squeeze() > 0).to(torch.uint8).cpu().numpy()
            if vis:
                mask = mask * 255

        elif type == "multiclass":
            # Generate the multi-label mask using accumulated heatmaps.
            multi_label_mask = generate_multiclass_mask_from_heatmaps(
                base_heatmaps_dir,
                img_name,
                label=1,
                iteration=iteration,
                threshold=threshold,
                envelope_start=envelope_start,
                envelope_scale=envelope_scale,
            )
            # multi_label_mask = generate_multiclass_mask_from_masks(
            #     base_heatmaps_dir, img_name, max_iteration=iteration
            # )

            # multi_label_mask = (multi_label_mask / (iteration + 1)) * 255

            # cv2.imwrite(mask_path, multi_label_mask.cpu().numpy().astype(np.uint8))
            # continue

            # Remove extra dimensions if necessary.
            if len(multi_label_mask.shape) > 2:
                multi_label_mask = multi_label_mask.squeeze(0)

            # Assume `heatmap` is a 2D tensor: shape [H, W]
            heatmap = (
                multi_label_mask.float().unsqueeze(0).unsqueeze(0)
            )  # shape: [1, 1, H, W]
            resized_heatmap = F.interpolate(
                heatmap,
                size=mask_size,
                mode="nearest",  # <--- this avoids interpolation artifacts
            )

            resized_heatmap = (
                resized_heatmap.squeeze()
            )  # shape: [new_height, new_width]

            # Convert the multi-label mask to a NumPy array.
            mask = resized_heatmap.to(torch.int).cpu().numpy()

            # TODO: this is hard-coded now. It is fine since the only dataset that
            # uses it is the polar_lows dataset, but the path should still be taken as
            # argument for clarity purposes.
            if remove_background:
                possible_paths = [
                    os.path.join("data/polar_lows/train/pos", f"{img_name}.png"),
                ]

                img_path = None
                for p in possible_paths:
                    if os.path.exists(p):
                        img_path = p
                        break

                if img_path is None:
                    raise FileNotFoundError(
                        f"Corresponding image for '{img_name}' not found."
                    )

                img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
                if img is None:
                    raise FileNotFoundError(f"Failed to load image at {img_path}")

                blind_mask = _get_blind_spot_mask(img_path, low_thresh=0)
                # blind_mask is a boolean array of shape (orig_H, orig_W). But we must resize it
                # (nearest) to the same shape as mask_np, which we already chose to be (mask_H, mask_W).

                blind_mask_resized = cv2.resize(
                    blind_mask.astype(np.uint8),  # convert to uint8 (0/1)
                    mask_size,  # notice: cv2 expects (width, height)
                    interpolation=cv2.INTER_NEAREST,
                ).astype(bool)

                # Now zero‐out exactly those blind‐spot pixels in `mask_np`:
                mask[blind_mask_resized] = 0

            if vis:
                # (iteration + 1) is the maximum label value.
                mask = (mask / (iteration + 1)) * 255

        # Save the mask. Using uint8 to store as an image.
        cv2.imwrite(mask_path, mask.astype(np.uint8))


def generate_negative_masks(
    negative_images_dir: str,
    mask_dir: str,
    mask_size: Tuple[int, int],  # (width, height)
) -> None:
    """
    Generates all-zero masks for negative images.

    For each .png/.jpg/.jpeg image in `negative_images_dir`, creates an empty mask of the specified
    `mask_size` filled with zeros and saves it to `mask_dir` with the same base name, in the .png
    format.

    Args:
        negative_images_dir (str): Directory containing negative images.
        mask_dir (str): Directory to save generated zero masks.
        mask_size (Tuple[int, int]): Output mask size (width, height).
    """
    # Ensure output directory exists
    os.makedirs(mask_dir, exist_ok=True)

    # List all .png/.jpg/.jpeg files in the negative_images_dir
    image_filenames = [
        f
        for f in sorted(os.listdir(negative_images_dir))
        if os.path.splitext(f)[1].lower() in [".png", ".jpg", ".jpeg"]
    ]

    print(f"Generating all-zero masks for {len(image_filenames)} negative images...")

    for filename in image_filenames:
        img_name, _ = os.path.splitext(filename)
        mask_path = os.path.join(mask_dir, f"{img_name}.png")

        # Create an all-zero mask of the target size
        # mask_size is (width, height), numpy expects (height, width)
        zero_mask = np.zeros((mask_size[1], mask_size[0]), dtype=np.uint8)

        # Save the all-zero mask
        cv2.imwrite(mask_path, zero_mask)

    print("Negative masks generation complete.")


# In order to run this script, run the following command from the root directory of the
# project:
# PYTHONPATH=$(pwd) python3 src/post/masks.py
if __name__ == "__main__":
    base_heatmaps_dir = "local/heatmaps"
    mask_dir = "local/masks"
    mask_size = (512, 512)
    threshold = 0.7
    iteration = 5

    import os

    import cv2
    import numpy as np

    # vis
    generate_masks(
        base_heatmaps_dir=base_heatmaps_dir,
        mask_dir=mask_dir + "/vis",
        mask_size=mask_size,
        threshold=threshold,
        type="multiclass",
        iteration=iteration,
        remove_background=False,
        vis=True,
    )

    # no vis
    # no_vis_dir = mask_dir + "/no_vis"
    # generate_masks(
    #     base_heatmaps_dir=base_heatmaps_dir,
    #     mask_dir=no_vis_dir,
    #     mask_size=mask_size,
    #     threshold=threshold,
    #     type="multiclass",
    #     iteration=iteration,
    #     remove_background=False,
    #     vis=False,
    # )

    # generate_negative_masks(
    #     negative_images_dir="data/bus/train/neg",
    #     mask_dir=no_vis_dir,
    #     mask_size=mask_size,
    # )
