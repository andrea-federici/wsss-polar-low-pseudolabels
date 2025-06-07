import os
from collections import deque
from typing import Tuple, Union

import cv2
import numpy as np
import torch
from tqdm import tqdm

from src.data.adversarial_erasing_io import (
    generate_multiclass_mask_from_heatmaps,
    generate_multiclass_mask_from_masks,
    load_accumulated_heatmap,
)


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
        img_path = filename.replace(".pt", ".png")
        mask_path = os.path.join(mask_dir, filename.replace(".pt", ".png"))

        heatmap = load_accumulated_heatmap(
            base_heatmaps_dir, img_path, label=1, iteration=iteration
        )

        if len(heatmap.shape) > 2:  # TODO: is this needed?
            heatmap = heatmap.squeeze(0)
        heatmap = heatmap.cpu().numpy()

        heatmap = cv2.resize(
            heatmap,
            mask_size,  # cv2.resize takes (width, height)
            interpolation=cv2.INTER_LINEAR,
        )

        binary_mask = (heatmap > threshold).astype(np.uint8)

        cv2.imwrite(mask_path, binary_mask * 255)


def fill_sparse_holes(
    mask_np: np.ndarray, kernel_size: Union[int, Tuple[int, int]] = 3
) -> np.ndarray:
    """
    Fills small “hole” pixels (zeros) inside a multi‐label mask by:
      1) Performing a morphological closing to detect which zeros are likely holes.
      2) Propagating a nonzero neighbor’s label into each hole pixel.
         (Explicitly skips any neighbor == 0.)

    Args:
        mask_np (np.ndarray):
            2D array of integer labels (0 = background, 1…N = classes).
        kernel_size (int or (h, w) tuple, optional):
            Size of the rectangular structuring element used for closing.
            Defaults to 3 (i.e. a 3×3 rectangle). If you pass an integer k,
            it will be interpreted as (k, k).

    Returns:
        np.ndarray:
            A new 2D array (same shape as `mask_np`) where tiny zero‐valued
            holes surrounded by nonzero pixels have been filled with the
            label of one of their nonzero neighbors. If no nonzero neighbor
            exists, the pixel remains 0.
    """
    # Ensure kernel_size is a tuple
    if isinstance(kernel_size, int):
        k_h = k_w = kernel_size
    else:
        k_h, k_w = kernel_size

    # 1) Create a binary mask of “valid” pixels (label > 0)
    binary = (mask_np > 0).astype(np.uint8)

    # 2) Morphological closing to fill tiny holes in 'binary'
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k_w, k_h))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)

    # 3) Identify hole pixels: zero in 'binary' but one in 'closed'
    hole_mask = (closed == 1) & (binary == 0)

    # If there are no holes, return a copy immediately
    if not hole_mask.any():
        return mask_np.copy()

    # 4) For each hole pixel, copy the label from any 8‐connected neighbor > 0
    filled = mask_np.copy()
    h, w = filled.shape
    ys, xs = np.nonzero(hole_mask)

    for y, x in zip(ys, xs):
        filled_label = 0
        # Look through 8 neighbors in arbitrary order until we find a nonzero label
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0:
                    continue
                ny, nx = y + dy, x + dx
                if not (0 <= ny < h and 0 <= nx < w):
                    continue
                neighbor_val = filled[ny, nx]
                if neighbor_val > 0:
                    filled_label = neighbor_val
                    break
            if filled_label > 0:
                break

        # Only overwrite if we found a label > 0
        if filled_label > 0:
            filled[y, x] = filled_label
        # else leave it as 0

    return filled


def get_blind_spot_mask(orig_img_path: str, low_thresh: int = 10) -> np.ndarray:
    """
    Reads an RGB (or RGBA) image from `orig_img_path` and returns a 2D boolean array
    of shape (H, W), where True = pixel belongs to a border‐connected black patch
    (“blind spot”), and False = pixel is “good data.”

    A pixel is considered “black” if all channels R,G,B are <= low_thresh.
    We then flood‐fill from any black pixel on the image’s boundary, collecting
    all 8‐connected black pixels. This way, any black patch that does *not* touch
    the border is ignored (so small dark cloud holes won’t be removed).

    Args:
        orig_img_path (str):
            Path to the PNG/JPEG satellite image. If RGBA‐encoded, the alpha is dropped.
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


def generate_multilabel_masks_from_heatmaps(
    base_heatmaps_dir: str,
    mask_dir: str,
    mask_size: Tuple[int, int],  # (width, height)
    threshold: float,  # Must match the one used in load_multilabel_mask
    iteration: int = 0,
    remove_background: bool = False,
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

    Returns:
        None. The function saves the generated masks as image files.
    """
    os.makedirs(mask_dir, exist_ok=True)
    iteration_dir = os.path.join(base_heatmaps_dir, f"iteration_{iteration}")
    heatmap_filenames = [f for f in os.listdir(iteration_dir) if f.endswith(".pt")]
    print(f"Loaded {len(heatmap_filenames)} heatmaps.")

    for filename in tqdm(heatmap_filenames, desc="Processing multi-label heatmaps"):
        # Assume the image name is encoded in the filename (without extension)
        img_name = os.path.splitext(os.path.basename(filename))[0]
        mask_path = os.path.join(mask_dir, f"{img_name}.png")

        # Generate the multi-label mask using accumulated heatmaps.
        multi_label_mask = generate_multiclass_mask_from_heatmaps(
            base_heatmaps_dir,
            img_name,
            label=1,
            iteration=iteration,
            threshold=threshold,
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

        import torch.nn.functional as F

        # Assume `heatmap` is a 2D tensor: shape [H, W]
        heatmap = (
            multi_label_mask.float().unsqueeze(0).unsqueeze(0)
        )  # shape: [1, 1, H, W]
        resized_heatmap = F.interpolate(
            heatmap,
            size=mask_size,
            mode="nearest",  # <--- this avoids interpolation artifacts
        )

        resized_heatmap = resized_heatmap.squeeze()  # shape: [new_height, new_width]

        # Convert the multi-label mask to a NumPy array.
        smoothed_mask = resized_heatmap.to(torch.int).cpu().numpy()

        if remove_background:
            possible_paths = [
                os.path.join("data/train/pos", f"{img_name}.png"),
                os.path.join("data/val/pos", f"{img_name}.png"),
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

            blind_mask = get_blind_spot_mask(img_path, low_thresh=0)
            # blind_mask is a boolean array of shape (orig_H, orig_W). But we must resize it
            # (nearest) to the same shape as mask_np, which we already chose to be (mask_H, mask_W).

            blind_mask_resized = cv2.resize(
                blind_mask.astype(np.uint8),  # convert to uint8 (0/1)
                mask_size,  # notice: cv2 expects (width, height)
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)

            # Now zero‐out exactly those blind‐spot pixels in `mask_np`:
            smoothed_mask[blind_mask_resized] = 0

            # smoothed_mask[bg_mask] = 0

            # smoothed_mask = fill_sparse_holes(smoothed_mask, kernel_size=5)

        # Optionally, if you want to visualize the mask as an image, you might normalize its values.
        # Here we scale the mask values to the range [0, 255].
        # (iteration + 1) is the maximum label value.
        smoothed_mask = (smoothed_mask / (iteration + 1)) * 255

        # Save the mask. Using uint8 to store as an image.
        cv2.imwrite(mask_path, smoothed_mask.astype(np.uint8))


# In order to run this script, run the following command from the root directory of the
# project:
# PYTHONPATH=$(pwd) python3 tools/generate_masks.py
if __name__ == "__main__":
    base_heatmaps_dir = "out/heatmaps"
    mask_dir = "out/masks"
    mask_size = (512, 512)
    threshold = 0.7
    iteration = 6

    generate_multilabel_masks_from_heatmaps(
        base_heatmaps_dir,
        mask_dir,
        mask_size,
        threshold,
        iteration,
        remove_background=False,
    )
