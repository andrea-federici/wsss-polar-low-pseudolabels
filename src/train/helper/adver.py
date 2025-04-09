import os
import random

import torch
import torchvision.transforms.functional as T

from src.utils.constants import ITERATION_FOLDER_PREFIX, HEATMAP_EXTENSION


def adversarial_erase(
    image: torch.Tensor, heatmap: torch.Tensor, threshold: float = 0.7, fill_color=0
) -> torch.Tensor:
    """
    Removes (erases) regions in the image where the corresponding heatmap
    values exceed a given threshold. The erased areas are replaced with a
    specified fill color.

    Args:
        image (torch.Tensor):
            A 4D tensor of shape (B, C, H, W) representing an image, where B
            is the batch size, C is the number of channels, and (H, W) are
            the spatial dimensions.
        heatmap (torch.Tensor):
            A 2D tensor of shape (h, w) representing the attention map, which
            indicates which regions should be erased. The heatmap will be
            resized to match (H, W) of the image.
        threshold (float, optional):
            A threshold value in the range [0, 1] that determines which
            regions are erased. Pixels in the heatmap greater than this
            threshold will be erased. Default is 0.7.
        fill_color (int, float, or torch.Tensor, optional):
            The color or value used to fill erased regions. If a scalar (int
            or float), all erased regions will be filled with this value. If a
            tensor, it must have shape (C,) to specify different colors per
            channel. Default is 0.

    Returns:
        torch.Tensor:
            A new tensor of the same shape as `image` (B, C, H, W) with the
            specified regions erased.

    Raises:
        AssertionError:
            - If `image` is not a 4D tensor.
            - If `heatmap` is not a 2D tensor.
            - If `fill_color` is a tensor and does not match the image
                channels.
    """
    assert image.dim() == 4, "Expected image to be a 4D tensor (B, C, H, W), "
    "but got shape {image.shape}"
    assert heatmap.dim() == 2, "Expected heatmap to be a 2D tensor (H, W)"

    B, C, H, W = image.shape

    heatmap = T.resize(
        heatmap.unsqueeze(0),  # (1, h, w)
        (H, W),
        interpolation=T.InterpolationMode.BILINEAR,
        antialias=True,
    ).unsqueeze(
        0
    )  # Now (1, 1, H, W)

    # Expand heatmap: (1, 1, H, W) -> (B, 1, H, W) to match batch
    # size.
    heatmap = heatmap.expand(B, 1, H, W)

    # If fill_color is a scalar (int or float), convert it to a tensor
    if isinstance(fill_color, (int, float)):
        fill_color = torch.full(
            (B, C, 1, 1), fill_color, dtype=image.dtype, device=image.device
        )
    else:
        # Ensure fill_color has the correct number of channels (C)
        assert (
            fill_color.shape[0] == C
        ), "fill_color should match the number \
            of channels of the image"
        fill_color = fill_color.view(1, C, 1, 1).expand(B, C, H, W)

    # Create erase mask: (B, 1, H, W) -> broadcast to (B, C, H, W)
    erase_mask = (heatmap > threshold).expand(B, C, H, W)

    # Clone images to avoid modifying original tensors
    erased_image = image.clone()

    print(f"erased_image shape: {erased_image.shape}, dtype: {erased_image.dtype}")
    print(f"erase_mask shape: {erase_mask.shape}, dtype: {erase_mask.dtype}")
    print(f"fill_color shape: {fill_color.shape}, dtype: {fill_color.dtype}")
    print(f"Expanded fill_color shape: {fill_color.expand_as(erased_image).shape}")

    # Check how many elements are being accessed
    print(f"erase_mask sum: {erase_mask.sum().item()} (number of pixels affected)")

    if torch.isnan(erased_image).any():
        print("NaN detected in erased_image!")
    if torch.isnan(fill_color).any():
        print("NaN detected in fill_color!")
    if torch.isnan(erase_mask).any():
        print("NaN detected in erase_mask!")

    # Apply erasing by filling masked regions with the specified fill color
    erased_image[erase_mask] = fill_color.expand_as(erased_image)[erase_mask]

    return erased_image


def load_heatmap(base_heatmaps_dir: str, iteration: int, img_name: str) -> torch.Tensor:
    """
    Load a heatmap tensor from the specified base heatmaps directory based on the
    provided iteration and image name.
    The base heatmaps directory should contain subdirectories named with the iteration
    prefix. Each subdirectory should contain heatmap files named after the images they
    correspond to.

    Example structure:

        heatmaps/
            iteration_0/
                image1.pt
                image2.pt
            iteration_1/
                image1.pt
                image2.pt

    Args:
        base_heatmaps_dir (str): The base directory containing the iteration folders.
        iteration (int): The iteration identifier to locate the specific heatmap folder.
        img_name (str): The name of the image file (including or excluding extension)
            from which the heatmap is derived. The img_name can also be the full path
            to the image file, and the function will automatically extract the base
            name.

    Returns:
        torch.Tensor: The loaded heatmap tensor.

    Raises:
        FileNotFoundError: If the heatmap file does not exist in the expected location.

    Example:
        >>> heatmap = load_heatmap("/path/to/heatmaps", "1", "image.png")
    """
    assert iteration >= 0, "Iteration must be greater than or equal to 0"

    # Remove the file extension and get the root name
    # ("data/train/image.png" -> "image")
    img_root = os.path.basename(os.path.splitext(img_name)[0])

    # Construct the heatmap path
    heatmap_path = os.path.join(
        base_heatmaps_dir,
        ITERATION_FOLDER_PREFIX + str(iteration),
        img_root + HEATMAP_EXTENSION,
    )

    if os.path.exists(heatmap_path):
        return torch.load(heatmap_path)
    raise FileNotFoundError(f"Heatmap not found: {heatmap_path}. ")


def random_heatmap(base_heatmaps_dir: str, iteration: int) -> torch.Tensor:
    """
    Load a random heatmap tensor from the specified base heatmaps directory
    based on the provided iteration.

    The base heatmaps directory should contain subdirectories named with the
    iteration prefix. Each subdirectory should contain heatmap files corresponding
    to different images.

    Example structure:

        heatmaps/
            iteration_0/
                image1.pt
                image2.pt
            iteration_1/
                image1.pt
                image2.pt

    Args:
        base_heatmaps_dir (str): The base directory containing the iteration folders.
        iteration (int): The iteration identifier to locate the specific heatmap folder.

    Returns:
        torch.Tensor: A randomly selected heatmap tensor from the specified iteration
            folder.

    Raises:
        FileNotFoundError: If the heatmap directory for the given iteration does not
            exist or if there are no heatmap files in the directory.
    """
    assert iteration >= 0, "Iteration must be greater than or equal to 0"

    heatmaps_dir = os.path.join(
        base_heatmaps_dir,
        ITERATION_FOLDER_PREFIX + str(iteration),
    )

    heatmap_files = [
        os.path.join(heatmaps_dir, f)
        for f in os.listdir(heatmaps_dir)
        if f.endswith(HEATMAP_EXTENSION)
    ]

    # If heatmap_files is empty, raise an error
    if not heatmap_files:
        raise FileNotFoundError(
            f"No heatmap files found in {heatmaps_dir}. "
            "Ensure the directory contains heatmap files."
        )

    return torch.load(random.choice(heatmap_files))


def load_matching_heatmap(
    base_heatmaps_dir: str, iteration: int, img_prefix: str
) -> torch.Tensor:
    """
    Load a heatmap tensor that matches a given image prefix from the specified
    base heatmaps directory and iteration.

    The base heatmaps directory contains subdirectories named with the iteration
    prefix, each containing heatmap files. This function searches for a heatmap
    file whose name starts with the provided image prefix.

    Example structure:

        heatmaps/
            iteration_0/
                image123.pt
                image456.pt
            iteration_1/
                image123.pt
                image456.pt

    Args:
        base_heatmaps_dir (str): The base directory containing the iteration folders.
        iteration (int): The iteration identifier to locate the specific heatmap folder.
        img_prefix (str): The prefix of the image name used to identify the
            corresponding heatmap file.

    Returns:
        torch.Tensor: The loaded heatmap tensor.

    Raises:
        FileNotFoundError: If no heatmap file matching the given prefix is found.
    """
    assert iteration >= 0, "Iteration must be greater than or equal to 0"

    # img_prefix = os.path.basename(img_path)[:6]  # First 6 characters
    heatmaps_dir = os.path.join(
        base_heatmaps_dir,
        ITERATION_FOLDER_PREFIX + str(iteration),
    )

    matching_files = [
        f
        for f in os.listdir(heatmaps_dir)
        if f.startswith(img_prefix) and f.endswith(HEATMAP_EXTENSION)
    ]

    if not matching_files:
        raise FileNotFoundError(
            f"No matching heatmap found for {img_prefix} in {heatmaps_dir}. "
        )

    first_match = os.path.join(heatmaps_dir, matching_files[0])

    return torch.load(first_match)


def load_accumulated_heatmap(
    base_heatmaps_dir: str, img_name: str, label: int, iteration: int
) -> torch.Tensor:
    """
    Load and accumulate heatmaps over multiple iterations for a given image and label.

    This function accumulates heatmaps from iteration 0 up to the specified iteration
    for a given image. If the label is positive (1), it loads the heatmaps corresponding
    to the image itself. If the label is negative (0), it loads matching heatmaps
    based on the first six characters of the image name.

    The accumulated heatmap is normalized to ensure values remain between 0 and 1.

    Args:
        base_heatmaps_dir (str): The base directory containing the iteration folders.
        img_name (str): The name of the image file (including or excluding extension).
        label (int): The class label (0 for negative, 1 for positive).
        iteration (int): The highest iteration (inclusive) to consider for heatmap
            accumulation.

    Returns:
        torch.Tensor: The accumulated and normalized heatmap tensor.

    Raises:
        AssertionError: If the iteration is negative or if the label is not 0 or 1.
        FileNotFoundError: If a required heatmap file is missing.
    """
    assert iteration >= 0, "Iteration must be greater than or equal to 0"
    assert label in [0, 1], "Label must be either 0 or 1 (negative or positive)"

    # Load a reference heatmap to get the proper shape.
    reference_heatmap = random_heatmap(base_heatmaps_dir, 0)

    # Initialize an all-zeros tensor for heatmap accumulation.
    accumulated_heatmap = torch.zeros_like(reference_heatmap, dtype=torch.float32)

    # Iterate from 0 to iteration (inclusive) to accumulate heatmaps.
    for it in range(iteration + 1):
        if label == 1:  # Positive sample: use its own heatmap
            heatmap = load_heatmap(base_heatmaps_dir, it, img_name)
        else:  # Negative sample: load the matching heatmap
            heatmap = load_matching_heatmap(base_heatmaps_dir, it, img_name[:6])

        accumulated_heatmap += heatmap

    # Normalize the accumulated heatmap to keep values between 0 and 1
    accumulated_heatmap = torch.clamp(accumulated_heatmap, 0, 1)

    return accumulated_heatmap


def load_multiclass_mask(
    base_heatmaps_dir: str,
    img_name: str,
    label: int,
    iteration: int,
    threshold: float,
):
    """
    Generate a multiclass activation mask based on accumulated heatmaps over multiple
    iterations.

    This function accumulates heatmaps from iteration 0 up to the specified iteration
    (included) for a given image. It then applies a threshold to identify activated
    pixels and assigns iteration indices to newly activated pixels in a multiclass mask.

    - If the label is positive (1), heatmaps corresponding to the image itself are used.
    - If the label is negative (0), heatmaps are selected based on the first six
      characters of the image name.

    The function tracks newly activated pixels at each iteration and assigns them the
    corresponding iteration index, effectively capturing the progression of activations
    over time.

    Args:
        base_heatmaps_dir (str): The base directory containing the iteration folders.
        img_name (str): The name of the image file (including or excluding extension).
        label (int): The class label (0 for negative, 1 for positive).
        iteration (int): The highest iteration (inclusive) to consider for heatmap
            accumulation.
        threshold (float): The activation threshold for binarizing the heatmap.

    Returns:
        torch.Tensor: A multiclass mask where each pixel is labeled with the iteration
            index at which it first became active.

    Raises:
        AssertionError: If the iteration is negative, the label is not 0 or 1, or the
            threshold in not in the range [0, 1].
        FileNotFoundError: If a required heatmap file is missing.
    """
    assert iteration >= 0, "Iteration must be greater than or equal to 0"
    assert label in [0, 1], "Label must be either 0 or 1 (negative or positive)"
    assert threshold >= 0 and threshold <= 1, "Threshold must be between 0 and 1"

    # Load a reference heatmap to get the proper shape.
    reference_heatmap = random_heatmap(base_heatmaps_dir, 0)
    multiclass_mask = torch.zeros_like(reference_heatmap, dtype=torch.int32)

    # For accumulation, start with an all-zeros tensor.
    cumulative_heatmap = torch.zeros_like(reference_heatmap, dtype=torch.float32)

    # This will hold the binary mask from the previous iteration.
    prev_binary = torch.zeros_like(reference_heatmap, dtype=torch.bool)

    for it in range(iteration + 1):
        # Load the heatmap for the current iteration.
        if label == 1:
            heatmap = load_heatmap(base_heatmaps_dir, it, img_name)
        else:
            heatmap = load_matching_heatmap(base_heatmaps_dir, it, img_name[:6])

        # Accumulate and then clamp the result
        cumulative_heatmap += heatmap
        clamped_heatmap = torch.clamp(cumulative_heatmap, 0, 1)

        # Threshold the clamped accumulated heatmap to get the binary activation.
        current_binary = clamped_heatmap > threshold

        # Identify new activations: pixels that are active now but were not previously.
        new_activation = current_binary & (~prev_binary)

        # Reverse the label: iteration 0 gets the highest value (iteration), iteration
        # 'iteration' gets 1.
        reversed_label = iteration - it + 1
        multiclass_mask[new_activation] = reversed_label
        # multiclass_mask[new_activation] = it

        # Update the previous binary mask.
        prev_binary = current_binary

    return multiclass_mask


### ----- TESTING ----- ###

# import torch.nn.functional as F


# def smooth_heatmap(heatmap, target_size=(500, 500)):
#     """
#     Resizes and smooths a heatmap using bilinear interpolation.

#     Parameters:
#       heatmap (torch.Tensor): Input heatmap (multi-label or continuous).
#       target_size (tuple): Desired output size (height, width).

#     Returns:
#       torch.Tensor: The resized and smoothed heatmap.
#     """
#     # Add batch and channel dimensions: [B, C, H, W]
#     heatmap = heatmap.unsqueeze(0).unsqueeze(0).float()

#     # Use bilinear interpolation which naturally smooths out the values.
#     smoothed = F.interpolate(
#         heatmap, size=target_size, mode="bilinear", align_corners=False
#     )

#     # Remove batch and channel dimensions.
#     return smoothed.squeeze(0).squeeze(0)


# if __name__ == "__main__":
#     import cv2
#     import numpy as np
#     import matplotlib.pyplot as plt

#     base_heatmaps_dir = "out/heatmaps"
#     img_path = "1d2200_20181006T085600_20181006T085715_mos_rgb.png.png"
#     label = 1
#     # iteration = 10
#     threshold = 0.75  # Must be the same as the one used in training

#     for it in range(1, 10):
#         multilabel_mask = load_multiclass_mask(
#             base_heatmaps_dir, img_path, label, it, threshold
#         )

#         # multilabel_mask = cv2.resize(
#         #     multilabel_mask.numpy(),
#         #     (500, 500),  # cv2.resize takes (width, height)
#         #     interpolation=cv2.INTER_NEAREST,
#         # )

#         multilabel_mask = smooth_heatmap(multilabel_mask, target_size=(500, 500))

#         multilabel_mask = cv2.GaussianBlur(
#             multilabel_mask.numpy().astype(np.float32), (11, 11), 0
#         )

#         # Plot the multi-label mask
#         plt.figure(figsize=(8, 8))
#         plt.imshow(multilabel_mask, cmap="viridis")
#         plt.colorbar()
#         plt.title("Multi-label mask")
#         plt.savefig(f"multilabel_mask_{it}.png")
#         plt.show()
