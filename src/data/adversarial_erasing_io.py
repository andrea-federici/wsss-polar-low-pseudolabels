import os
import random
from typing import Union

import torch

from src.utils.constants import ITERATION_FOLDER_PREFIX, PYTORCH_EXTENSION


def load_tensor(base_dir: str, iteration: int, img_name: str) -> torch.Tensor:
    """
    Load a PyTorch tensor from a specified iteration folder.

    The directory structure must follow:
        base_dir/
            iteration_<iteration>/
                <img_root>.pt

    <img_root> is derived by stripping any path and extension from img_name.

    Args:
        base_dir (str): Root directory containing iteration subfolders.
        iteration (int): Non-negative index of the iteration folder to load from.
        img_name (str): Image filename (with or without extension) or full path; only the
                        base name (without extension) is used to locate the tensor.

    Returns:
        torch.Tensor: The loaded tensor. For binary masks, convert using .bool().

    Raises:
        AssertionError: If `iteration` is negative.
        FileNotFoundError: If the tensor file does not exist at the expected path.
    """
    assert iteration >= 0, "Iteration must be greater than or equal to 0"

    # Remove the file extension and get the root name
    # ("data/train/image.png" -> "image")
    img_root = os.path.basename(os.path.splitext(img_name)[0])

    # Construct the tensor path
    tensor_path = os.path.join(
        base_dir,
        f"{ITERATION_FOLDER_PREFIX}{iteration}",
        img_root + PYTORCH_EXTENSION,
    )

    if not os.path.exists(tensor_path):
        raise FileNotFoundError(f"Tensor not found: {tensor_path}")

    return torch.load(tensor_path)


def pick_random_tensor(
    base_dir: str, iteration: int, return_path: bool = False
) -> Union[torch.Tensor, str]:
    """
    Load a random .pt tensor from a specified iteration folder.

    Directory structure:
        base_dir/
            iteration_<iteration>/
                *.pt

    Args:
        base_dir (str): Root directory containing iteration subfolders.
        iteration (int): Non-negative index of the iteration folder to sample from.
        return_path (bool, optional): If True, return the file path of the selected
            tensor instead of loading it. Defaults to False.

    Returns:
        torch.Tensor or str
            - If `return_path=False`, returns the loaded torch.Tensor.
            - If `return_path=True`, returns the filesystem path (str) of the chosen .pt file.

    Raises:
        AssertionError: If `iteration` is negative.
        FileNotFoundError: If the iteration folder does not exist or contains no .pt files.
    """
    assert iteration >= 0, "Iteration must be greater than or equal to 0"

    iteration_folder = os.path.join(
        base_dir,
        f"{ITERATION_FOLDER_PREFIX}{iteration}",
    )
    if not os.path.isdir(iteration_folder):
        raise FileNotFoundError(
            f"Iteration folder not found: {iteration_folder}. "
            "Ensure the directory structure is correct."
        )

    tensor_files = [
        os.path.join(iteration_folder, f)
        for f in os.listdir(iteration_folder)
        if f.endswith(PYTORCH_EXTENSION)
    ]

    if not tensor_files:
        raise FileNotFoundError(f"No .pt files found in: {iteration_folder}")

    tensor_path = random.choice(tensor_files)

    return tensor_path if return_path else torch.load(tensor_path)


def load_matching_tensor(
    base_dir: str, iteration: int, img_prefix: str
) -> torch.Tensor:
    """
    Load the first .pt tensor that matches a given image prefix from the specified
    base directory and iteration.

    Directory structure:
        base_dir/
            iteration_<iteration>/
                <img_prefix>* .pt

    Args:
        base_dir (str): Root directory containing iteration subfolders.
        iteration (int): Non-negative index of the iteration folder to search.
        img_prefix (str): The prefix of the image name used to identify the
            corresponding tensor.

    Returns:
        torch.Tensor: The loaded heatmap tensor. Convert to .bool() for binary masks.

    Raises:
        AssertionError: If `iteration` is negative.
        FileNotFoundError: If the iteration folder does not exist or no files match.
    """
    assert iteration >= 0, "Iteration must be greater than or equal to 0"

    # img_prefix = os.path.basename(img_path)[:6]  # First 6 characters
    iteration_folder = os.path.join(
        base_dir,
        f"{ITERATION_FOLDER_PREFIX}{iteration}",
    )
    if not os.path.isdir(iteration_folder):
        raise FileNotFoundError(
            f"Iteration folder not found: {iteration_folder}. "
            "Ensure the directory structure is correct."
        )

    matching_files = [
        f
        for f in os.listdir(iteration_folder)
        if f.startswith(img_prefix) and f.endswith(PYTORCH_EXTENSION)
    ]

    if not matching_files:
        raise FileNotFoundError(
            f"No matching .pt files found in: {iteration_folder} for prefix: {img_prefix}"
        )

    first_match = os.path.join(iteration_folder, matching_files[0])

    return torch.load(first_match)


def load_accumulated_heatmap(
    base_heatmaps_dir: str,
    img_name: str,
    label: int,
    iteration: int,
    negative_load_strategy: str = None,  # We use type str here to avoid circular
    # import. It would be better to use the enum type.
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
        negative_load_strategy (str, optional): The strategy for loading negative samples
            when label is 0. It can be one of the following:
            - "random": Load a random heatmap from the base directory and use it for
                all iterations.
            - "pl_specific": Load a heatmap that matches the first 6 characters of the
                image name.

    Returns:
        torch.Tensor: The accumulated and normalized heatmap tensor.

    Raises:
        ValueError: If `iteration` is negative, `label` is not 0 or 1, or if
            `negative_load_strategy` is not specified for negative samples.
    """
    # Lazy import to avoid circular dependencies
    from src.models.erase_strategies import NegativeLoadStrategy

    if iteration < 0:
        raise ValueError("Iteration must be non-negative")
    if label not in [0, 1]:
        raise ValueError("Label must be either 0 or 1 (negative or positive)")

    if negative_load_strategy is None and label == 0:
        raise ValueError(
            "Negative load strategy must be specified for negative samples (label=0)."
        )

    # Load a reference heatmap to get the proper shape.
    reference_heatmap = pick_random_tensor(base_heatmaps_dir, 0)

    # Initialize an all-zeros tensor for heatmap accumulation.
    accumulated_heatmap = torch.zeros_like(reference_heatmap, dtype=torch.float32)

    if negative_load_strategy == NegativeLoadStrategy.RANDOM.value and label == 0:
        random_img_name = pick_random_tensor(
            base_heatmaps_dir, iteration=0, return_path=True
        )

    # Iterate from 0 to iteration (inclusive) to accumulate heatmaps.
    for it in range(iteration + 1):
        if label == 1:  # Positive sample: use its own heatmap
            heatmap = load_tensor(base_heatmaps_dir, it, img_name)
        else:  # Negative sample: load the matching heatmap
            if negative_load_strategy not in NegativeLoadStrategy.list():
                raise ValueError(
                    f"Invalid negative load strategy: {negative_load_strategy}. "
                    f"Must be one of {NegativeLoadStrategy.list()}."
                )
            if negative_load_strategy == NegativeLoadStrategy.RANDOM.value:
                heatmap = load_tensor(base_heatmaps_dir, it, random_img_name)
            elif negative_load_strategy == NegativeLoadStrategy.PL_SPECIFIC.value:
                heatmap = load_matching_tensor(base_heatmaps_dir, it, img_name[:6])

        accumulated_heatmap += heatmap

    # Normalize the accumulated heatmap to keep values between 0 and 1
    accumulated_heatmap = torch.clamp(accumulated_heatmap, 0, 1)

    return accumulated_heatmap


def load_accumulated_mask(
    base_masks_dir: str,
    img_name: str,
    label: int,
    iteration: int,
    negative_load_strategy: str = None,
) -> torch.Tensor:
    """
    Load and accumulate binary masks over multiple iterations for a given image and label.

    This function accumulates binary masks from iteration 0 up to the specified iteration
    for a given image. If the label is positive (1), it loads the masks corresponding
    to the image itself. If the label is negative (0), it loads matching masks
    based on the first six characters of the image name.

    The accumulated mask is computed as the union (logical OR) of all masks,
    resulting in a binary tensor indicating any positive detection across iterations.

    Args:
        base_masks_dir (str): The base directory containing the iteration folders.
        img_name (str): The name of the image file (including or excluding extension).
        label (int): The class label (0 for negative, 1 for positive).
        iteration (int): The highest iteration (inclusive) to consider for mask
            accumulation.
        negative_load_strategy (str, optional): The strategy for loading negative samples
            when label is 0. It can be one of the following:
            - "random": Load a random mask from the base directory and use it for all
                iterations.
            - "pl_specific": Load a mask that matches the first 6 characters of the
                image name.

    Returns:
        torch.Tensor: The accumulated binary mask tensor (dtype=torch.bool).

    Raises:
        ValueError: If `iteration` is negative, `label` is not 0 or 1, or if
            `negative_load_strategy` is not specified for negative samples.
    """
    from src.models.erase_strategies import NegativeLoadStrategy

    # Validate inputs
    if iteration < 0:
        raise ValueError("Iteration must be non-negative")
    if label not in [0, 1]:
        raise ValueError("Label must be either 0 or 1 (negative or positive)")

    if negative_load_strategy is None and label == 0:
        raise ValueError(
            "Negative load strategy must be specified for negative samples (label=0)."
        )

    # Load a reference mask to get the proper shape and dtype
    reference_mask = pick_random_tensor(base_masks_dir, 0)
    if reference_mask.dtype != torch.bool:
        reference_mask = reference_mask.bool()

    # Initialize an all-False tensor for mask accumulation
    accumulated_mask = torch.zeros_like(reference_mask, dtype=torch.bool)

    if negative_load_strategy == NegativeLoadStrategy.RANDOM.value and label == 0:
        random_img_name = pick_random_tensor(
            base_masks_dir, iteration=0, return_path=True
        )

    # Iterate from 0 to iteration (inclusive) to accumulate masks
    for it in range(iteration + 1):
        if label == 1:
            mask = load_tensor(base_masks_dir, it, img_name)
        else:
            if negative_load_strategy not in NegativeLoadStrategy.list():
                raise ValueError(
                    f"Invalid negative load strategy: {negative_load_strategy}. "
                    f"Must be one of {NegativeLoadStrategy.list()}."
                )

            if negative_load_strategy == NegativeLoadStrategy.RANDOM.value:
                mask = load_tensor(base_masks_dir, it, random_img_name)
            elif negative_load_strategy == NegativeLoadStrategy.PL_SPECIFIC.value:
                mask = load_matching_tensor(base_masks_dir, it, img_name[:6])

        # Ensure mask is boolean
        if mask.dtype != torch.bool:
            mask = mask.bool()

        # Union masks across iterations
        accumulated_mask |= mask

    return accumulated_mask


def generate_multiclass_mask_from_heatmaps(
    base_dir: str,
    img_name: str,
    label: int,
    iteration: int,
    threshold: float,
) -> torch.Tensor:
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
    reference_heatmap = pick_random_tensor(base_dir, iteration=0)
    multiclass_mask = torch.zeros_like(reference_heatmap, dtype=torch.int32)

    # For accumulation, start with an all-zeros tensor.
    cumulative_heatmap = torch.zeros_like(reference_heatmap, dtype=torch.float32)

    # This will hold the binary mask from the previous iteration.
    prev_binary = torch.zeros_like(reference_heatmap, dtype=torch.bool)

    for it in range(iteration + 1):
        # Load the heatmap for the current iteration.
        if label == 1:
            heatmap = load_tensor(base_dir, it, img_name)
        else:
            heatmap = load_matching_tensor(base_dir, it, img_name[:6])

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

        # Update the previous binary mask.
        prev_binary = current_binary

    return multiclass_mask


def generate_multiclass_mask_from_masks(
    base_dir: str,
    img_name: str,
    max_iteration: int,
) -> torch.Tensor:
    assert max_iteration >= 0, "Iteration must be non-negative"

    reference_mask = pick_random_tensor(base_dir, iteration=0)
    multiclass_mask = torch.zeros_like(reference_mask, dtype=torch.int32)
    prev_binary = torch.zeros_like(reference_mask, dtype=torch.bool)

    for it in range(max_iteration + 1):
        mask = load_tensor(base_dir, it, img_name)
        mask = mask.bool()
        new_activation = mask & (~prev_binary)
        reversed_label = max_iteration - it + 1
        multiclass_mask[new_activation] = reversed_label
        prev_binary = prev_binary | mask

    # un = torch.unique(multiclass_mask, return_counts=True)
    # print(un[0], un[1])
    return multiclass_mask
