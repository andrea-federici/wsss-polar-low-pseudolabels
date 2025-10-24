from typing import Tuple

import cv2
import numpy as np
import torch
from captum.attr import IntegratedGradients

from src.data.image_processing import normalize_image_to_range


def generate_heatmap(
    model: torch.nn.Module,
    image: torch.Tensor,
    *,
    target_class: int = None,
    n_steps: int = 50,
    baseline=None,
    device: str = "cpu",
) -> torch.Tensor:
    """
    Generates an Integrated Gradients heatmap for a given model and input image.

    Args:
        model (torch.nn.Module): The model to explain.
        image (torch.Tensor): Input image tensor of shape (1, C, H, W).
        target_class (int, optional): Class index to compute attributions for.
            If None, uses the predicted class.
        n_steps (int): Number of steps for IG path integral approximation.
        baseline (torch.Tensor, optional): Baseline input tensor. Defaults to a black image.
        device (str, optional): The device to run the computations on. Defaults to "cpu".

    Returns:
        torch.Tensor: A 2D heatmap tensor of shape (H, W), normalized to [0, 1].
    """
    if image.dim() != 4:
        raise ValueError(
            f"Expected image to be a 4D tensor (1, C, H, W), but "
            f"got shape {image.shape}"
        )

    was_training = model.training
    model.zero_grad()
    model.eval()

    try:
        image = image.to(device)  # Move image to the same device as the model

        if baseline is None:
            baseline = torch.zeros_like(
                image
            )  # Use a black image as the baseline if no baseline is provided

        ig = IntegratedGradients(model)

        if target_class is None:
            # Get the predicted class for the input image
            logits = model(image)
            target_class = torch.argmax(logits, dim=1).item()

        attributions = ig.attribute(
            image,
            baselines=baseline,
            target=target_class,
            n_steps=n_steps,
        )

        attributions = attributions.squeeze().detach().mean(dim=0)

        # Normalize to [0, 1]
        min_val = attributions.min()
        max_val = attributions.max()
        denom = (max_val - min_val).clamp(min=1e-8)
        norm_attributions = (attributions - min_val) / denom

        return norm_attributions

    finally:
        if was_training:
            model.train()


def generate_mask_from_heatmap(
    image: torch.Tensor,
    heatmap: torch.Tensor,
    *,
    percentile: int = 98,
    kernel_size: Tuple[int, int] = (15, 15),
    min_area_fraction: float = 0.2,
) -> np.ndarray:
    """
    Generates a binary mask from a heatmap by applying percentile thresholding,
    morphological operations, and area-based filtering.

    Args:
        image (torch.Tensor): Input image tensor of shape (1, C, H, W), used
            to determine the target size for resizing the heatmap.
        heatmap (torch.Tensor): A 2D heatmap tensor of shape (H, W) with values
            typically in the range [0, 1].
        percentile (int, optional): Percentile threshold used to binarize the
            heatmap. Defaults to 98, meaning only the top 2% of heatmap values
            are retained.
        kernel_size (Tuple[int, int], optional): Size of the structuring element
            used for morphological closing. Defaults to (15, 15).
        min_area_fraction (float, optional): Minimum area for a region to be
            kept, expressed as a fraction of the largest detected contour.
            Defaults to 0.2 (i.e., 20% of the largest region).

    Returns:
        np.ndarray: A 2D binary mask (uint8) of shape (H, W), where salient regions
            are marked with 1s and the background with 0s.

    Raises:
        ValueError: If `heatmap` is not a 2D tensor or if `image` is not a 4D tensor.
    """
    if heatmap.dim() != 2:
        raise ValueError(
            f"Expected heatmap to be a 2D tensor (H, W), but got shape {heatmap.shape}"
        )
    if image.dim() != 4:
        raise ValueError(
            "Expected input_image to be a 4D tensor (B, C, H, W), but got shape "
            "{image.shape}"
        )

    heatmap = heatmap.detach().cpu().numpy()

    _, _, H, W = image.shape
    heatmap_resized = cv2.resize(heatmap, (W, H), interpolation=cv2.INTER_LINEAR)

    # Binarize the heatmap using a threshold calculated based on the percentile
    heatmap_flattened = heatmap_resized.flatten()
    threshold = np.percentile(heatmap_flattened, percentile)
    binary_mask = np.where(heatmap_resized >= threshold, 1, 0).astype(np.uint8)

    # Morphological Closing: Dilate, then erode to merge dense regions. Finally, blur the mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel_size)
    closed_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
    blurred_mask = cv2.GaussianBlur(closed_mask, (15, 15), 0)

    # Area opening: Remove small objects based on minimum area
    contours, _ = cv2.findContours(
        blurred_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    areas = [cv2.contourArea(c) for c in contours]
    if areas:
        max_area = max(areas)
        min_area = max_area * min_area_fraction
    else:
        min_area = 0

    for c in contours:
        if cv2.contourArea(c) < min_area:
            cv2.drawContours(blurred_mask, [c], -1, 0, -1)

    # Apply Gaussian blur again to smooth the mask
    final_mask = cv2.GaussianBlur(blurred_mask, (31, 31), 0).astype(np.uint8)

    return final_mask


def overlay_mask(
    input_image: torch.Tensor, mask: np.ndarray, *, alpha: float = 0.5
) -> np.ndarray:
    """
    Overlays a binary mask onto a tensor image using a specified transparency.

    Args:
        input_image (torch.Tensor): Image tensor of shape (1, C, H, W).
        mask (np.ndarray): Binary mask of shape (H, W) with values 0 or 1.
        alpha (float, optional): Blending factor for the overlay. Defaults to 0.5.

    Returns:
        np.ndarray: Image array of shape (H, W, C) with the overlay applied.

    Raises:
        ValueError: If `mask` is not a binary mask with values 0 or 1.
    """
    if not np.array_equal(mask, mask.astype(bool)):
        raise ValueError("Expected binary mask with values 0 or 1.")

    img_np = input_image.squeeze(0).permute(1, 2, 0).numpy()
    img_np = normalize_image_to_range(img_np, target_range=(0, 1))

    # Create a color overlay from the binary mask
    color_overlay = np.zeros_like(img_np)
    color_overlay[mask == 1] = [75, 0, 130]  # Purple

    # Blend the original image with the color overlay
    img_with_mask = img_np.copy()
    img_with_mask[mask == 1] = (
        alpha * color_overlay[mask == 1] + (1 - alpha) * img_np[mask == 1]
    )

    return img_with_mask


def overlay_heatmap(
    image: torch.Tensor,
    heatmap: np.ndarray,
    target_class: int,
    alpha: float = 0.6,
    percentile_neg: float = 97,
    percentile_pos: float = 98,
) -> np.ndarray:
    """
    Overlays a green heatmap onto an input image based on class-specific saliency regions.

    Args:
        image (torch.Tensor): Input image tensor of shape (1, C, H, W), expected to be in
            the range [0, 1] or [0, 255].
        heatmap (np.ndarray): 2D array of shape (H, W) representing the saliency or activation
            map to be overlaid.
        target_class (int): Class label for which the overlay should be generated. If 0,
            `percentile_neg` is used; otherwise, `percentile_pos`.
        alpha (float, optional): Transparency level of the overlay. Defaults to 0.6.
        percentile_neg (float, optional): Percentile used to threshold the heatmap for class 0.
            Defaults to 97.
        percentile_pos (float, optional): Percentile used to threshold the heatmap for non-zero
            classes. Defaults to 98.

    Returns:
        np.ndarray: A NumPy array representing the RGB image with a green heatmap overlay,
        of shape (H, W, 3), dtype uint8, in the range [0, 255].

    Raises:
        ValueError: If the image is not a 4D tensor or if the heatmap is not a 2D array.
    """
    if image.dim() != 4:
        raise ValueError(
            f"Expected image to be a 4D tensor (1, C, H, W), but got shape {image.shape}"
        )

    if heatmap.ndim != 2:
        raise ValueError(
            f"Expected heatmap to be a 2D array (H, W), but got shape {heatmap.shape}"
        )

    img_np = image.squeeze(0).permute(1, 2, 0).detach().cpu().numpy().astype(np.float32)

    heatmap_resized = cv2.resize(heatmap, (img_np.shape[1], img_np.shape[0]))

    # Set the percentile based on the target class
    percentile = percentile_neg if target_class == 0 else percentile_pos

    # Calculate the threshold based on the percentile
    threshold = np.percentile(heatmap_resized.flatten(), percentile)

    # Apply threshold to focus on high attribution regions
    heatmap_thresholded = np.where(heatmap_resized >= threshold, heatmap_resized, 0)

    # Create a green-only heatmap
    green_heatmap = np.zeros_like(img_np, dtype=np.float32)
    green_heatmap[:, :, 1] = np.clip(heatmap_thresholded, 0, 1)

    # Apply a small Gaussian blur to make dots more visible
    green_heatmap = cv2.GaussianBlur(green_heatmap, (7, 7), 0)

    # Blend the heatmap with the original image
    image_with_heatmap = cv2.addWeighted(green_heatmap, alpha, img_np, 1 - alpha, 0)

    return image_with_heatmap
