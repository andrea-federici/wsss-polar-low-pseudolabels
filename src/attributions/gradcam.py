from typing import Sequence, Tuple, Union

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from captum.attr import LayerGradCam

from src.data.image_processing import convert_to_np_array, normalize_image_to_range


def generate_heatmap(
    model: torch.nn.Module,
    image: torch.Tensor,
    *,
    target_class: int = None,
    layer: torch.nn.Module = None,
) -> torch.Tensor:
    """
    Generates a Grad-CAM heatmap for a given input image and target class.
    The heatmap is normalized between 0 and 1.

    Args:
        model (torch.nn.Module): The model for which Grad-CAM is computed.
            It must implement a `get_last_conv_layer()` method if `layer` is not
            provided.
        image (torch.Tensor): The input image tensor of shape
            (1, C, H, W), where:
                - C is the number of channels
                - H is the height
                - W is the width
        target_class (int, optional): The target class for which the
            Grad-CAM heatmap is computed. If None, the model's predicted
            class is used.
        layer (torch.nn.Module, optional): The layer of the model to
            compute Grad-CAM attributions from. If None, the last
            convolutional layer of the model is used.

    Returns:
        torch.Tensor: A 2D tensor of shape (H, W) representing the
            normalized heatmap, where values range from 0 to 1.

    Raises:
        ValueError: If `image` is not a 4D tensor (batch dimension
            required).
    """
    if image.dim() != 4:
        raise ValueError(
            f"Expected image to be a 4D tensor (1, C, H, W), but "
            f"got shape {image.shape}"
        )

    was_training = model.training
    model.eval()
    model.zero_grad()

    try:
        # If no layer is specified, use the last layer of the feature
        # extractor
        if layer is None:
            layer = model.get_last_conv_layer()

        # Initialize GradCAM with the model and the specified layer
        gradcam = LayerGradCam(model, layer)

        # Move image to the same device as the model
        image = image.to(next(model.parameters()).device)

        if target_class is None:
            # Get the predicted class for the input image
            logits = model(image)
            target_class = torch.argmax(logits, dim=1).item()

        # Compute the GradCAM attributions for the input image
        attr = gradcam.attribute(image, target_class)

        # Remove batch dimension and detach from the computation graph
        attr = attr.squeeze().detach()

        # Remove negative values in the heatmap
        heatmap = torch.clamp(attr, min=0)

        # Normalize the heatmap between 0 and 1
        max_value = torch.max(heatmap)
        norm_heatmap = heatmap / max_value if max_value != 0 else heatmap
        return norm_heatmap

    finally:
        if was_training:
            model.train()


# When using this method we want to pass the image at the highest possible resolution
def generate_super_heatmap(
    model: torch.nn.Module,
    image: torch.Tensor,
    *,
    target_size: Tuple[int, int],  # (H, W)
    sizes: Sequence[int],
    target_class: int,
    layer: torch.nn.Module = None,
) -> torch.Tensor:
    """
    Generates a multi-scale (super) Grad-CAM heatmap by computing Grad-CAM
    at multiple input resolutions and averaging the results. The final heatmap
    is normalized between 0 and 1.

    Args:
        model (torch.nn.Module): The model for which Grad-CAM is computed.
            If `layer` is not provided, the model must implement a
            `get_last_conv_layer()` method.
        image (torch.Tensor): Input image tensor of shape (1, C, H, W).
        target_size (Tuple[int, int]): The target (height, width) of the
            final output heatmap.
        sizes (Sequence[int]): A list of integers specifying the spatial
            resolutions (s x s) to which the input image is resized before
            computing individual Grad-CAM heatmaps.
        target_class (int): The class index for which Grad-CAM is computed.
        layer (torch.nn.Module, optional): The convolutional layer to use for
            Grad-CAM. If None, the last convolutional layer of the model is used.

    Returns:
        torch.Tensor: A 2D tensor of shape `target_size` (H, W), representing
            the normalized super-resolution Grad-CAM heatmap with values in [0, 1].

    Raises:
        ValueError: If `image` is not a 4D tensor of shape (1, C, H, W),
            or if `sizes` contains invalid values.
    """
    if image.dim() != 4:
        raise ValueError(
            f"Expected image to be a 4D tensor (1, C, H, W), but "
            f"got shape {image.shape}"
        )

    if not all(isinstance(s, int) and s > 0 for s in sizes):
        raise ValueError("All elements in `sizes` must be positive integers.")

    # Move image to the same device as the model
    image = image.to(next(model.parameters()).device)

    was_training = model.training
    model.eval()
    model.zero_grad()

    try:
        # If no layer is specified, use the last layer of the feature
        # extractor
        if layer is None:
            layer = model.get_last_conv_layer()

        H, W = target_size
        resized_heatmaps = []

        for s in sizes:
            img_resized = F.interpolate(
                image, size=(s, s), mode="bilinear", align_corners=False
            )
            heatmap = generate_heatmap(
                model, img_resized, target_class=target_class, layer=layer
            )
            heatmap_up = (
                F.interpolate(
                    heatmap.unsqueeze(0).unsqueeze(0),
                    size=(H, W),
                    mode="bilinear",
                    align_corners=False,
                )
                .squeeze(0)
                .squeeze(0)
            )
            resized_heatmaps.append(heatmap_up)

        stacked = torch.stack(resized_heatmaps, dim=0)
        mean_heatmap = stacked.mean(dim=0)

        hmin = mean_heatmap.min()
        hmax = mean_heatmap.max()
        denom = (hmax - hmin).clamp_min(1e-6)
        super_heatmap = (mean_heatmap - hmin) / denom

        return super_heatmap

    finally:
        if was_training:
            model.train()


def overlay_heatmap(
    image: Union[torch.Tensor, np.ndarray],
    heatmap: torch.Tensor,
    alpha: float = 0.5,
) -> torch.Tensor:
    """
    Overlays a 2D Grad-CAM heatmap onto an input image with both hard and smooth blending,
        and returns the result as a normalized torch tensor.

    Args:
        image (Union[torch.Tensor, np.ndarray]): The input image of shape (C, H, W)
            (torch.Tensor) or (H, W, C) (np.ndarray). Must have 3 channels.
            Values can be in any range, but will be normalized internally.
        heatmap (torch.Tensor): A 2D tensor of shape (H, W) with values in [0, 1],
            representing the normalized Grad-CAM heatmap.
        alpha (float, optional): The blending factor for the hard overlay (default: 0.5).
            Higher values make the heatmap more prominent.

    Returns:
        torch.Tensor: A tensor of shape (3, H, W) with values in [0, 1], representing
            the blended visualization. Returned on the same device as the input `image`.

    Raises:
        AssertionError: If the heatmap is not 2D.
        ValueError: If the heatmap contains values outside the range [0, 1].
    """
    device = image.device if isinstance(image, torch.Tensor) else "cpu"

    # Ensure heatmap is 2D and between 0 and 1
    assert heatmap.dim() == 2, "Heatmap must be a 2D tensor"
    if torch.min(heatmap) < 0 or torch.max(heatmap) > 1:
        raise ValueError("Heatmap values must be between 0 and 1")

    # Convert image to NumPy array and normalize
    img_np = normalize_image_to_range(convert_to_np_array(image), target_range=(0, 255))

    # Convert heatmap to Numpy array
    heatmap = heatmap.detach().cpu().numpy()

    # Resize the heatmap to match the image size
    heatmap_resized = cv2.resize(heatmap, (img_np.shape[1], img_np.shape[0]))

    # Convert heatmap from grayscale (1 channel) to RGB (3 channels)
    heatmap_colored = cv2.applyColorMap(
        np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET
    )  # Apply color map to heatmap
    heatmap_colored = cv2.cvtColor(
        heatmap_colored, cv2.COLOR_BGR2RGB
    )  # Convert color space from BGR (Blue, Green, Red) to RGB

    # Overlay the heatmap on the image
    ov_img = cv2.addWeighted(heatmap_colored, alpha, img_np, 1 - alpha, 0)

    # Smooth overlay using the heatmap values (Transparency)
    heatmap_resized = np.repeat(
        heatmap_resized[:, :, np.newaxis], 3, axis=2
    )  # Repeat the heatmap values across 3 channels
    ov_smooth_img = (ov_img * heatmap_resized + img_np * (1 - heatmap_resized)).astype(
        np.uint8
    )  # Smooth overlay

    # Normalize to [0, 1]
    ov_norm = normalize_image_to_range(ov_smooth_img, target_range=(0, 1))

    # Convert to torch tensor
    ov_tensor = torch.tensor(ov_norm).permute(2, 0, 1).float().to(device)

    return ov_tensor


def generate_and_overlay_bounding_boxes(
    image: Union[torch.Tensor, np.ndarray],
    heatmap: torch.Tensor,
    heatmap_threshold: float = 0.5,
) -> np.ndarray:
    """
    Generates bounding boxes from a Grad-CAM heatmap and overlays them on the input image.

    This function thresholds the heatmap to extract high-activation regions, computes
    bounding boxes around those regions, and draws them onto the image using OpenCV.

    Args:
        image (Union[torch.Tensor, np.ndarray]): The input image of shape (C, H, W) or (H, W, C).
            Values can be in any range, and the image will be normalized internally for
            visualization.
        heatmap (torch.Tensor): A 2D tensor of shape (H, W) containing normalized Grad-CAM
            values in the range [0, 1].
        heatmap_threshold (float, optional): The threshold used to binarize the heatmap for
            contour detection. Defaults to 0.5. Must be between 0 and 1.

    Returns:
        np.ndarray: The input image as a NumPy array with red bounding boxes overlaid
            around high-activation regions.

    Raises:
        AssertionError: If the heatmap is not a 2D tensor.
        ValueError: If the heatmap contains values outside the range [0, 1].

    """
    assert heatmap.ndim == 2, "Heatmap must be a 2D tensor or array"
    if np.min(heatmap) < 0 or np.max(heatmap) > 1:
        raise ValueError("Heatmap values must be in the range [0, 1]")

    img_np = normalize_image_to_range(
        convert_to_np_array(image), target_range=(0, 255)
    ).copy()

    # Convert heatmap to Numpy array
    heatmap = heatmap.detach().cpu().numpy()

    # Resize the heatmap to match the image size
    heatmap_resized = cv2.resize(heatmap, (img_np.shape[1], img_np.shape[0]))

    # Generate binary mask from heatmap
    heatmap_resized = (heatmap_resized * 255).astype(np.uint8)
    _, binary_mask = cv2.threshold(
        heatmap_resized, heatmap_threshold * 255, 255, cv2.THRESH_BINARY
    )

    # Find contours in the binary mask
    contours, _ = cv2.findContours(
        binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    # Draw bounding boxes on the image
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(
            img_np, (x, y), (x + w, y + h), (255, 0, 0), 2
        )  # Red bounding box

    return img_np
