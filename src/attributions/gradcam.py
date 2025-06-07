from typing import Sequence, Tuple, Union

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from captum.attr import LayerGradCam

from src.data.image_processing import convert_to_np_array, normalize_image_to_range


class GradCAM:

    def __init__(self, model: torch.nn.Module, device: str):
        self.model = model
        self.device = device

        # TODO: This is fine for now. But if the model is still going to be
        # used for training after performing GradCAM operations, then it
        # should be reverted back to its original training mode.
        # Look here: https://stackoverflow.com/questions/65344578/how-to-check-if-a-model-is-in-train-or-eval-mode-in-pytorch
        # Should probably not use a class but just individual methods
        self.model.eval()

    def generate_heatmap(
        self,
        image: torch.Tensor,
        target_class: int = None,
        layer: torch.nn.Module = None,
    ) -> torch.Tensor:
        """
        Generates a Grad-CAM heatmap for a given input image and target class.
        The heatmap is normalized between 0 and 1.

        Args:
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

        # If no layer is specified, use the last layer of the feature
        # extractor
        if layer is None:
            layer = self.model.get_last_conv_layer()

        # Initialize GradCAM with the model and the specified layer
        gradcam = LayerGradCam(self.model, layer)

        # Move image to the same device as the model
        image = image.to(self.device)

        if target_class is None:
            # Get the predicted class for the input image
            logits = self.model(image)
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

    # When using this method we want to pass the image at the highest possible resolution
    def generate_super_heatmap(
        self,
        image: torch.Tensor,
        target_size: Tuple[int],  # (H, W)
        sizes: Sequence[int],
        target_class: int,
        layer: torch.nn.Module = None,
    ) -> torch.Tensor:
        if image.dim() != 4:
            raise ValueError(
                f"Expected image to be a 4D tensor (1, C, H, W), but "
                f"got shape {image.shape}"
            )

        # If no layer is specified, use the last layer of the feature
        # extractor
        if layer is None:
            layer = self.model.get_last_conv_layer()

        H, W = target_size
        resized_heatmaps = []

        for s in sizes:
            img_resized = F.interpolate(
                image, size=(s, s), mode="bilinear", align_corners=False
            )
            heatmap = self.generate_heatmap(
                img_resized, target_class=target_class, layer=layer
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

    def overlay_heatmap(
        self,
        image: Union[torch.Tensor, np.ndarray],
        heatmap: torch.Tensor,
        alpha: float = 0.5,
    ) -> torch.Tensor:
        # Convert image to NumPy array and normalize
        img_np = normalize_image_to_range(
            convert_to_np_array(image), target_range=(0, 255)
        )

        # Ensure heatmap is 2D and between 0 and 1
        assert heatmap.dim() == 2, "Heatmap must be a 2D tensor"
        if torch.min(heatmap) < 0 or torch.max(heatmap) > 1:
            raise ValueError("Heatmap values must be between 0 and 1")

        # Convert heatmap to Numpy array
        heatmap = heatmap.cpu().numpy()

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
        ov_smooth_img = (
            ov_img * heatmap_resized + img_np * (1 - heatmap_resized)
        ).astype(
            np.uint8
        )  # Smooth overlay

        # Normalize to [0, 1]
        ov_norm = normalize_image_to_range(ov_smooth_img, target_range=(0, 1))

        # Convert to torch tensor
        ov_tensor = torch.tensor(ov_norm).permute(2, 0, 1).float()

        return ov_tensor

    def generate_and_overlay_bounding_boxes(
        self, image, heatmap: np.ndarray, heatmap_threshold: float = 0.5
    ) -> np.ndarray:
        img_np = normalize_image_to_range(
            convert_to_np_array(image), target_range=(0, 255)
        ).copy()

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
