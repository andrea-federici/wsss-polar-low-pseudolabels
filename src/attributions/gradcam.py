import cv2
import numpy as np
import torch
from captum.attr import LayerGradCam

from data.image_processing import convert_to_np_array, normalize_image_to_range

import warnings

class GradCAM:
    
    def __init__(self, model: torch.nn.Module, device: str):
        self.model = model
        self.device = device

        # TODO: This is fine for now. But if the model is still going to be 
        # used for training after performing GradCAM operations, then it 
        # should be reverted back to its original training mode.
        self.model.eval()
    

    def generate_heatmap_ndarray(
        self, 
        input_image: torch.Tensor, 
        target_class: int = None, 
        layer: torch.nn.Module = None
    ) -> np.ndarray:
        warnings.warn(
            "generate_heatmap_ndarray() is deprecated. Use generate_heatmap() "
            "instead.",
            DeprecationWarning,
            stacklevel=2 # Ensures the warning points to the caller of 
                         # this function rather than the warning line itself
        )

        # If no layer is specified, use the last layer of the feature 
        # extractor
        if layer is None:
            layer = self.model.get_last_conv_layer()

        # Initialize GradCAM with the model and the specified layer
        gradcam = LayerGradCam(self.model, layer)

        # Move image to the same device as the model
        input_image = input_image.to(self.device)

        if target_class is None:
            # Get the predicted class for the input image
            logits = self.model(input_image)
            target_class = torch.argmax(logits, dim=1).item()

        # Compute the GradCAM attributions for the input image
        attr = gradcam.attribute(input_image, target_class)

        attr = attr.squeeze().detach().cpu().numpy() # Convert to NumPy format
        heatmap = np.maximum(attr, 0) # Remove negative values in the heatmap
        norm_heatmap = heatmap / np.max(heatmap) if np.max(heatmap) != 0 else 1 # Normalize the heatmap between 0 and 1

        return norm_heatmap
    

    def generate_heatmap(
        self, 
        image: torch.Tensor, 
        target_class: int = None, 
        layer: torch.nn.Module = None
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
    

    def overlay_heatmap(
        self, image, heatmap: np.ndarray, alpha: float = 0.5
    ) -> np.ndarray:
        # Convert image to NumPy array and normalize
        img_np = normalize_image_to_range(convert_to_np_array(image), target_range=(0, 255))

        # Resize the heatmap to match the image size
        heatmap_resized = cv2.resize(heatmap, (img_np.shape[1], img_np.shape[0]))

        # Convert heatmap from grayscale (1 channel) to RGB (3 channels)
        heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET) # Apply color map to heatmap
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB) # Convert color space from BGR (Blue, Green, Red) to RGB

        # Overlay the heatmap on the image
        ov_img = cv2.addWeighted(heatmap_colored, alpha, img_np, 1 - alpha, 0)

        # Smooth overlay using the heatmap values (Transparency)
        heatmap_resized = np.repeat(heatmap_resized[:, :, np.newaxis], 3, axis=2) # Repeat the heatmap values across 3 channels
        ov_smooth_img = (ov_img * heatmap_resized + img_np * (1 - heatmap_resized)).astype(np.uint8) # Smooth overlay

        return normalize_image_to_range(ov_smooth_img, target_range=(0, 1))
    

    def generate_gradcam_overlay(
            self,
            input_image: torch.Tensor,
            target_class: int = None,
            layer: torch.nn.Module = None,
            alpha: float = 0.5
            ) -> np.ndarray:
        heatmap = self.generate_heatmap_ndarray(input_image, target_class=target_class, layer=layer)
        gradcam_overlay = self.overlay_heatmap(input_image, heatmap, alpha)
        return gradcam_overlay
    

    def generate_and_overlay_bounding_boxes(self, image, heatmap: np.ndarray, heatmap_threshold: float = 0.5) -> np.ndarray:
        img_np = normalize_image_to_range(convert_to_np_array(image), target_range=(0, 255)).copy()
        
        # Resize the heatmap to match the image size
        heatmap_resized = cv2.resize(heatmap, (img_np.shape[1], img_np.shape[0]))
        
        # Generate binary mask from heatmap
        heatmap_resized = (heatmap_resized * 255).astype(np.uint8)
        _, binary_mask = cv2.threshold(heatmap_resized, heatmap_threshold * 255, 255, cv2.THRESH_BINARY)

        # Find contours in the binary mask
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Draw bounding boxes on the image
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(img_np, (x, y), (x + w, y + h), (255, 0, 0), 2)  # Red bounding box
        
        return img_np
