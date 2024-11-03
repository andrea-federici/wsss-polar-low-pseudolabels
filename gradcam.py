import cv2

import numpy as np
import torch
from captum.attr import LayerGradCam

from image_utility import convert_to_np_array, normalize_image

class GradCAM:
    
    def __init__(self, model: torch.nn.Module, device):
        self.model = model
        self.device = device
    

    def generate_heatmap(self, input_image: torch.Tensor, layer: torch.nn.Module = None) -> np.ndarray:
        """
        Generates a GradCAM heatmap for the given input image using the specified layer of the model.

        This method computes the GradCAM attributions for the input image, visualizing the regions 
        that the model focuses on when making a prediction. If no layer is specified, the function 
        defaults to using the last convolutional layer of the model's feature extractor.
        
        Parameters:
            input_image (torch.Tensor): The input image tensor for which the heatmap will be generated. 
                                        It should be preprocessed and shaped correctly for the model.
            layer (torch.nn.Module, optional): The convolutional layer to use for GradCAM. 
                                If None, the last convolutional layer of the model will be used.

        Returns:
            numpy.ndarray: A normalized heatmap array representing the areas of importance 
                        in the input image. The values are scaled between 0 and 1, 
                        where 1 indicates the highest importance.
        """
        
        # If no layer is specified, use the last layer of the feature extractor
        if layer is None:
            layer = self.model.get_last_conv_layer()

        # Initialize GradCAM with the model and the specified layer
        gradcam = LayerGradCam(self.model, layer)

        self.model.eval()

        # Move image to the same device as the model
        input_image = input_image.to(self.device)

        # Get the predicted class for the input image
        logits = self.model(input_image)
        target_class = torch.argmax(logits, dim=1).item()

        # Compute the GradCAM attributions for the input image
        attr = gradcam.attribute(input_image, target_class)

        attr = attr.squeeze().detach().cpu().numpy() # Convert to NumPy format
        heatmap = np.maximum(attr, 0) # Remove negative values in the heatmap
        norm_heatmap = heatmap / np.max(heatmap) if np.max(heatmap) != 0 else 1 # Normalize the heatmap between 0 and 1

        return norm_heatmap
    

    def overlay_heatmap(self, image, heatmap: np.ndarray, alpha: float = 0.5) -> np.ndarray:
        """
        Overlays a heatmap onto an image, blending them to create a smooth overlay effect.
        
        Parameters
        ----------
        image : PIL.Image, np.ndarray, or torch.Tensor
            The input image onto which the heatmap will be overlayed. This can be a PIL Image, 
            a NumPy array, or a PyTorch tensor. If a tensor, it should be in the format 
            (C, H, W) or (B, C, H, W).
        
        heatmap : np.ndarray
            A 2D array representing the heatmap to overlay. This should be a single-channel 
            grayscale heatmap with values normalized between 0 and 1.
            
        alpha : float, optional, default=0.5
            The blending factor for the heatmap overlay. A value of 1.0 will use the heatmap fully, 
            while 0.0 will use only the original image.
            
        Returns
        -------
        np.ndarray
            The resulting image with the heatmap overlay, as a NumPy array in RGB format.
        """
        # Convert image to NumPy array and normalize
        img_np = normalize_image(convert_to_np_array(image), target_range=(0, 255))

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

        return ov_smooth_img
    

    def generate_and_overlay_bounding_boxes(self, image, heatmap: np.ndarray, heatmap_threshold: float = 0.5) -> np.ndarray:
        """
        Generates bounding boxes around regions in a heatmap that exceed a specified threshold 
        and overlays these bounding boxes onto an image.

        Parameters
        ----------
        image : PIL.Image, np.ndarray, or torch.Tensor
            The input image on which to overlay bounding boxes. This can be a PIL Image,
            a NumPy array, or a PyTorch tensor. If a tensor, it should be in the format
            (C, H, W) or (B, C, H, W).
        
        heatmap : np.ndarray
            A 2D array representing the heatmap used to identify regions of interest.
            The heatmap should have values normalized between 0 and 1.

        heatmap_threshold : float, optional, default=0.5
            A threshold value for the heatmap. Pixels in the heatmap above this value are 
            considered part of a region of interest and are used to generate bounding boxes.
            
        Returns
        -------
        np.ndarray
            The resulting image with bounding boxes overlayed, as a NumPy array in RGB format.
        """
        img_np = normalize_image(convert_to_np_array(image), target_range=(0, 255)).copy()
        
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
