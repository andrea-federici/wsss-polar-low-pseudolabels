import cv2

import numpy as np
import torch
from torchvision.transforms import ToPILImage
from captum.attr import LayerGradCam
from PIL import Image

from image_utility import normalize_image

class GradCAM:
    
    def __init__(self, model, device):
        self.model = model
        self.device = device
    
    # Generate the GradCAM heatmap for the input image
    def generate_gradcam_heatmap(self, input_image, layer=None):
        """
        Generate a GradCAM heatmap for the given input image using the specified layer of the model.

        This method computes the GradCAM attributions for the input image, visualizing the regions 
        that the model focuses on when making a prediction. If no layer is specified, the function 
        defaults to using the last convolutional layer of the model's feature extractor.
        
        Parameters:
            input_image (torch.Tensor): The input image tensor for which the heatmap will be generated. 
                                        It should be preprocessed and shaped correctly for the model.
            layer (str, optional): The name of the convolutional layer to use for GradCAM. 
                                If None, the last convolutional layer of the model will be used.

        Returns:
            numpy.ndarray: A normalized heatmap array representing the areas of importance 
                        in the input image. The values are scaled between 0 and 1, 
                        where 1 indicates the highest importance.
        """
        
        # If no layer is specified, use the last layer of the feature extractor
        if layer is None:
            layer = self.model.get_last_conv_layer()

        gradcam = LayerGradCam(self.model, layer) # Initialize GradCAM method

        self.model.eval()

        input_image = input_image.to(self.device) # Move image to the same device as the model

        logits = self.model(input_image)
        target_class = torch.argmax(logits, dim=1).item() # Get the predicted class

        attr = gradcam.attribute(input_image, target_class) # Compute the GradCAM attributions

        attr = attr.squeeze().detach().cpu().numpy() # Convert to NumPy format
        heatmap = np.maximum(attr, 0) # Remove negative values in the heatmap
        norm_heatmap = heatmap / np.max(heatmap) if np.max(heatmap) != 0 else 1 # Normalize the heatmap

        return norm_heatmap
    
    # Visualize the heatmap overlay on the original image
    def overlay_heatmap(self, original_image, heatmap, alpha: float = 0.4, colormap: int = cv2.COLORMAP_JET):
        if isinstance(original_image, torch.Tensor):
            # Convert PyTorch tensor to NumPy array
            if original_image.ndim == 4:
                original_image = original_image.squeeze(0) # Remove batch dimension if present
            original_image = original_image.permute(1, 2, 0).cpu().numpy() # Convert CHW tensor to HWC NumPy array
            original_image = normalize_image(original_image, target_range=(0, 255))
        elif isinstance(original_image, Image.Image):
            # Convert PIL image to NumPy array. Remember that PIL contains images in RGB format (0-255).
            original_image = np.array(original_image)
        else:
            raise ValueError("Input image must be a PyTorch tensor or PIL image.")

        heatmap_resized = cv2.resize(heatmap, (original_image.shape[1], original_image.shape[0])) # Resize the heatmap to match the original image size

        heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), colormap) # Apply color map to heatmap
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB) # Convert color space from BGR (Blue, Green, Red) to RGB

        image_with_heatmap = cv2.addWeighted(heatmap_colored, alpha, original_image, 1 - alpha, 0) # Overlay the heatmap on the original image

        return image_with_heatmap
    
    def generate_and_overlay_bounding_boxes(self, original_image, heatmap, alpha: float = 0.4, colormap: int = cv2.COLORMAP_JET, bb_threshold=0.5):
        # Convert PyTorch tensor to NumPy array
        if original_image.ndim == 4:
            original_image = original_image.squeeze(0) # Remove batch dimension if present
        original_image = original_image.permute(1, 2, 0).cpu().numpy() # Convert CHW tensor to HWC NumPy array
        original_image = normalize_image(original_image, target_range=(0, 255))
        
        # Resize the heatmap to the original image size for overlay purposes
        heatmap_resized = cv2.resize(heatmap, (original_image.shape[1], original_image.shape[0]))
        
        heatmap_resized = (heatmap_resized * 255).astype(np.uint8)
        _, binary_mask = cv2.threshold(heatmap_resized, bb_threshold * 255, 255, cv2.THRESH_BINARY)

        # Generate binary mask from heatmap
        # _, binary_mask = cv2.threshold(heatmap_resized, bb_threshold, 1, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(binary_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Draw bounding boxes on the overlayed image
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(original_image.astype(np.uint8), (x, y), (x + w, y + h), (255, 0, 0), 2)  # Red bounding box
        
        # Convert the image back to PIL format for consistency
        final_image_with_boxes = ToPILImage()(original_image)
        
        return final_image_with_boxes
