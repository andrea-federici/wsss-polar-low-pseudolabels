import cv2

import numpy as np
import torch
from torchvision.transforms import ToPILImage
from captum.attr import LayerGradCam

class GradCAM:
    def __init__(self, model, device):
        self.model = model
        self.device = device
    
    # Generate the GradCAM heatmap for the input image
    def generate_gradcam_heatmap(self, input_image, layer=None, verbose=False):
        function_name = 'generate_gradcam_heatmap'

        # If no layer is specified, use the last layer of the feature extractor
        if layer is None:
            layer = self.model.get_last_conv_layer()
            if verbose:
                print(f"[{function_name}] No layer specified. Using last conv layer: {layer}")

        gradcam = LayerGradCam(self.model, layer) # Initialize GradCAM method

        self.model.eval()

        input_image = input_image.to(self.device) # Move image to the same device as the model
        if verbose:
            print(f"[{function_name}] Moved input image to device: {self.device}")
            print(f"[{function_name}] Input image shape after transfer: {input_image.shape}")

        logits = self.model(input_image)
        target_class = torch.argmax(logits, dim=1).item() # Get the predicted class
        if verbose:
            print(f"[{function_name}] Predicted target class: {target_class}")

        attr = gradcam.attribute(input_image, target_class) # Compute the GradCAM attributions
        if verbose:
            print(f"[{function_name}] GradCAM attributions calculated.")
            print(f"[{function_name}] GradCAM attribution shape: {attr.shape}")

        attr = attr.squeeze().detach().cpu().numpy() # Convert to NumPy format
        heatmap = np.maximum(attr, 0) # Remove negative values in the heatmap
        norm_heatmap = heatmap / np.max(heatmap) if np.max(heatmap) != 0 else 1 # Normalize the heatmap

        if verbose:
            print(f"[{function_name}] Generated heatmap after removing negatives and normalization.")
            print(f"[{function_name}] Heatmap shape: {norm_heatmap.shape}")
            print(f"[{function_name}] Heatmap max value after normalization: {np.max(norm_heatmap)}")

        return norm_heatmap
    
    # Visualize the heatmap overlay on the original image
    def overlay_heatmap(self, original_image, heatmap, alpha: float = 0.4, colormap: int = cv2.COLORMAP_JET):
        image_np = np.array(original_image) # Convert the PIL image to a NumPy array

        heatmap_resized = cv2.resize(heatmap, (image_np.shape[1], image_np.shape[0])) # Resize the heatmap to match the original image size

        heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), colormap) # Apply color map to heatmap
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB) # Convert color space from BGR (Blue, Green, Red) to RGB

        image_with_heatmap = cv2.addWeighted(heatmap_colored, alpha, image_np, 1 - alpha, 0) # Overlay the heatmap on the original image

        image_with_heatmap_pil = ToPILImage()(image_with_heatmap) # Convert the NumPy array back to PIL image

        return image_with_heatmap_pil
    
    def generate_and_overlay_bounding_boxes(self, original_image, heatmap, alpha: float = 0.4, colormap: int = cv2.COLORMAP_JET, bb_threshold=0.5):
        # Convert PIL image to NumPy array for OpenCV processing
        image_np = np.array(original_image)
        
        # Resize the heatmap to the original image size for overlay purposes
        heatmap_resized = cv2.resize(heatmap, (image_np.shape[1], image_np.shape[0]))
        heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), colormap)
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
        
        # Overlay the heatmap on the original image
        # image_with_heatmap = cv2.addWeighted(heatmap_colored, alpha, image_np, 1 - alpha, 0)
        
        # Generate binary mask from heatmap
        _, binary_mask = cv2.threshold(heatmap_resized, bb_threshold, 1, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(binary_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Draw bounding boxes on the overlayed image
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(image_np, (x, y), (x + w, y + h), (255, 0, 0), 2)  # Red bounding box
        
        # Convert the image back to PIL format for consistency
        final_image_with_boxes = ToPILImage()(image_np)
        
        return final_image_with_boxes
