
from cv2 import normalize

import numpy as np
import torch
from PIL import Image

def normalize_image(image, target_range=(0, 1)):
    if not isinstance(image, np.ndarray):
        raise ValueError("Input image should be a NumPy array.")

    min_val = image.min()
    max_val = image.max()
    
    # Normalize to [0, 1]
    normalized_image = (image - min_val) / (max_val - min_val)

    if target_range == (0, 1):
        normalized_image = normalized_image.astype(np.float32)
    elif target_range == (0, 255):
        normalized_image = (normalized_image * 255).astype(np.uint8)
    else:
        raise ValueError("Target range should be either (0, 1) or (0, 255).")
    
    return normalized_image


def convert_to_np_array(image):
    if isinstance(image, np.ndarray):
        return image
    elif isinstance(image, torch.Tensor):
        if image.ndim == 4:
            image = image.squeeze(0) # Remove batch dimension if present
        image = image.permute(1, 2, 0).cpu().numpy() # Convert CHW tensor to HWC NumPy array
        return image
    elif isinstance(image, Image.Image):
        return np.array(image)
    else:
        raise ValueError("Input image must be a PyTorch tensor or PIL image.")
