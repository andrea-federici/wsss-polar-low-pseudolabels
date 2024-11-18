
import numpy as np
import torch
import torchvision.transforms.functional as T
from PIL import Image

from train_config import device

def normalize_image(image, target_range=(0, 1)):
    if not isinstance(image, np.ndarray):
        raise TypeError("Input image should be a NumPy array.")

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
        raise TypeError("Input image must be a PyTorch tensor or PIL image.")


def unnormalize_image_mean_std(image, mean, std):
    """Unnormalize a tensor image using the provided mean and std."""
    mean = torch.tensor(mean, device=device).view(-1, 1, 1)
    std = torch.tensor(std, device=device).view(-1, 1, 1)
    return image * std + mean


def normalize_image_mean_std(image, mean, std):
    """Normalize a tensor image using the provided mean and std."""
    mean = torch.tensor(mean, device=device).view(-1, 1, 1)
    std = torch.tensor(std, device=device).view(-1, 1, 1)
    return (image - mean) / std


def translate_image(image: torch.Tensor, translation_amount, mean, std):
    unnormalized_image = unnormalize_image_mean_std(image, mean, std)

    translated_image = T.affine(
        unnormalized_image,
        angle=0, # No rotation
        translate=translation_amount, # Translation in pixels
        scale=1.0, # No scaling
        shear=[0.0, 0.0], # No shearing
        fill=[0] # Fill new regions with black
    )

    normalized_image = normalize_image_mean_std(translated_image, mean, std)

    return normalized_image


