import cv2

import numpy as np
import torch
from captum.attr import IntegratedGradients

from train_config import device
from image_utility import normalize_image


def generate_heatmap(model: torch.nn.Module, input_image: torch.Tensor, target_class: int = -1, n_steps: int = 50, baseline = None):
    model.eval()

    input_image = input_image.to(device) # Move image to the same device as the model

    if baseline is None:
        baseline = torch.zeros_like(input_image) # Use a black image as the baseline is no baseline is provided

    ig = IntegratedGradients(model.forward)

    # Get predicted class
    logits = model(input_image)
    predicted_class = torch.argmax(logits, dim=1).item()

    # If the target class is not specified, use the predicted class
    if target_class == -1:
        target_class = predicted_class

    attributions, _ = ig.attribute(
        input_image,
        baselines=baseline,
        target=target_class,
        n_steps=n_steps,
        return_convergence_delta=True
    )

    print(f'Attributions shape: {attributions.shape}')

    attributions = attributions.squeeze().detach().cpu().numpy() # Convert to NumPy

    # Aggregate attributions across channels
    attributions = np.mean(attributions, axis=0)

    # Normalize the attributions to [0, 1]
    norm_attributions = (attributions - np.min(attributions)) / (np.max(attributions) - np.min(attributions) + 1e-8) # Normalize the attributions

    return norm_attributions, predicted_class
    

def generate_mask_from_heatmap(input_image: torch.Tensor, heatmap, percentile=98, kernel_size=(15,15), min_area=50) -> np.ndarray:
    original_size = input_image.shape[2:] # Assume the input image is in the format (C, H, W)
    heatmap_resized = cv2.resize(heatmap, (original_size[1], original_size[0]), interpolation=cv2.INTER_LINEAR)

    # Binarize the heatmap using a threshold calculated based on the percentile
    heatmap_flattened = heatmap_resized.flatten()
    threshold = np.percentile(heatmap_flattened, percentile)
    binary_mask = np.where(heatmap_resized >= threshold, 1, 0).astype(np.uint8)

    # Morphological Closing: Dilate, then erode to merge dense regions. Finally, blur the mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel_size)
    closed_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
    blurred_mask = cv2.GaussianBlur(closed_mask, (15, 15), 0)

    # Area opening: Remove small objects based on minimum area
    fraction_of_biggest = 0.2
    ao_mask = blurred_mask.copy()
    contours, _ = cv2.findContours(ao_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    areas = [cv2.contourArea(c) for c in contours]
    if areas:
        max_area = max(areas)
        min_area = max_area * fraction_of_biggest
    else:
        min_area = 0
    
    for c in contours:
        if cv2.contourArea(c) < min_area:
            cv2.drawContours(ao_mask, [c], -1, 0, -1)

    # Apply Gaussian blur again to smooth the mask
    final_mask = cv2.GaussianBlur(ao_mask, (31, 31), 0)

    return final_mask


def overlay_mask(input_image: torch.Tensor, mask: np.ndarray, alpha: float = 0.5):
    # Convert the input image to a NumPy array and normalize it
    img_np = input_image.squeeze(0).permute(1, 2, 0).numpy()
    img_np = normalize_image(img_np, target_range=(0, 1))

    # Create a color overlay from the binary mask
    color_overlay = np.zeros_like(img_np)
    color_overlay[mask == 1] = [75, 0, 130]

    # Blend the original image with the color overlay
    img_with_mask = img_np.copy()
    img_with_mask[mask == 1] = cv2.addWeighted(
        color_overlay[mask == 1],
        alpha,
        img_np[mask == 1],
        1 - alpha,
        0
    )
    return img_with_mask


def overlay_heatmap(input_image: torch.Tensor, heatmap: np.ndarray, predicted_class, target_class=-1, alpha: float = 0.6, percentile_neg: float = 97, percentile_pos: float = 98) -> np.ndarray:
    img_np = input_image.squeeze(0).permute(1, 2, 0).numpy()

    heatmap_resized = cv2.resize(heatmap, (img_np.shape[1], img_np.shape[0]))

    # If the target class is not specified, use the predicted class
    if target_class == -1:
        target_class = predicted_class

    # Set the percentile based on the target class
    if target_class == 0:
        percentile = percentile_neg
    else:
        percentile = percentile_pos

    # Flatten the heatmap
    heatmap_flattened = heatmap_resized.flatten()

    # Calculate the threshold based on the percentile
    threshold = np.percentile(heatmap_flattened, percentile)

    # Apply threshold to focus on high attribution regions
    heatmap_thresholded = np.where(heatmap_resized >= threshold, heatmap_resized, 0)

    # Create a green-only heatmap
    green_heatmap = np.zeros_like(img_np)
    green_heatmap[:, :, 1] = (heatmap_thresholded * 255).astype(np.uint8) # Green channel

    # Apply a small Gaussian blur to make dots more visible
    green_heatmap = cv2.GaussianBlur(green_heatmap, (15, 15), 0)

    image_with_heatmap = cv2.addWeighted(green_heatmap, alpha, img_np, 1 - alpha, 0) # Overlay the heatmap on the original image

    return image_with_heatmap
