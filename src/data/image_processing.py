import os
from io import BytesIO
import cv2

import numpy as np
import torch
import torchvision.transforms.functional as T
from PIL import Image
from matplotlib.figure import Figure
from tqdm import tqdm


def normalize_image_to_range(
    image: np.ndarray, target_range=(0, 1)
) -> np.ndarray:
    """
    Normalize a NumPy image array to a specified range.

    This function rescales the pixel values of the input image to either the 
    range [0, 1] or [0, 255].

    Args:
        image (np.ndarray): The input image as a NumPy array.
        target_range (tuple, optional): The desired range for normalization. 
            Supported values are (0, 1) for float32 output and (0, 255) for 
            uint8 output. Defaults to (0, 1).

    Returns:
        np.ndarray: The normalized image array with values in the specified 
        range.

    Raises:
        TypeError: If the input is not a NumPy array.
        ValueError: If the target range is not (0, 1) or (0, 255).
    """
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


def convert_to_np_array(image) -> np.ndarray:
    """
    Convert an image to a NumPy array.

    This function takes an image in various formats (NumPy array, PyTorch 
    tensor, or PIL image) and converts it into a NumPy array with HWC format 
    (Height x Width x Channels).

    Args:
        image (Union[np.ndarray, torch.Tensor, PIL.Image.Image]): The input 
            image, which can be a NumPy array, a PyTorch tensor, or a PIL 
            image.

    Returns:
        np.ndarray: The converted image as a NumPy array in HWC format.

    Raises:
        TypeError: If the input is not a NumPy array, PyTorch tensor, or PIL 
        image.
    """
    if isinstance(image, np.ndarray):
        return image
    elif isinstance(image, torch.Tensor):
        if image.ndim == 4:
            # Remove batch dimension if present
            image = image.squeeze(0)
        # Convert CHW tensor to HWC NumPy array
        image = image.permute(1, 2, 0).cpu().numpy()
        return image
    elif isinstance(image, Image.Image):
        return np.array(image)
    else:
        raise TypeError("Input image must be a PyTorch tensor or PIL image.")


def normalize_image_by_statistics(
    image: torch.Tensor, mean, std
) -> torch.Tensor:
    """
    Normalize a tensor image using the provided mean and standard deviation.
    This function standardizes an image tensor by subtracting the mean and 
    dividing by the standard deviation for each channel.

    Args:
        image (torch.Tensor): The input image tensor with shape (C, H, W), 
            where C is the number of channels.
        mean (array_like): A sequence (e.g., list or tuple) of mean values, 
            one for each channel.
        std (array_like): A sequence (e.g., list or tuple) of standard 
            deviation values, one for each channel.

    Returns:
        torch.Tensor: The normalized image tensor with the same shape 
        as the input. The returned tensor is located on the same device as the 
        input image.

    Raises:
        ValueError: If the length of `mean` or `std` does not match the 
        number of channels in the image tensor.

    """
    # Ensure mean and std have the same number of elements as image channels
    num_channels = image.shape[0]
    if len(mean) != num_channels or len(std) != num_channels:
        raise ValueError(
            f"Length of mean ({len(mean)}) and std ({len(std)}) must match "
            f"the number of channels in the image ({num_channels})."
        )
    
    device = image.device
    
    # Convert mean and std to tensors and reshape to match image dimensions
    mean = torch.tensor(mean, device=device).view(-1, 1, 1)
    std = torch.tensor(std, device=device).view(-1, 1, 1)

    return (image - mean) / std


def unnormalize_image_by_statistics(
    image: torch.Tensor, mean, std
) -> torch.Tensor:
    """
    Unnormalize a tensor image using the provided mean and standard deviation.
    This function restores the original pixel values by scaling the tensor 
    using the provided mean and standard deviation.

    Args:
        image (torch.Tensor): The normalized image tensor with shape 
            (C, H, W), where C is the number of channels.
        mean (array_like): A sequence (e.g., list or tuple) of mean values, 
            one for each channel.
        std (array_like): A sequence (e.g., list or tuple) of standard 
            deviation values, one for each channel.

    Returns:
        torch.Tensor: The unnormalized image tensor with the same shape 
        as the input. The returned tensor is located on the same device as the 
        input image.
    
    Raises:
        ValueError: If the length of `mean` or `std` does not match the 
        number of channels in the image tensor.
    """
    # Ensure mean and std have the same number of elements as image channels
    num_channels = image.shape[0]
    if len(mean) != num_channels or len(std) != num_channels:
        raise ValueError(
            f"Length of mean ({len(mean)}) and std ({len(std)}) must match "
            f"the number of channels in the image ({num_channels})."
        )
    
    device = image.device

    # Convert mean and std to tensors and reshape to match image dimensions
    mean = torch.tensor(mean, device=device).view(-1, 1, 1)
    std = torch.tensor(std, device=device).view(-1, 1, 1)

    return image * std + mean


# TODO: instead of taking the device as argument, use device = image.device
def translate_image(
    image: torch.Tensor,
    translation: tuple[int, int],
    mean,
    std,
    device: str = 'cpu'
) -> torch.Tensor:
    """
    Apply a translation transformation to an image tensor. New areas 
    introduced due to the translation are filled with black (pixel value 0).

    Args:
        image (torch.Tensor): The input image tensor with shape (C, H, W).
        translation (tuple[int, int]): A tuple (dx, dy) specifying 
            the number of pixels to shift the image. A positive `dx` moves 
            the image to the right, while a positive `dy` moves it downward.
            If the values are of type float, they will be floored.
        mean (array_like): A sequence of mean values, one for each channel.
        std (array_like): A sequence of standard deviation values, 
            one for each channel.
        device (str, optional): The device for computation (e.g., "cpu" or 
            "cuda"). Defaults to "cpu".

    Returns:
        torch.Tensor: The translated image tensor.
    """
    # Convert float values to integers. This will floor the values.
    translation = tuple(map(int, translation))

    # Unnormalize the image before applying transformation. We need this so 
    # that the black regions introduced by translation have the correct pixel 
    # values.
    unnormalized_image = unnormalize_image_by_statistics(
        image, mean, std, device=device
    )

    # Apply the translation transformation
    translated_image = T.affine(
        unnormalized_image,
        angle=0, # No rotation
        translate=translation, # (dx, dy): dx -> right, dy -> down
        scale=1.0, # No scaling
        shear=[0.0, 0.0], # No shearing
        fill=[0] # Fill new regions with black
    )

    # Normalize the translated image back to the original range
    normalized_image = normalize_image_by_statistics(
        translated_image, mean, std, device=device
    )

    return normalized_image


def adversarial_erase(
    image: torch.Tensor,
    heatmap: torch.Tensor, 
    threshold: float = 0.7, 
    fill_color=0
) -> torch.Tensor:
    """
    Removes (erases) regions in the image where the corresponding heatmap 
    values exceed a given threshold. The erased areas are replaced with a 
    specified fill color.

    Args:
        image (torch.Tensor):
            A 4D tensor of shape (B, C, H, W) representing an image, where B 
            is the batch size, C is the number of channels, and (H, W) are 
            the spatial dimensions.
        heatmap (torch.Tensor):
            A 2D tensor of shape (h, w) representing the attention map, which 
            indicates which regions should be erased. The heatmap will be 
            resized to match (H, W) of the image.
        threshold (float, optional):
            A threshold value in the range [0, 1] that determines which 
            regions are erased. Pixels in the heatmap greater than this 
            threshold will be erased. Default is 0.7.
        fill_color (int, float, or torch.Tensor, optional):
            The color or value used to fill erased regions. If a scalar (int 
            or float), all erased regions will be filled with this value. If a
            tensor, it must have shape (C,) to specify different colors per 
            channel. Default is 0.

    Returns:
        torch.Tensor:
            A new tensor of the same shape as `image` (B, C, H, W) with the
            specified regions erased.

    Raises:
        AssertionError:
            - If `image` is not a 4D tensor.
            - If `heatmap` is not a 2D tensor.
            - If `fill_color` is a tensor and does not match the image 
                channels.
    """
    assert image.dim() == 4, "Expected image to be a 4D tensor (B, C, H, W), "
    "but got shape {image.shape}"
    assert heatmap.dim() == 2, "Expected heatmap to be a 2D tensor (H, W)"
    
    B, C, H, W = image.shape

    heatmap = T.resize(
        heatmap.unsqueeze(0),  # (1, h, w)
        (H, W),
        interpolation=T.InterpolationMode.BILINEAR,
        antialias=True
    ).unsqueeze(0)  # Now (1, 1, H, W)

    # Expand heatmap: (1, 1, H, W) -> (B, 1, H, W) to match batch 
    # size.
    heatmap = heatmap.expand(B, 1, H, W)

    # If fill_color is a scalar (int or float), convert it to a tensor
    if isinstance(fill_color, (int, float)):
        fill_color = torch.full(
            (B, C, 1, 1),
            fill_color,
            dtype=image.dtype,
            device=image.device
        )
    else:
        # Ensure fill_color has the correct number of channels (C)
        assert fill_color.shape[0] == C, "fill_color should match the number \
            of channels of the image"
        fill_color = fill_color.view(1, C, 1, 1).expand(B, C, H, W)
    
    # Create erase mask: (B, 1, H, W) -> broadcast to (B, C, H, W)
    erase_mask = (heatmap > threshold).expand(B, C, H, W)

    # Clone images to avoid modifying original tensors
    erased_image = image.clone()

    # Apply erasing by filling masked regions with the specified fill color
    erased_image[erase_mask] = fill_color.expand_as(erased_image)[erase_mask]

    return erased_image


def plot_to_pil_image(fig: Figure) -> Image:
    """
    Converts a matplotlib Figure to a PIL Image.

    Args:
        fig (Figure): A matplotlib figure object.

    Returns:
        Image: A PIL Image of the rendered figure.
    """
    # Create an in-memory buffer to store the figure
    buf = BytesIO()

    # Save the Matplotlib figure to the buffer in PNG format
    # 'bbox_inches="tight"' ensures the plot is saved tightly around the 
    # figure content
    # 'dpi' defines the resolution of the saved figure
    fig.savefig(buf, format='PNG', bbox_inches='tight', dpi=100)

    # Move the buffer's cursor back to the beginning to be read
    buf.seek(0)

    # Load the content from the buffer as a PIL Image and convert it to RGB
    # mode
    # The 'convert("RGB")' ensures the image is compatible by converting it 
    # to a 3-channel RGB image
    pil_image = Image.open(buf).convert('RGB')

    # Make an independent copy of the PIL Image to avoid dependencies on the 
    # buffer
    independent_image = pil_image.copy()

    # Close the buffer and original PIL Image to release resources
    buf.close()
    pil_image.close()

    return independent_image


# TODO: at the moment this function is only able to generate masks from the 
# given heatmaps. It does not accumulate the heatmaps. So, it is currently 
# only usable for the iteration 0 heatmaps.
def generate_masks_from_heatmaps(
        heatmap_dir, mask_dir, image_size, threshold=0.5
):
    os.makedirs(mask_dir, exist_ok=True)
    heatmap_filenames = [
        f for f in os.listdir(heatmap_dir) if f.endswith('.pt')
    ]

    print(f'Loaded {len(heatmap_filenames)} heatmaps.')

    for filename in tqdm(heatmap_filenames, desc='Processing heatmaps'):
        heatmap_path = os.path.join(heatmap_dir, filename)
        mask_path = os.path.join(mask_dir, filename.replace('.pt', '.png'))

        heatmap = torch.load(heatmap_path)
        if len(heatmap.shape) > 2:
            heatmap = heatmap.squeeze(0)
        heatmap = heatmap.numpy()

        heatmap = cv2.resize(
            heatmap,
            (image_size[1], image_size[0]), 
            interpolation=cv2.INTER_LINEAR
        )

        # print(f'Path: {heatmap_path}')
        # if not heatmap:
        #     print('Heatmap is None')

        binary_mask = (heatmap > threshold).astype(np.uint8)

        cv2.imwrite(mask_path, binary_mask * 255)

