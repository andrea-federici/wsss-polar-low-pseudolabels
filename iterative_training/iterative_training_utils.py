
import os

from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
import torch
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

from train_config import (
    device,
    resized_image_res,
    mean,
    std,
    transform_prep,
)
from image_utility import translate_image, normalize_image


def find_maximum_translations(model, image, resolution = -1, initial_step_size=-1, threshold=0.5):
    if not isinstance(image, torch.Tensor):
        raise TypeError('Image should be a torch Tensor.')
    
    if not image.dim() == 4:
        raise ValueError('Image should be a 4D tensor.')
    
    _, _, h, w = image.shape

    directions = {'up': (0, -1), 'down': (0, 1), 'left': (-1, 0), 'right': (1, 0)}
    max_translations = {'up': 0, 'down': 0, 'left': 0, 'right': 0}
    translation_limit = {'up': h, 'down': h, 'left': w, 'right': w}

    model.eval()

    with torch.no_grad():
        logits = model(image)
        initial_confidence = torch.softmax(logits, dim=1)[:, 1].item()

    # If the image is initially classified as "negative" (confidence < threshold), return zeros
    if initial_confidence < threshold:
        return max_translations

    for direction, (dx, dy) in directions.items():
        current_translation_limit = translation_limit[direction]
        if initial_step_size == -1:
            step_size = current_translation_limit // 4
        else:
            step_size = initial_step_size

        if resolution == -1:
            min_step_size = current_translation_limit // 32
            if min_step_size < 1:
                min_step_size = 1
        else:
            min_step_size = resolution

        current_translation = 0

        while step_size >= min_step_size:
            found_boundary = False

            while current_translation < current_translation_limit + step_size:
                translated_image = translate_image(image, (dx * current_translation, dy * current_translation), mean, std)

                with torch.no_grad():
                    logits = model(translated_image)
                    confidence = torch.softmax(logits, dim=1)[:, 1].item()

                if confidence < threshold:
                    found_boundary = True
                    break

                current_translation += step_size

            if found_boundary:
                current_translation -= step_size
                step_size //= 2
            else:
                break

        max_translations[direction] = current_translation

    return max_translations


def batch_find_maximum_translations(model, batch_dir: str, data_transform: transforms.Compose, resolution: int = -1, initial_step_size: int = -1, threshold: float = 0.5):
    batch_max_translations = {}

    image_paths = [os.path.join(batch_dir, fname) for fname in os.listdir(batch_dir)]

    for image_path in tqdm(image_paths, desc='Calculating maximum translation values for images in the batch directory'):
        image_filename = os.path.basename(image_path)

        transformed_image = data_transform(Image.open(image_path).convert('RGB'))
        assert isinstance(transformed_image, torch.Tensor), 'Data preparation should return a torch Tensor.'

        transformed_image = transformed_image.to(device).unsqueeze(0)

        img_max_translations = find_maximum_translations(model, transformed_image, resolution, initial_step_size, threshold)
        batch_max_translations[image_filename] = img_max_translations

    return batch_max_translations


def bounding_box_from_max_translations(max_translations, image_size = None):
    if image_size is None:
        image_size = resized_image_res

    # Calculate bounding box coordinates based on max translations
    w, h = image_size
    # x_min = max(0, w - max_translations['left'])
    # y_min = max(0, h - max_translations['up'])
    # x_max = min(w, max_translations['right'])
    # y_max = min(h, max_translations['down'])
    x_min = max(0, w - max_translations['right'])
    y_min = max(0, h - max_translations['down'])
    x_max = min(w, max_translations['left'])
    y_max = min(h, max_translations['up'])

    if x_max <= x_min or y_max <= y_min:
        return 0, 0, 0, 0
  
    return x_min, y_min, x_max, y_max


def area_from_max_translations(max_translations, image_size = None):
    if image_size is None:
        image_size = resized_image_res
    x_min, y_min, x_max, y_max = bounding_box_from_max_translations(max_translations, image_size)
    if x_max <= x_min or y_max <= y_min:
        return 0
    return (x_max - x_min) * (y_max - y_min)


def average_bounding_box_area(max_translations_dict, image_size=None):
    if image_size is None:
        image_size = resized_image_res
    
    total_area = 0
    num_entries = len(max_translations_dict)

    if num_entries == 0:
        return 0 # Avoid division by zero
    
    # Calculate the total area of bounding boxes
    for max_translations in max_translations_dict.values():
        total_area += area_from_max_translations(max_translations, image_size)
    
    return total_area / num_entries


def plot_bounding_box(image: torch.Tensor, bounding_box):
    image_np = normalize_image(image.squeeze(0).permute(1, 2, 0).cpu().numpy())
    x_min, y_min, x_max, y_max = bounding_box
    plt.imshow(image_np)
    plt.gca().add_patch(Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, fill=False, edgecolor='red', linewidth=2))
    plt.show()


def plot_non_discriminative_regions(image: torch.Tensor, max_translations):
    # Ensure the image is a 4D tensor and extract height and width
    if image.dim() != 4:
        raise ValueError('Image should be a 4D torch tensor (batch, channels, height, width).')

    # Remove batch dimension and convert to numpy for plotting
    image_np = image.squeeze(0).permute(1, 2, 0).cpu().numpy()  # [H, W, C]
    h, w, _ = image_np.shape

    _, ax = plt.subplots(1)
    ax.imshow(normalize_image(image_np))

    # Define colors for each region
    colors = {
        'left': 'blue',
        'right': 'green',
        'up': 'orange',
        'down': 'red',
    }

    # Define non-discriminative regions based on max_translations
    regions = {
        'left': (0, 0, w - max_translations['right'], h),  # x, y, width, height
        'right': (max_translations['left'], 0, w - max_translations['left'], h),
        'up': (0, 0, w, h - max_translations['down']),
        'down': (0, max_translations['up'], w, h - max_translations['up']),
    }

    # Add semi-transparent rectangles for each non-discriminative region
    for direction, (x, y, width, height) in regions.items():
        if width > 0 and height > 0:  # Only draw if there is an area to shade
            rect = Rectangle((x, y), width, height, linewidth=1,
                                     edgecolor=colors[direction], facecolor=colors[direction],
                                     alpha=0.2, label=f'Non-discriminative region ({direction})')
            ax.add_patch(rect)

    # Add legend to distinguish regions
    ax.legend(loc='upper right', fontsize=10)

    # Add title and show the plot
    ax.set_title("Non-discriminative Regions")
    plt.axis('off')
    plt.tight_layout()
    plt.show()



# ------------------------------ TESTING ------------------------------

# import torch.nn as nn
# import torch.optim as optim

# from models import XceptionModel
# from model_container import ModelContainer
# from data_utils import load_and_transform_image, pick_random_image
# from image_utility import translate_image


# torch_model = XceptionModel(num_classes=2)

# criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(filter(lambda p: p.requires_grad, torch_model.parameters()), lr=0.001)

# model = ModelContainer.load_from_checkpoint('checkpoints/xception-xaug.ckpt', model=torch_model, criterion=criterion, optimizer=optimizer)

# # image_path = pick_random_image('data/train', 'pos')
# image_path = 'data/train/pos/5d6964_20160902T082858_20160902T083102_mos_rgb.png'
# print(image_path)
# image = load_and_transform_image(image_path, transform_prep)
# image = translate_image(image, (100, 150), mean, std)
# max_translations = find_maximum_translations(model, image)
# print(max_translations)
# bb = bounding_box_from_max_translations(max_translations)
# print(bb)
# print(area_from_max_translations(max_translations))

# plot_non_discriminative_regions(image, max_translations)
