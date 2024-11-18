
import torch
from torchvision import transforms

# Data paths
train_dir = 'data/train'
test_dir = 'data/test'

# General settings
verbose = False

# Hardware settings
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_workers = 8 if device.type == 'cuda' else 6  # 8 should be optimal if GPU is available, 6 for CPU.
if verbose:
    print(f"Using device: {device}")
    print(f"Number of workers: {num_workers}")

# Data settings
resized_image_res = (299, 299)  # Image resolution to use for training

# Normalization settings
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

transform_aug = transforms.Compose([
    transforms.RandomAffine(degrees=45, translate=(0.35, 0.35), scale=(0.8, 1.2), fill=0),  # Random Rotation, Translation, and Zoom
    transforms.RandomHorizontalFlip(),  # Random Horizontal Flip
    transforms.RandomVerticalFlip(),  # Random Vertical Flip
    transforms.Resize(resized_image_res),  # Resize
    transforms.ToTensor(),  # Convert to tensor
    transforms.Normalize(mean=mean, std=std)  # Rescaling / Normalizing
])

# Transformations for the validation and test sets
transform_prep = transforms.Compose([
    transforms.Resize(resized_image_res),
    transforms.ToTensor(),  # Convert to tensor
    transforms.Normalize(mean=mean, std=std)  # Rescaling / Normalizing
])

