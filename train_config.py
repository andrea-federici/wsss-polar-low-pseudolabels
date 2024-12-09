import os
from dotenv import load_dotenv

import torch
from torchvision import transforms


# Data paths
default_data_dir = 'data'

def get_train_dir(data_dir: str):
    return f'{data_dir if data_dir else default_data_dir}/train'

def get_test_dir(data_dir: str): 
    return f'{data_dir if data_dir else default_data_dir}/test'


# General settings
verbose = False

# Environment setup
load_dotenv()
neptune_api_key = os.getenv("NEPTUNE_API_KEY")
if not neptune_api_key:
    raise ValueError('Neptune API key is missing.')

# Hardware settings
cuda_available = torch.cuda.is_available()
accelerator = 'gpu' if cuda_available else 'cpu'
device = torch.device("cuda" if cuda_available else "cpu")
print(f"Using device: {device}")

num_workers = 8 if device.type == 'cuda' else 6  # 8 should be optimal if GPU is available, 6 for CPU.
if verbose:
    print(f"Number of workers: {num_workers}")

# Data settings
resized_image_res = (299, 299)  # Image resolution to use for training

# Training settings
batch_size = 32
max_epochs = 100
learning_rate = 1e-3

# Normalization settings
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

# Transformations for the training set
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


# Callbacks

# Early Stopping
es_patience = 10

# Model Checkpoint
cc_filename = 'best-checkpoint'


# Learning Rate Scheduler
lr_patience = 5
lr_factor = 0.5
