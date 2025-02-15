import os
from dotenv import load_dotenv

import torch
from torchvision import transforms


path_displacement = ''

# Data paths
default_data_dir = 'data'

def train_dir():
    return f'{path_displacement}{default_data_dir}/train'

def test_dir(): 
    return f'{path_displacement}{default_data_dir}/test'


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

num_workers = 8 if device.type == 'cuda' else 6  # 8 should be optimal if GPU
                                                 # is available, 6 for CPU.
if verbose:
    print(f"Number of workers: {num_workers}")


# Neptune logger
log_model_checkpoints = False

# Data settings
resized_image_res = (800, 800)  # Image resolution to use for training

# Training settings

default_train_mode = 'single'
possible_train_modes = ['single', 'single_optuna', 'ft_iterative']

batch_size = 32
max_epochs = 100
learning_rate = 1e-3

# Normalization settings
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

# Transformations for the training set
aug_translate_frac = 0.4 # Using Optuna, the best value was found to be 0.4113
transform_aug = transforms.Compose([
    transforms.RandomAffine(
        degrees=20,
        translate=(aug_translate_frac, aug_translate_frac),
        scale=(0.9, 1.1),
        fill=0),
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


# Fine-tuning
ft_learning_rate = 1e-5
max_iterations = 10
