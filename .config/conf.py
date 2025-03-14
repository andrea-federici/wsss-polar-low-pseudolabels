import os
from dotenv import load_dotenv

import torch
from torchvision import transforms


# Data paths
default_data_dir = 'data'

path_displacement = ''

def train_dir() -> str:
    return f'{path_displacement}{default_data_dir}/train'

def test_dir() -> str: 
    return f'{path_displacement}{default_data_dir}/test'


# General settings
verbose = False

# Environment setup
load_dotenv()

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
neptune_api_key = os.getenv("NEPTUNE_API_KEY")
if not neptune_api_key:
    raise ValueError('Neptune API key is missing.')
neptune_project = 'andreaf/polarlows'
log_model_checkpoints = False

# Data settings
resized_image_res = (500, 500)  # Image resolution to use for training

# Training settings

default_train_mode = 'single'
possible_train_modes = ['single', 'single_optuna', 'ft_iterative', 'adv']

batch_size = 16
max_epochs = 100
learning_rate = 0.5e-3

# Normalization settings
# From polar lows dataset
mean = [0.2872, 0.2872, 0.4595]
std = [0.1806, 0.1806, 0.2621]
# From ImageNet
# mean = [0.485, 0.456, 0.406]
# std = [0.229, 0.224, 0.225]

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
