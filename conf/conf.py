import torch

# Data paths
default_data_dir = 'data'

path_displacement = ''

def train_dir() -> str:
    return f'{path_displacement}{default_data_dir}/train'

def test_dir() -> str: 
    return f'{path_displacement}{default_data_dir}/test'

# Hardware settings
cuda_available = torch.cuda.is_available()
accelerator = 'gpu' if cuda_available else 'cpu'
device = torch.device("cuda" if cuda_available else "cpu")
print(f"Using device: {device}")

num_workers = 8 if device.type == 'cuda' else 6  # 8 should be optimal if GPU
                                                 # is available, 6 for CPU.

# Training settings

batch_size = 16
max_epochs = 100
learning_rate = 0.5e-3


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
