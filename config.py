
from matplotlib.pylab import f
import torch

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

