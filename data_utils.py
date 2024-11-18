import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from sklearn.model_selection import StratifiedShuffleSplit
import torch
from torch.utils.data import Subset
from torchvision import transforms
from collections import Counter
from PIL import Image


def dataset_stratified_shuffle_split(dataset, test_size=0.2, random_state=42):
    """Split the dataset into training and test sets while preserving class proportions."""
    split = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    labels = dataset.targets

    train_data = None
    train_indices, val_indices = [], []
    for train_indices, val_indices in split.split(np.zeros(len(labels)), labels):
        train_indices = train_indices.tolist()
        val_indices = val_indices.tolist()
        train_data = Subset(dataset, train_indices)

    return train_data, train_indices, val_indices


def dataset_calculate_class_weights(dataset, train_indices):
    """Calculate class weights based on the dataset and the indices used for training."""
    train_counts = Counter([dataset.targets[i] for i in train_indices])

    # Calculate class weights
    class_weights_not_normalized = { cls: 1.0 / count for cls, count in train_counts.items() }

    # Normalize the weights
    total_weights = sum(class_weights_not_normalized.values())
    class_weights = { cls: weight / total_weights for cls, weight in class_weights_not_normalized.items() }
    
    return class_weights


def dataset_print_stats(dataset, dataset_name=""):
    """Print statistics about the dataset."""
    # Retrive targes, class information, and class-to-index mapping
    targets, classes, class_to_idx = dataset_get_targets_and_classes(dataset)

    # Count the occurrences of each class in the dataset
    dataset_counts = Counter(targets)

    # Get class indices for 'pos' and 'neg'
    pos_idx = class_to_idx['pos']
    neg_idx = class_to_idx['neg']

    # Get the number of positive and negative samples
    pos_count = dataset_counts[pos_idx]
    neg_count = dataset_counts[neg_idx]
    total_count = len(targets)

    # Calculate the class ratio and percentages
    class_ratio = pos_count / neg_count if neg_count > 0 else float('inf')
    pos_percentage = (pos_count / total_count) * 100
    neg_percentage = (neg_count / total_count) * 100

    # Print dataset statistics
    print(f"'{dataset_name}' dataset:")
    print(f"\tNumber of samples: {total_count} (neg: {neg_count}, pos: {pos_count})")
    print(f"\tNumber of classes: {len(classes)}")
    print(f"\tClass names: {classes}")
    print(f"\tClass distribution ratio (pos:neg): {class_ratio:.2f}")
    print(f"\tClass percentages: {pos_percentage:.2f}% pos, {neg_percentage:.2f}% neg")
    print()


def dataset_get_targets_and_classes(dataset):
    """Get the targets, classes, and class-to-index mapping from the dataset."""
    if isinstance(dataset, Subset):
        original_dataset = dataset.dataset
        subset_indices = dataset.indices
        targets = [original_dataset.targets[i] for i in subset_indices]
        classes = original_dataset.classes
        class_to_idx = dataset.dataset.class_to_idx
    else:
        targets = dataset.targets
        classes = dataset.classes
        class_to_idx = dataset.class_to_idx
    return targets, classes, class_to_idx


def pick_random_image(train_dir: str, class_name: str = ""):
    if class_name == "":
        class_name = np.random.choice(['pos', 'neg'])
    class_dir = os.path.join(train_dir, class_name)
    return os.path.join(class_dir, np.random.choice(os.listdir(class_dir)))


def load_and_transform_image(image_path: str, data_transform: transforms.Compose) -> torch.Tensor:
    image = Image.open(image_path).convert('RGB')
    transformed_image = data_transform(image)
    assert isinstance(transformed_image, torch.Tensor), 'Data preparation should return a torch Tensor.'
    return transformed_image.unsqueeze(0)

