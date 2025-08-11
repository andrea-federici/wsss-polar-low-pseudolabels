import os
from typing import Optional, Tuple

import torch
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler, default_collate
from torchvision import datasets, transforms

from src import data

# Registry for supported dataset types
_DATASET_REGISTRY = {
    "standard": {
        "dataset_class": datasets.ImageFolder,
        "kwargs": lambda data_dir, transform: {
            "root": data_dir,
            "transform": transform,
        },
    },
    "adversarial_erasing": {
        "dataset_class": data.custom_datasets.ImageFilenameDataset,
        "kwargs": lambda data_dir, transform: {
            "root": data_dir,
            "transform": transform,
        },
    },
}


def create_data_loaders(
    data_dir: str,
    *,
    batch_size: int,
    num_workers: int,
    transform_train: transforms.Compose,
    transform_val: transforms.Compose,
    transform_test: Optional[transforms.Compose] = None,
    dataset_type: str = "standard",
    pin_memory: bool = False,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Creates and returns DataLoaders for training, validation, and testing datasets.

    This function loads image datasets from the given directory, applies the appropriate
    transformations, performs a stratified split of the training data into training and
    validation sets, computes class weights for the training sampler, and returns
    PyTorch DataLoaders for each split.

    Args:
        data_dir (str): Path to the root data directory containing 'train/' and 'test/' subdirectories.
        batch_size (int): Number of samples per batch to load.
        num_workers (int): Number of subprocesses to use for data loading.
        transform_train (transforms.Compose): Transformations to apply to the training data.
        transform_val (transforms.Compose): Transformations to apply to the validation data.
        transform_test (Optional[transforms.Compose], optional): Transformations to apply to the test data.
            If not provided, `transform_val` will be used.
        dataset_type (str, optional): Type of dataset to load. Must be a key in `_DATASET_REGISTRY`.
            Defaults to "standard".
        pin_memory (bool, optional): If True, the data loader will copy Tensors into CUDA pinned memory
            before returning them. Useful when using GPU training. Defaults to False.

    Returns:
        Tuple[DataLoader, DataLoader, DataLoader]: A tuple containing:
            - `train_loader`: DataLoader for the training set, using weighted sampling for class balance.
            - `val_loader`: DataLoader for the validation set.
            - `test_loader`: DataLoader for the test set.

    Raises:
        ValueError: If `dataset_type` is not found in `_DATASET_REGISTRY`.
    """
    # Validate dataset type
    if dataset_type not in _DATASET_REGISTRY:
        valid = ", ".join(_DATASET_REGISTRY.keys())
        raise ValueError(
            f"Unsupported dataset type '{dataset_type}'. Valid types are: {valid}"
        )

    # Prepare directories
    train_dir = os.path.join(data_dir, "train")
    test_dir = os.path.join(data_dir, "test")

    # Use validation transform for test if test transform not provided
    transform_test = transform_test or transform_val

    ## LOAD TRAIN (AND VAL) DATA ##
    entry = _DATASET_REGISTRY[dataset_type]
    train_val_data = entry["dataset_class"](
        **entry["kwargs"](train_dir, transform_train)
    )

    # Split training data while preserving class proportions
    train_data, train_indices, val_indices = (
        data.utils.dataset_stratified_shuffle_split(train_val_data)
    )

    ## LOAD VAL DATA ##
    val_data = Subset(
        entry["dataset_class"](**entry["kwargs"](train_dir, transform_val)), val_indices
    )

    ## LOAD TEST DATA ##
    test_data = datasets.ImageFolder(test_dir, transform=transform_test)

    # Calculate class weights
    class_weights = data.utils.dataset_calculate_class_weights(
        train_val_data, train_indices
    )

    # Assign weights to each sample
    sample_weights = [class_weights[train_val_data.targets[i]] for i in train_indices]

    sampler = WeightedRandomSampler(
        weights=sample_weights, num_samples=len(sample_weights), replacement=True
    )

    def collate_fn(batch):
        first = batch[0]
        if isinstance(first, tuple) and len(first) == 2:
            images, labels = zip(*batch)
            return torch.stack(images), torch.tensor(labels)
        elif isinstance(first, tuple) and len(first) == 3:
            images, labels, extra = zip(*batch)
            return torch.stack(images), torch.tensor(labels), extra
        else:
            return default_collate(batch)

    # DataLoader factory
    def make_loader(dataset, *, sampler=None, shuffle=None):
        return DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=collate_fn,
        )

    train_loader = make_loader(train_data, sampler=sampler, shuffle=None)
    val_loader = make_loader(val_data, sampler=None, shuffle=False)
    test_loader = make_loader(test_data, sampler=None, shuffle=False)

    return train_loader, val_loader, test_loader


def create_class_dataloader(
    data_dir: str,
    *,
    target_class: int,
    batch_size: int,
    num_workers: int,
    transform: transforms.Compose,
    dataset_type: str = "standard",
    shuffle: bool = False,
) -> DataLoader:
    """
    Creates a DataLoader for samples belonging to a specific target class from the dataset.

    This function loads a dataset from the specified directory, filters the samples to
    include only those belonging to the given target class, and returns a PyTorch DataLoader
    over this filtered subset.

    Args:
        data_dir (str): Path to the dataset directory.
        target_class (int): The target class index to filter samples by.
        batch_size (int): Number of samples per batch to load.
        num_workers (int): Number of subprocesses to use for data loading.
        transform (transforms.Compose): Transformations to apply to the dataset.
        dataset_type (str, optional): Type of dataset to load. Must be a key in `_DATASET_REGISTRY`.
            Defaults to "standard".
        shuffle (bool, optional): Whether to shuffle the data each epoch. Defaults to False.

    Returns:
        DataLoader: A DataLoader containing only the samples of the specified target class.

    Raises:
        ValueError: If `dataset_type` is not found in `_DATASET_REGISTRY`.
    """
    if dataset_type not in _DATASET_REGISTRY:
        valid = ", ".join(_DATASET_REGISTRY.keys())
        raise ValueError(
            f"Unsupported dataset type '{dataset_type}'. Valid types are: {valid}"
        )

    entry = _DATASET_REGISTRY[dataset_type]
    ds = entry["dataset_class"](**entry["kwargs"](data_dir, transform))

    class_indices = [i for i, label in enumerate(ds.targets) if label == target_class]

    class_data = Subset(ds, class_indices)
    return DataLoader(
        class_data,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
    )
