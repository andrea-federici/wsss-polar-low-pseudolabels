import torch
from torchvision import datasets
from torch.utils.data import (
    DataLoader, Subset, WeightedRandomSampler
)

from data.data_utils import (
    dataset_calculate_class_weights,
    dataset_stratified_shuffle_split,
)
from train_config import transform_prep
from custom_datasets import MaxTranslationsDataset, ImageFilenameDataset

import warnings


def create_data_loaders(
    train_dir,
    test_dir,
    batch_size,
    num_workers,
    transform_train,
    transform_val = transform_prep,
    dataset_type: str = 'default',
    **kwargs
):
    assert dataset_type in (
        'default', 'max_translations', 'adversarial_erase'
    ), (
        f"Invalid dataset type: {dataset_type}. Expected one of 'default', "
        "'max_translations', 'adversarial_erase'."
    )

    if dataset_type == 'default':
        train_val_data = datasets.ImageFolder(
            train_dir, transform=transform_train
        )
    elif dataset_type == 'max_translations':
        max_translations = kwargs.get('max_translations', None)

        if max_translations is None:
            raise ValueError(
                "max_translations must be provided when dataset_type is "
                "'max_translations'."
            )
        
        if transform_train != transform_prep:
            warnings.warn(
                (
                    "For the 'max_translations' dataset, 'transform_train' "
                    "should be set to 'transform_prep', since the "
                    "transformations are applied in the training loop."
                ),
                UserWarning
            )

        train_val_data = MaxTranslationsDataset(
            train_dir, 
            max_translations=max_translations, 
            transform=transform_train
        )
    elif dataset_type == 'adversarial_erase':
        if transform_train != transform_prep:
            warnings.warn(
                (
                    "For the 'adversarial_erase' dataset, 'transform_train' "
                    "should be set to 'transform_prep', since the "
                    "transformations are applied in the training loop."
                ),
                UserWarning
            )

        train_val_data = ImageFilenameDataset(
            train_dir, transform=transform_train
        )
    
    test_data = datasets.ImageFolder(test_dir, transform=transform_prep)

    # Split training data while preserving class proportions
    train_data, train_indices, val_indices = dataset_stratified_shuffle_split(
        train_val_data
    )

    # Validation data uses a different transform
    if dataset_type == 'default':
        val_data = Subset(
            datasets.ImageFolder(
                train_dir, transform=transform_val
            ), 
            val_indices
        )
    elif dataset_type == 'max_translations':
        val_data = Subset(
            MaxTranslationsDataset(
                train_dir, 
                max_translations=max_translations, 
                transform=transform_val
            ), 
            val_indices
        )
    elif dataset_type == 'adversarial_erase':
        val_data = Subset(
            ImageFilenameDataset(
                train_dir, 
                transform=transform_val
            ), 
            val_indices
        )
    
    # Calculate class weights
    class_weights = dataset_calculate_class_weights(
        train_val_data, train_indices
    )

    # Assign weights to each sample
    sample_weights = [
        class_weights[train_val_data.targets[i]] for i in train_indices
    ]

    sampler = WeightedRandomSampler(
        weights=sample_weights, 
        num_samples=len(sample_weights), 
        replacement=True
    )

    def collate_fn(batch):
        if dataset_type == 'default':
            images, labels = zip(*batch)
            return torch.stack(images), torch.tensor(labels)
        elif dataset_type == "max_translations":
            images, labels, max_translations = zip(*batch)
            return torch.stack(images), torch.tensor(labels), max_translations
        elif dataset_type == "adversarial_erase":
            images, labels, filenames = zip(*batch)
            return torch.stack(images), torch.tensor(labels), filenames

    # Create DataLoaders
    train_loader = DataLoader(
        train_data, 
        batch_size=batch_size, 
        sampler=sampler, 
        num_workers=num_workers, 
        collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_data, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers, 
        collate_fn=collate_fn
    )
    test_loader = DataLoader(
        test_data, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers
    )

    return train_loader, val_loader, test_loader