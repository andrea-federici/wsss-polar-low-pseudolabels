import os

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler

from src import data


def create_data_loaders(
    data_dir: str,
    batch_size: int,
    num_workers: int,
    transform_train: transforms.Compose,
    transform_val: transforms.Compose,
    transform_test: transforms.Compose = None,
    dataset_type: str = "standard",
    only_positives: bool = False,
    positive_label: int = 1,
    **kwargs,
):
    assert dataset_type in ("standard", "max_translations", "adversarial_erasing"), (
        f"Invalid dataset type: {dataset_type}. Expected one of 'standard', "
        "'max_translations', 'adversarial_erasing'."
    )

    train_dir = os.path.join(data_dir, "train")
    test_dir = os.path.join(data_dir, "test")

    if transform_test is None:
        transform_test = transform_val

    if dataset_type == "standard":
        train_val_data = datasets.ImageFolder(train_dir, transform=transform_train)
    elif dataset_type == "max_translations":
        max_translations = kwargs.get("max_translations", None)

        if max_translations is None:
            raise ValueError(
                "max_translations must be provided when dataset_type is "
                "'max_translations'."
            )

        train_val_data = data.custom_datasets.MaxTranslationsDataset(
            train_dir, max_translations=max_translations, transform=transform_train
        )
    elif dataset_type == "adversarial_erasing":
        train_val_data = data.custom_datasets.ImageFilenameDataset(
            train_dir, transform=transform_train
        )

    test_data = datasets.ImageFolder(test_dir, transform=transform_test)

    # Split training data while preserving class proportions
    train_data, train_indices, val_indices = (
        data.data_utils.dataset_stratified_shuffle_split(train_val_data)
    )

    # If we only want positives, filter out any index whose label is not positive
    if only_positives:
        train_indices = [
            i for i in train_indices if train_val_data.targets[i] == positive_label
        ]
        val_indices = [
            i for i in val_indices if train_val_data.targets[i] == positive_label
        ]

        train_data = Subset(train_val_data, train_indices)

    # Validation data uses a different transform
    if dataset_type == "standard":
        val_data = Subset(
            datasets.ImageFolder(train_dir, transform=transform_val), val_indices
        )
    elif dataset_type == "max_translations":
        val_data = Subset(
            data.custom_datasets.MaxTranslationsDataset(
                train_dir, max_translations=max_translations, transform=transform_val
            ),
            val_indices,
        )
    elif dataset_type == "adversarial_erasing":
        val_data = Subset(
            data.custom_datasets.ImageFilenameDataset(
                train_dir, transform=transform_val
            ),
            val_indices,
        )

    if only_positives:
        sampler = None
        shuffle = True
    else:
        # Calculate class weights
        class_weights = data.data_utils.dataset_calculate_class_weights(
            train_val_data, train_indices
        )

        # Assign weights to each sample
        sample_weights = [
            class_weights[train_val_data.targets[i]] for i in train_indices
        ]

        sampler = WeightedRandomSampler(
            weights=sample_weights, num_samples=len(sample_weights), replacement=True
        )

        shuffle = False

    def collate_fn(batch):
        if dataset_type == "standard":
            images, labels = zip(*batch)
            return torch.stack(images), torch.tensor(labels)
        elif dataset_type == "max_translations":
            images, labels, max_translations = zip(*batch)
            return torch.stack(images), torch.tensor(labels), max_translations
        elif dataset_type == "adversarial_erasing":
            images, labels, filenames = zip(*batch)
            return torch.stack(images), torch.tensor(labels), filenames

    # Create DataLoaders
    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )
    test_loader = DataLoader(
        test_data, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    return train_loader, val_loader, test_loader
