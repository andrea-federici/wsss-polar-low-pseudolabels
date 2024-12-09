
from torchvision import datasets
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler

from data_utils import (
    dataset_calculate_class_weights,
    dataset_print_stats,
    dataset_stratified_shuffle_split,
)
from train_config import (
    transform_aug,
    transform_prep,
)


def create_data_loaders(train_dir, test_dir, batch_size, num_workers, verbose=False):
    # Load the data
    train_val_data = datasets.ImageFolder(train_dir, transform=transform_aug)
    test_data = datasets.ImageFolder(test_dir, transform=transform_prep)

    if verbose:
        dataset_print_stats(train_val_data, "Train val")
        dataset_print_stats(test_data, "Test")
    
    # Split the data into training and validation sets while preserving class proportions
    train_data, train_indices, val_indices = dataset_stratified_shuffle_split(train_val_data)
    
    # For the validation data images are reloaded with the data_prep transformation
    val_data = Subset(datasets.ImageFolder(train_dir, transform=transform_prep), val_indices)

    if verbose:
        dataset_print_stats(train_data, "Train")
        dataset_print_stats(val_data, "Val")

    if verbose:
        print('Transforms applied to the training data:')
        print(train_data.dataset.transform)
        print('Transforms applied to the validation data:')
        print(val_data.dataset.transform)

    # Calculate class weights
    class_weights = dataset_calculate_class_weights(train_val_data, train_indices)

    # Assign the corresponding weight to each sample in the train dataset
    sample_weights = [class_weights[train_val_data.targets[i]] for i in train_indices]

    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

    # Create DataLoader for training data using the sampler
    train_loader = DataLoader(train_data, batch_size=batch_size, sampler=sampler, num_workers=num_workers)

    # Create DataLoader for validation and test data without any sampler
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader

