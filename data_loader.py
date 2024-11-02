
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from sklearn.model_selection import StratifiedShuffleSplit
from collections import Counter
import numpy as np


def get_transforms(resized_image_res):
    # Transformations for the training set
    data_augmentation = transforms.Compose([
        transforms.RandomAffine(degrees=45, translate=(0.35, 0.35), scale=(0.8, 1.2), fill=0),  # Random Rotation, Translation, and Zoom
        transforms.RandomHorizontalFlip(),  # Random Horizontal Flip
        transforms.RandomVerticalFlip(),  # Random Vertical Flip
        transforms.Resize(resized_image_res),  # Resize
        transforms.ToTensor(),  # Convert to tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Rescaling / Normalizing
    ])

    # Transformations for the validation and test sets
    data_prep = transforms.Compose([
        transforms.Resize(resized_image_res),
        transforms.ToTensor(),  # Convert to tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Rescaling / Normalizing
    ])

    return data_augmentation, data_prep


def create_data_loaders(train_dir, test_dir, resized_image_res, batch_size, num_workers, device='cpu', verbose=False):
    data_augmentation, data_prep = get_transforms(resized_image_res)

    # Load the data
    train_val_data = datasets.ImageFolder(train_dir, transform=data_augmentation)
    test_data = datasets.ImageFolder(test_dir, transform=data_prep)

    if verbose:
        print_dataset_stats(train_val_data, "Train val")
        print_dataset_stats(test_data, "Test")
    
    # Split the data into training and validation sets while preserving class proportions
    train_data, train_indices, val_indices = stratified_shuffle_split(train_val_data)
    
    # For the validation data images are reloaded with the data_prep transformation
    val_data = Subset(datasets.ImageFolder(train_dir, transform=data_prep), val_indices)

    if verbose:
        print_dataset_stats(train_data, "Train")
        print_dataset_stats(val_data, "Val")

    if verbose:
        print('Transforms applied to the training data:')
        print(train_data.dataset.transform)
        print('Transforms applied to the validation data:')
        print(val_data.dataset.transform)

    # Calculate class weights
    class_weights = calculate_class_weights(train_val_data, train_indices)

    # Assign the corresponding weight to each sample in the train dataset
    sample_weights = [class_weights[train_val_data.targets[i]] for i in train_indices]

    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

    # Create DataLoader for training data using the sampler
    train_loader = DataLoader(train_data, batch_size=batch_size, sampler=sampler, num_workers=num_workers)

    # Create DataLoader for validation and test data without any sampler
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader


def stratified_shuffle_split(dataset, test_size=0.2, random_state=42):
    """Split the dataset into training and test sets while preserving class proportions."""
    
    split = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    labels = dataset.targets
    for train_indices, val_indices in split.split(np.zeros(len(labels)), labels):
        train_data = Subset(dataset, train_indices)

    return train_data, train_indices, val_indices


def calculate_class_weights(dataset, train_indices):
    """Calculate class weights based on the dataset and the indices used for training."""
    train_counts = Counter([dataset.targets[i] for i in train_indices])

    # Calculate class weights
    class_weights_not_normalized = { cls: 1.0 / count for cls, count in train_counts.items() }

    # Normalize the weights
    total_weights = sum(class_weights_not_normalized.values())
    class_weights = { cls: weight / total_weights for cls, weight in class_weights_not_normalized.items() }
    
    return class_weights


def print_dataset_stats(dataset, dataset_name=""):
    """Print statistics about the dataset."""
    # Retrive targes, class information, and class-to-index mapping
    targets, classes, class_to_idx = get_targets_and_classes(dataset)

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


def get_targets_and_classes(dataset):
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
