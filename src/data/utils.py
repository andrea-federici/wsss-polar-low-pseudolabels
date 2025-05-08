import os

import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
import torch
from torch.utils.data import Dataset, Subset, DataLoader
from torchvision import transforms
from collections import Counter
from PIL import Image


def dataset_stratified_shuffle_split(
    dataset: Dataset, test_size: float = 0.2, random_state: int = None
) -> tuple[Subset, list[int], list[int]]:
    """
    Perform a stratified shuffle split on a dataset, splitting it into
    training and test sets while preserving the class distribution.

    Args:
        dataset (Dataset): The dataset to be split. It must have a `targets`
                            attribute containing the labels for
                            stratification.
        test_size (float, optional): The proportion of the dataset to include
                                        in the test set. Default is 0.2 (20%
                                        of the dataset is used for testing).
        random_state (int, optional): The seed used by the random number
                                        generator to ensure reproducibility.
                                        Default is None.

    Returns:
        tuple: A tuple containing the following elements:
            - `train_data` (Subset): A `Subset` of the dataset containing the
                                        training data.
            - `train_indices` (list): A list of indices for the training data.
            - `test_indices` (list): A list of indices for the test data.
    """
    split = StratifiedShuffleSplit(
        n_splits=1, test_size=test_size, random_state=random_state
    )

    # Get the labels for stratification
    labels = dataset.targets

    train_data = None
    train_indices, test_indices = [], []

    # Perform the split, retrieving indices for train and test sets
    for train_indices, test_indices in split.split(np.zeros(len(labels)), labels):
        train_indices = train_indices.tolist()
        test_indices = test_indices.tolist()

        # Create a Subset for the training data using the indices
        train_data = Subset(dataset, train_indices)

    return train_data, train_indices, test_indices


def dataset_calculate_class_weights(dataset: Dataset, train_indices: list[int]) -> dict:
    """
    Calculate the class weights for a dataset based on the distribution of
    labels in the training set.

    The class weights are calculated as the inverse of the class frequency in
    the training set, and then normalized such that the sum of all weights
    equals 1.

    Args:
        dataset (Dataset): The dataset that contains the target labels
                            (`targets` attribute). It is expected that
                            `dataset.targets` contains the class labels.
        train_indices (list[int]): A list of indices representing the subset
                                    of the dataset used for training. These
                                    indices should correspond to entries in
                                    `dataset.targets`.

    Returns:
        dict: A dictionary where the keys are the class labels, and the values
                are the normalized class weights. The class weights are
                inversely proportional to the frequency of each class in the
                training set, and their sum is 1.
    """
    # Count the occurrences of each class in the training set based on
    # train_indices
    train_counts = Counter([dataset.targets[i] for i in train_indices])

    # Calculate the class weights (inverse of count for each class)
    class_weights_not_normalized = {
        cls: 1.0 / count for cls, count in train_counts.items()
    }

    # Normalize the class weights so that their sum is equal to 1
    total_weights = sum(class_weights_not_normalized.values())
    class_weights = {
        cls: weight / total_weights
        for cls, weight in class_weights_not_normalized.items()
    }

    return class_weights


# TODO: check that this works properly and format it well
def compute_dataset_mean_std(dataset, batch_size=64):
    """
    Compute the dataset mean and standard deviation while ignoring black pixels.

    Args:
        dataset (torch.utils.data.Dataset): The dataset of images.
        batch_size (int): The batch size for processing images.

    Returns:
        tuple: (mean, std) of the dataset, ignoring black pixels.
    """
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    sum_pixels = torch.zeros(3)  # Sum of pixel values per channel
    sum_squares = torch.zeros(3)  # Sum of squared pixel values per channel
    valid_pixel_count = torch.zeros(3)  # Count of non-black pixels per channel

    for images, _ in loader:
        images = images.view(
            images.shape[0], images.shape[1], -1
        )  # Flatten (B, C, H*W)

        # Identify non-black pixels: sum along channels > 0 (not all zero)
        mask = images.sum(dim=1) > 0  # Shape (B, H*W), True if not black

        for c in range(3):  # Loop over channels
            valid_pixels = images[:, c][mask]  # Select only valid (non-black) pixels

            if valid_pixels.numel() > 0:  # If there are valid pixels
                sum_pixels[c] += valid_pixels.sum()
                sum_squares[c] += (valid_pixels**2).sum()
                valid_pixel_count[c] += valid_pixels.numel()

    mean = sum_pixels / valid_pixel_count
    std = torch.sqrt(sum_squares / valid_pixel_count - mean**2)

    return mean, std


def dataset_print_stats(dataset: Dataset, dataset_name: str = None) -> None:
    """
    Print statistics about a dataset.

    This function assumes that the dataset contains a `targets` attribute,
    which holds the class labels, and that the class names 'pos' and 'neg'
    exist in the dataset.

    The statistics printed include:
    - Total number of samples
    - Number of positive ('pos') and negative ('neg') samples
    - Class distribution ratio (positive to negative)
    - Percentages of positive and negative samples in the dataset

    Args:
        dataset (Dataset): The dataset for which to compute and print
                            statistics. It should have a `targets` attribute
                            containing the class labels.
        dataset_name (str, optional): A custom name for the dataset to be
                                        displayed in the output. If not
                                        provided, the statistics will be
                                        printed without a dataset name.

    Returns:
        None: This function prints the dataset statistics but does not return
                any value.
    """
    # Retrieve targets, class information, and class-to-index mapping
    targets, classes, class_to_idx = dataset_get_targets_and_classes(dataset)

    # Count the occurrences of each class in the dataset
    dataset_counts = Counter(targets)

    # Get class indices for 'pos' and 'neg'
    pos_idx = class_to_idx["pos"]
    neg_idx = class_to_idx["neg"]

    # Get the number of positive and negative samples
    pos_count = dataset_counts[pos_idx]
    neg_count = dataset_counts[neg_idx]
    total_count = len(targets)

    # Calculate the class ratio and percentages
    class_ratio = pos_count / neg_count if neg_count > 0 else float("inf")
    pos_percentage = (pos_count / total_count) * 100
    neg_percentage = (neg_count / total_count) * 100

    # Print dataset statistics
    if dataset_name:
        print(f"'{dataset_name}' dataset:")
    print(
        f"\tNumber of samples: {total_count} (neg: {neg_count}, pos: " f"{pos_count})"
    )
    print(f"\tNumber of classes: {len(classes)}")
    print(f"\tClass names: {classes}")
    print(f"\tClass distribution ratio (pos:neg): {class_ratio:.2f}")
    print(
        f"\tClass percentages: {pos_percentage:.2f}% pos, " f"{neg_percentage:.2f}% neg"
    )
    print()


def dataset_get_targets_and_classes(
    dataset: Dataset,
) -> tuple[list[int], list[str], dict[str, int]]:
    """
    Retrieve the targets, classes, and class-to-index mapping from a dataset.

    This function extracts the target labels (`targets`),
    the class names (`classes`), and the mapping between class names and their
    corresponding indices (`class_to_idx`).

    If the dataset is a subset, it will retrieve these properties from the
    original dataset (the parent dataset from which the subset is derived).

    Args:
        dataset (Dataset): A PyTorch Dataset object. It could be a standard
                            dataset or a subset of a larger dataset.

    Returns:
        tuple: A tuple containing:
            - targets (list[int]): A list of class labels (integer values) for
                                    the samples in the dataset.
            - classes (list[str]): A list of class names (strings)
                                    corresponding to the labels.
            - class_to_idx (dict[str, int]): A dictionary mapping class names
                                                to their corresponding indices.
    """
    # Check if the dataset is a subset (subset of a larger dataset)
    if isinstance(dataset, Subset):
        # Retrieve the original dataset and subset indices
        original_dataset = dataset.dataset
        subset_indices = dataset.indices

        # Extract targets, classes, and class_to_idx from the original dataset
        targets = [original_dataset.targets[i] for i in subset_indices]
        classes = original_dataset.classes
        class_to_idx = dataset.dataset.class_to_idx
    else:
        # For a full dataset, directly use its attributes
        targets = dataset.targets
        classes = dataset.classes
        class_to_idx = dataset.class_to_idx

    return targets, classes, class_to_idx


def pick_random_image_path(train_dir: str, class_name: str = None) -> str:
    """
    Select a random image path from the specified training directory and class.

    This function randomly selects an image from a specified class directory
    within the training directory. If a class name is not provided, it
    randomly chooses between the 'pos' and 'neg' class directories.

    Args:
        train_dir (str): The path to the main training directory that contains
                         subdirectories for each class (e.g., 'pos', 'neg').
        class_name (str, optional): The name of the class to select the image
                                    from. If not provided, a random class
                                    ('pos' or 'neg') is chosen.

    Returns:
        str: The full file path of a randomly selected image within the class
                directory.

    Raises:
        ValueError: If the specified `train_dir` or `class_name` is invalid or
                    does not exist.
    """
    if class_name is None:
        class_name = np.random.choice(["pos", "neg"])

    class_dir = os.path.join(train_dir, class_name)

    # Check if the class directory exists
    if not os.path.exists(class_dir):
        raise ValueError(
            f"Class directory '{class_dir}' does not exist in the"
            " provided training directory."
        )

    # Get all image files in the class directory
    image_files = os.listdir(class_dir)

    # Check if there are any images in the directory
    if not image_files:
        raise ValueError(f"No images found in the class directory " f"'{class_dir}'.")

    # Select and return the random image path
    return os.path.join(class_dir, np.random.choice(image_files))


def load_and_transform_image(
    image_path: str, transform: transforms.Compose, device="cpu"
) -> torch.Tensor:
    """
    Loads an image from the specified file path, applies a series of
    transformations, and returns the transformed image as a tensor with an
    added batch dimension, moved to the specified device.

    Args:
        image_path (str): The path to the image file to be loaded.
        transform (transforms.Compose): A torchvision.transforms.Compose
                                        object that contains a sequence of
                                        transformations to be applied to the
                                        image.
        device (str, optional): The device to which the image tensor should be
                                moved. Defaults to 'cpu'.

    Returns:
        torch.Tensor: The transformed image as a tensor, with a batch
                        dimension added and moved to the specified device.

    Raises:
        FileNotFoundError: If the image file cannot be loaded from the
                            provided path.
        TypeError: If the transformation does not return a torch.Tensor.
    """
    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as e:
        raise FileNotFoundError(f"Error loading image from path {image_path}: {e}")

    transformed_image = transform(image)

    # Ensure that the transformed image is a Tensor
    if not isinstance(transformed_image, torch.Tensor):
        raise TypeError(
            f"Expected a torch.Tensor but got " f"{type(transformed_image)}"
        )

    # Add batch dimension and move to device
    return transformed_image.unsqueeze(0).to(device)
