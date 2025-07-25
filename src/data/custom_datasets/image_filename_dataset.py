import os

from torch.utils.data import Dataset
from torchvision import datasets


class ImageFilenameDataset(Dataset):
    def __init__(self, root: str, transform=None):
        self.dataset = datasets.ImageFolder(root, transform=transform)
        self.targets = self.dataset.targets

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]

        # dataset.samples is a list containing tuples of file paths and labels.
        image_path, _ = self.dataset.samples[idx]

        # Extract only the filename from the full path
        image_filename = os.path.basename(image_path)

        return image, label, image_filename
