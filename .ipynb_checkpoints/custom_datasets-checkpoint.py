import os

from torchvision import datasets
from torch.utils.data import Dataset


class MaxTranslationsDataset(Dataset):
    def __init__(self, data_dir, max_translations, transform=None):
        self.dataset = datasets.ImageFolder(data_dir, transform=transform)
        self.max_translations = max_translations
        self.transform = transform
        self.targets = self.dataset.targets

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        image_path, _ = self.dataset.samples[idx]

        image_filename = os.path.basename(image_path)

        # Return None if max_translations is not available for the image
        img_max_translations = self.max_translations.get(image_filename, None)

        return image, label, img_max_translations


class ImageFilenameDataset(Dataset):
    def __init__(self, data_dir: str, transform=None):
        self.dataset = datasets.ImageFolder(data_dir, transform=transform)
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