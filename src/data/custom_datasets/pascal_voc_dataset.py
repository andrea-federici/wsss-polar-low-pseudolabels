import os
from collections import defaultdict
from typing import List

from PIL import Image
import torch
from torch.utils.data import Dataset


PASCAL_VOC_CLASSES: List[str] = [
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor",
]


class PascalVOCDataset(Dataset):
    """Pascal VOC multi-label dataset.

    Expected directory structure::

        root/
            JPEGImages/
            ImageSets/
                Main/
                    <class>_{train,val}.txt

    Each ``<class>_<split>.txt`` file contains lines of the form
    ``<image_id> <label>`` where ``label`` is ``1`` if the image
    contains the class, ``-1`` otherwise. A multi-hot label vector is
    constructed for each image across all classes.
    """

    def __init__(self, root: str, *, split: str = "train", transform=None):
        self.root = root
        self.split = split
        self.transform = transform

        self.classes = PASCAL_VOC_CLASSES
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
        num_classes = len(self.classes)

        self.image_dir = os.path.join(root, "JPEGImages")
        sets_dir = os.path.join(root, "ImageSets", "Main")

        # Build label vectors for each image
        labels_dict: defaultdict[str, List[int]] = defaultdict(
            lambda: [0] * num_classes
        )
        for cls in self.classes:
            file_path = os.path.join(sets_dir, f"{cls}_{split}.txt")
            if not os.path.exists(file_path):
                continue
            with open(file_path) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) != 2:
                        continue
                    image_id, flag = parts
                    # Ensure entry exists
                    vec = labels_dict[image_id]
                    if flag == "1":
                        vec[self.class_to_idx[cls]] = 1

        self.filenames = sorted(labels_dict.keys())
        self.targets = [
            torch.tensor(labels_dict[f], dtype=torch.float32) for f in self.filenames
        ]

    def __len__(self) -> int:
        return len(self.filenames)

    def __getitem__(self, idx: int):
        image_id = self.filenames[idx]
        image_path = os.path.join(self.image_dir, f"{image_id}.jpg")
        image = Image.open(image_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        label_vector = self.targets[idx]
        filename = f"{image_id}.jpg"
        return image, label_vector, filename
