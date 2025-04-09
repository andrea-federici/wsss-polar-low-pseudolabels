from torchvision import transforms
from omegaconf import DictConfig


def get_transform(cfg_transforms: DictConfig, stage: str) -> transforms.Compose:
    image_size = (cfg_transforms.image_width, cfg_transforms.image_height)

    mean = cfg_transforms.normalization.mean
    std = cfg_transforms.normalization.std

    if stage == "train":
        transform_list = [
            transforms.RandomAffine(
                degrees=cfg_transforms.augmentation.degrees,
                translate=tuple(cfg_transforms.augmentation.translate_frac),
                scale=tuple(cfg_transforms.augmentation.scale),
                fill=0,
            ),
            transforms.Resize(image_size),
        ]

        if cfg_transforms.augmentation.horizontal_flip:
            transform_list.append(transforms.RandomHorizontalFlip())

        if cfg_transforms.augmentation.vertical_flip:
            transform_list.append(transforms.RandomVerticalFlip())

        transform_list.extend(
            [transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)]
        )

        transform = transforms.Compose(transform_list)
    elif stage in ["val", "test"]:
        transform = transforms.Compose(
            [
                transforms.Resize(image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )
    else:
        raise ValueError(
            f"Invalid stage: {stage}. Expected one of 'train', " f"'val', 'test'."
        )

    return transform
