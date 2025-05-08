from dataclasses import dataclass
from typing import Optional, Sequence, Tuple, Union

from omegaconf import DictConfig
from torchvision import transforms


@dataclass
class AugConfig:
    resize_width: int
    resize_height: int
    mean: Optional[Sequence[float]] = None
    std: Optional[Sequence[float]] = None
    translate_frac: Optional[float] = None
    degrees: Optional[float] = None
    scale: Optional[Tuple[float, float]] = None
    horizontal_flip: bool = False
    vertical_flip: bool = False

    def __post_init__(self):
        # Validate image dimensions
        if not isinstance(self.resize_width, int) or self.resize_width <= 0:
            raise ValueError(
                f"resize_width must be a positive integer, got {self.resize_width}"
            )
        if not isinstance(self.resize_height, int) or self.resize_height <= 0:
            raise ValueError(
                f"resize_height must be a positive integer, got {self.resize_height}"
            )

        # Validate normalization parameters
        if (self.mean is not None) ^ (self.std is not None):
            raise ValueError(
                "Both mean and std must be provided together or left as None."
            )
        if self.mean is not None and self.std is not None:
            if len(self.mean) != len(self.std):
                raise ValueError(
                    f"Mean and std must have the same length, got {len(self.mean)} "
                    f"and {len(self.std)}"
                )
            for m in self.mean:
                if not isinstance(m, (int, float)):
                    raise ValueError(f"Mean values must be int or float, got {type(m)}")
            for s in self.std:
                if not isinstance(s, (int, float)):
                    raise ValueError(f"Std values must be int or float, got {type(s)}")

        # Validate translation fraction
        if self.translate_frac is not None:
            if not isinstance(self.translate_frac, float) or not (
                0 <= self.translate_frac <= 1
            ):
                raise ValueError(
                    f"translate_frac must be a float between 0 and 1, got "
                    f"{self.translate_frac}"
                )

        # Validate degrees
        if self.degrees is not None:
            if not isinstance(self.degrees, float) or not (0 <= self.degrees <= 360):
                raise ValueError(
                    f"degrees must be a float between 0 and 360, got {self.degrees}"
                )

        # Validate scale
        if self.scale is not None:
            if not isinstance(self.scale, tuple) or len(self.scale) != 2:
                raise ValueError(
                    f"scale must be a tuple of two floats, got {self.scale}"
                )
            if not (0 < self.scale[0] <= self.scale[1]):
                raise ValueError(
                    f"scale must be a tuple of two floats where the first is less than "
                    f"or equal to the second, got {self.scale}"
                )

        # Validate horizontal_flip
        if not isinstance(self.horizontal_flip, bool):
            raise ValueError(
                f"horizontal_flip must be a boolean, got {type(self.horizontal_flip)}"
            )

        # Validate vertical_flip
        if not isinstance(self.vertical_flip, bool):
            raise ValueError(
                f"vertical_flip must be a boolean, got {type(self.vertical_flip)}"
            )


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
