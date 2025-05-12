from dataclasses import dataclass
from typing import Optional, Sequence, Tuple, Union

import torch
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
    fill_color: Union[int, float, Sequence[float]] = 0
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

        # Validate fill color
        if isinstance(self.fill_color, (int, float)):
            pass
        elif isinstance(self.fill_color, Sequence):
            if not all(isinstance(x, (int, float)) for x in self.fill_color):
                raise ValueError(
                    f"fill_color sequence must contain only ints/floats, got "
                    f"{self.fill_color}"
                )
        else:
            raise ValueError(
                f"fill_color must be int, float, or sequence thereof, got "
                f"{type(self.fill_color)}"
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


def to_aug_config(cfg: DictConfig) -> AugConfig:
    if not isinstance(cfg, DictConfig):
        raise ValueError(
            f"cfg_transforms must be an instance of DictConfig, got {type(cfg)}"
        )

    resize_width = cfg.get("image_width", None)
    resize_height = cfg.get("image_height", None)

    normalization = cfg.get("normalization", None)
    if normalization is not None:
        mean = normalization.get("mean", None)
        std = normalization.get("std", None)
    else:
        mean = None
        std = None

    augmentation = cfg.get("augmentation", None)
    if augmentation is not None:
        translate_frac = augmentation.get("translate_frac", None)
        degrees = augmentation.get("degrees", None)
        raw_scale = augmentation.get("scale", None)
        scale = tuple(raw_scale) if raw_scale is not None else None
        fill_color = augmentation.get("fill_color", 0)
        horizontal_flip = augmentation.get("horizontal_flip", False)
        vertical_flip = augmentation.get("vertical_flip", False)
    else:
        translate_frac = None
        degrees = None
        scale = None
        fill_color = 0
        horizontal_flip = False
        vertical_flip = False

    return AugConfig(
        resize_width=resize_width,
        resize_height=resize_height,
        mean=mean,
        std=std,
        translate_frac=translate_frac,
        degrees=degrees,
        scale=scale,
        fill_color=fill_color,
        horizontal_flip=horizontal_flip,
        vertical_flip=vertical_flip,
    )


def to_compose(aug_config: AugConfig, stage: str) -> transforms.Compose:
    assert stage in [
        "train",
        "val",
        "test",
    ], f"stage should be either 'train', 'val', or 'test', got '{stage}'"

    transform_list = []

    if stage == "train":
        if aug_config.translate_frac is not None:
            translate = tuple(aug_config.translate_frac)
        else:
            translate = None

        transform_list.append(
            transforms.RandomAffine(
                degrees=aug_config.degrees or 0.0,
                translate=translate,
                scale=aug_config.scale or (1.0, 1.0),
                fill=aug_config.fill_color,
            )
        )

        if aug_config.horizontal_flip:
            transform_list.append(transforms.RandomHorizontalFlip())

        if aug_config.vertical_flip:
            transform_list.append(transforms.RandomVerticalFlip())

    transform_list.extend(
        [
            transforms.Resize((aug_config.resize_width, aug_config.resize_height)),
            transforms.ToTensor(),
        ]
    )

    if aug_config.mean is not None and aug_config.std is not None:
        transform_list.append(
            transforms.Normalize(mean=aug_config.mean, std=aug_config.std),
        )

    return transforms.Compose(transform_list)
