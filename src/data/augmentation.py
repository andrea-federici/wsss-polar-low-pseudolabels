from dataclasses import asdict, dataclass
from pprint import pformat
from typing import Optional, Sequence, Tuple, Union

from omegaconf import DictConfig
from torchvision import transforms


@dataclass
class RandomErasingConfig:
    p: float
    scale: Tuple[float, float]

    def __post_init__(self):
        if not isinstance(self.p, (int, float)) or not (0.0 <= self.p <= 1.0):
            raise ValueError(f"p must be a float between 0 and 1, got {self.p}")
        if not isinstance(self.scale, tuple) or len(self.scale) != 2:
            raise ValueError(f"scale must be a tuple of two floats, got {self.scale}")
        if not all(isinstance(s, (int, float)) for s in self.scale):
            raise ValueError("scale values must be floats")
        if not (0.0 <= self.scale[0] <= self.scale[1] <= 1.0):
            raise ValueError(
                "scale values must be between 0 and 1 and in ascending order"
            )

    def __str__(self) -> str:
        return f"RandomErasingConfig(p={self.p}, scale={self.scale})"

    __repr__ = __str__


@dataclass
class AugConfig:
    resize_width: int
    resize_height: int
    mean: Optional[Sequence[float]] = None
    std: Optional[Sequence[float]] = None
    translate_frac: Optional[float] = None
    degrees: Optional[Union[int, float]] = None
    scale: Optional[Tuple[float, float]] = None
    fill_color: Union[int, float, Sequence[float]] = 0
    horizontal_flip: bool = False
    vertical_flip: bool = False
    brightness: Optional[float] = None
    contrast: Optional[float] = None
    saturation: Optional[float] = None
    hue: Optional[float] = None
    random_erasing: Optional[RandomErasingConfig] = None

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
            if not isinstance(self.degrees, (int, float)) or not (
                0 <= self.degrees <= 360
            ):
                raise ValueError(
                    f"degrees must be a number between 0 and 360, got {self.degrees}"
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

        # Validate color jitter parameters
        for name in ["brightness", "contrast", "saturation"]:
            value = getattr(self, name)
            if value is not None:
                if not isinstance(value, (int, float)) or value < 0:
                    raise ValueError(
                        f"{name} must be a non-negative number, got {value}"
                    )

        if self.hue is not None:
            if not isinstance(self.hue, (int, float)) or not (0 <= self.hue <= 0.5):
                raise ValueError(
                    f"hue must be a number between 0 and 0.5, got {self.hue}"
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

        # Validate random erasing configuration
        if self.random_erasing is not None and not isinstance(
            self.random_erasing, RandomErasingConfig
        ):
            raise ValueError(
                "random_erasing must be an instance of RandomErasingConfig or None"
            )

    def is_valid_for_adversarial_erasing(self) -> bool:
        """
        Check if the configuration is valid for adversarial erasing.
        This means that all necessary fields are set.
        """
        necessary = ["mean", "std", "translate_frac", "degrees", "scale"]
        return all(getattr(self, f) is not None for f in necessary)

    def to_pretty_string(self) -> str:
        data = asdict(self)
        body = pformat(data, indent=2, width=100, sort_dicts=False)
        meta = f"adversarial_erasing_ready={self.is_valid_for_adversarial_erasing()}"
        return f"{body}\n# {meta}"

    def __str__(self) -> str:
        return f"AugConfig(\n{self.to_pretty_string()}\n)"

    __repr__ = __str__


def to_aug_config(cfg: DictConfig) -> AugConfig:
    if not isinstance(cfg, DictConfig):
        raise ValueError(
            f"cfg_transforms must be an instance of DictConfig, got {type(cfg)}"
        )

    resize_width = cfg.get("resize_width", None)
    resize_height = cfg.get("resize_height", None)

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
        brightness = augmentation.get("brightness", None)
        contrast = augmentation.get("contrast", None)
        saturation = augmentation.get("saturation", None)
        hue = augmentation.get("hue", None)
        random_erasing_cfg = augmentation.get("random_erasing", None)
        if random_erasing_cfg is not None:
            re_p = random_erasing_cfg.get("p", None)
            re_scale_raw = random_erasing_cfg.get("scale", None)
            re_scale = tuple(re_scale_raw) if re_scale_raw is not None else None
            random_erasing = RandomErasingConfig(p=re_p, scale=re_scale)
        else:
            random_erasing = None
    else:
        translate_frac = None
        degrees = None
        scale = None
        fill_color = 0
        horizontal_flip = False
        vertical_flip = False
        brightness = None
        contrast = None
        saturation = None
        hue = None
        random_erasing = None

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
        brightness=brightness,
        contrast=contrast,
        saturation=saturation,
        hue=hue,
        random_erasing=random_erasing,
    )


def to_compose(aug_config: AugConfig, stage: str) -> transforms.Compose:
    assert isinstance(aug_config, AugConfig), (
        f"aug_config should be of type AugConfig, received type {type(aug_config)}"
    )

    assert stage in [
        "train",
        "val",
        "test",
    ], f"stage should be either 'train', 'val', or 'test', got '{stage}'"

    transform_list = []

    if stage == "train":
        tf = aug_config.translate_frac
        if tf is not None:
            translate = (tf, tf)
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

        cj_params = {}
        if aug_config.brightness is not None:
            cj_params["brightness"] = aug_config.brightness
        if aug_config.contrast is not None:
            cj_params["contrast"] = aug_config.contrast
        if aug_config.saturation is not None:
            cj_params["saturation"] = aug_config.saturation
        if aug_config.hue is not None:
            cj_params["hue"] = aug_config.hue
        if cj_params:
            transform_list.append(transforms.ColorJitter(**cj_params))

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

    if stage == "train" and aug_config.random_erasing:
        transform_list.append(
            transforms.RandomErasing(
                p=aug_config.random_erasing.p,
                scale=aug_config.random_erasing.scale,
            )
        )

    return transforms.Compose(transform_list)
