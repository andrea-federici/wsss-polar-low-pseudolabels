from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum

import torch

from src.data.adversarial_erasing_io import (
    load_accumulated_heatmap,
    load_accumulated_mask,
)
from src.data.image_processing import (
    erase_region_using_heatmap,
    erase_region_using_mask,
)


class NegativeLoadStrategy(Enum):
    RANDOM = "random"
    PL_SPECIFIC = "pl_specific"  # Specific to the naming convention used in the polar
    # lows project, where the first 6 characters of the image name can be used to load
    # a negative sample corresponding to a positive image.

    @classmethod
    def list(cls) -> list[str]:
        return [strategy.value for strategy in cls]


@dataclass(frozen=True, kw_only=True)
class BaseEraseStrategy(ABC):
    base_dir: str
    fill_color: float
    negative_load_strategy: NegativeLoadStrategy = NegativeLoadStrategy.RANDOM

    def __post_init__(self):
        raw = self.negative_load_strategy

        # Ensure that the negative load strategy is a valid enum value. Note that we
        # don't need to check for the case where it's already an instance of
        # NegativeLoadStrategy, since it can only be set to one of the enum values.
        if isinstance(raw, str) and raw.lower() not in NegativeLoadStrategy.list():
            raise ValueError(
                f"Invalid negative load strategy: '{raw}'. "
                f"Must be one of {NegativeLoadStrategy.list()}."
            )

        # If a string is passed, convert it to the corresponding enum value
        strat = (
            NegativeLoadStrategy(raw.lower())
            if isinstance(self.negative_load_strategy, str)
            else self.negative_load_strategy
        )

        # Use object.__setattr__ to set the frozen dataclass attribute
        object.__setattr__(self, "negative_load_strategy", strat)

    @abstractmethod
    def erase(
        self,
        img: torch.Tensor,
        *,
        img_name: str,
        label: int,
        current_iteration: int,
    ) -> torch.Tensor:
        pass


@dataclass(frozen=True, kw_only=True)
class HeatmapEraseStrategy(BaseEraseStrategy):
    heatmap_threshold: float

    def erase(
        self,
        img: torch.Tensor,
        *,
        img_name: str,
        label: int,
        current_iteration: int,
    ) -> torch.Tensor:
        if current_iteration > 0:
            accumulated_heatmap = load_accumulated_heatmap(
                self.base_dir,
                img_name,
                label,
                current_iteration - 1,
                negative_load_strategy=self.negative_load_strategy.value,
            )

            img = erase_region_using_heatmap(
                img.unsqueeze(0),
                accumulated_heatmap,
                threshold=self.heatmap_threshold,
                fill_color=self.fill_color,
            ).squeeze(0)
        return img


@dataclass(frozen=True, kw_only=True)
class MaskEraseStrategy(BaseEraseStrategy):
    def erase(
        self,
        img: torch.Tensor,
        *,
        img_name: str,
        label: int,
        current_iteration: int,
    ) -> torch.Tensor:
        if current_iteration > 0:
            accumulated_mask = load_accumulated_mask(
                self.base_dir,
                img_name,
                label,
                current_iteration - 1,
                negative_load_strategy=self.negative_load_strategy.value,
            )

            img = erase_region_using_mask(
                img.unsqueeze(0),
                accumulated_mask,
                fill_color=self.fill_color,
            ).squeeze(0)
        return img
