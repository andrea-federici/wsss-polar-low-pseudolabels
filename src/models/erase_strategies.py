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
    FIRST_SIX = "first_six"

    @classmethod
    def list(cls) -> list[str]:
        return [strategy.value for strategy in cls]


@dataclass(frozen=True, kw_only=True)
class BaseEraseStrategy(ABC):
    base_dir: str
    fill_color: float
    negative_load_strategy: NegativeLoadStrategy = NegativeLoadStrategy.RANDOM

    # TODO: what if the user passes in a string that is not a valid strategy?
    def __post_init__(self):
        # if it was passed in as a string, convert it
        strat = (
            NegativeLoadStrategy(self.negative_load_strategy)
            if isinstance(self.negative_load_strategy, str)
            else self.negative_load_strategy
        )
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
