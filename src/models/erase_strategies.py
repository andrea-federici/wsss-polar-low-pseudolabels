from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch

from src.data.adversarial_erasing_io import (
    load_accumulated_heatmap,
    load_accumulated_mask,
)
from src.data.image_processing import (
    erase_region_using_heatmap,
    erase_region_using_mask,
)


@dataclass()
class BaseEraseStrategy(ABC):
    base_dir: str
    fill_color: float

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


@dataclass
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
                self.base_dir, img_name, label, current_iteration - 1
            )

            img = erase_region_using_heatmap(
                img.unsqueeze(0),
                accumulated_heatmap,
                threshold=self.heatmap_threshold,
                fill_color=self.fill_color,
            ).squeeze(0)
        return img


@dataclass
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
                self.base_dir, img_name, label, current_iteration - 1
            )

            img = erase_region_using_mask(
                img.unsqueeze(0),
                accumulated_mask,
                fill_color=self.fill_color,
            ).squeeze(0)
        return img
