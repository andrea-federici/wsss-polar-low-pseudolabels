from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch

from src.data.heatmaps import load_accumulated
from src.data.image_processing import erase_region_using_heatmap


@dataclass()
class BaseEraseStrategy(ABC):
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
    base_heatmaps_dir: str
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
            accumulated_heatmap = load_accumulated(
                self.base_heatmaps_dir, img_name, label, current_iteration - 1
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
        pass
