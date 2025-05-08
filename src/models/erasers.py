from abc import ABC, abstractmethod

import torch

from src.data.heatmaps import load_accumulated
from src.data.image_processing import erase_region_using_heatmap


class BaseEraser(ABC):
    def __init__(self, fill_color=0):
        self.fill_color = fill_color

    @abstractmethod
    def erase(self, *args, **kwargs) -> torch.Tensor:
        pass


class HeatmapEraser(BaseEraser):
    def __init__(self, threshold: float, fill_color=0):
        super().__init__(fill_color)
        self.threshold = threshold

    def erase(
        self,
        img: torch.Tensor,
        *,
        img_name: str,
        label: int,
        base_heatmaps_dir: str,
        current_iteration: int,
        heatmap_threshold: float
    ) -> torch.Tensor:
        if current_iteration > 0:
            accumulated_heatmap = load_accumulated(
                base_heatmaps_dir, img_name, label, current_iteration - 1
            )

            img = erase_region_using_heatmap(
                img.unsqueeze(0),
                accumulated_heatmap,
                threshold=heatmap_threshold,
                fill_color=self.fill_color,
            ).squeeze(0)
