from dataclasses import dataclass

from src.data.augmentation import AugConfig
from src.models.erase_strategies import BaseEraseStrategy


@dataclass
class AdversarialErasingBaseConfig:
    iteration: int
    aug_config: AugConfig
    erase_strategy: BaseEraseStrategy


@dataclass
class AdversarialErasingHeatmapConfig(AdversarialErasingBaseConfig):
    base_heatmaps_dir: str
    heatmap_threshold: float


@dataclass
class AdversarialErasingMaskConfig(AdversarialErasingBaseConfig):
    base_masks_dir: str
