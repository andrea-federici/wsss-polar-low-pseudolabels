from dataclasses import dataclass

from src.data.augmentation import AugConfig
from src.models.erase_strategies import BaseEraseStrategy


@dataclass
class BaseConfig:
    pass


@dataclass
class AdversarialErasingBaseConfig(BaseConfig):
    iteration: int
    aug_config: AugConfig
    erase_strategy: BaseEraseStrategy
