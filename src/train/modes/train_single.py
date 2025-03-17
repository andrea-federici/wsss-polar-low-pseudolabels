from omegaconf import DictConfig

from src.models.torch import Xception
from src.models.lightning import BaseModel
from src.data.data_loading import create_data_loaders
from src.train.setups import create_standard_setup
from src.train.components.trainers import create_trainer
from src.utils.model_utils import criterion_getter, optimizer_getter
from src.data.transforms import get_transform
from src.train.loggers import create_neptune_logger


def run(cfg: DictConfig) -> None:
    neptune_logger = create_neptune_logger(
        cfg.neptune.project,
        cfg.neptune.api_kei
    )

    neptune_logger.experiment["source_files/train_config"].upload("train_config.py")

    torch_model = Xception(num_classes=2)

    lit_model, train_loader, val_loader = create_standard_setup(torch_model)

    train_loader, val_loader, _ = create_data_loaders(
        cfg.data_dir,
        cfg.batch_size,
        cfg.num_workers,
        get_transform(cfg, 'train'),
        get_transform(cfg, 'val'),
        dataset_type=cfg.mode.dataset_type,
    )

    criterion = criterion_getter(cfg.criterion)
    optimizer = optimizer_getter(cfg.optimizer, torch_model, cfg.learning_rate)

    lightning_model = BaseModel()

    # Train model
    trainer = create_trainer(neptune_logger)
    trainer.fit(lit_model, train_loader, val_loader)
