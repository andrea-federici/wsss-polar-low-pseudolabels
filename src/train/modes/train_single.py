from omegaconf import DictConfig

from src.train.setups import train_setup
from src.train.components.trainers import create_trainer
from src.train.loggers import create_neptune_logger


def run(cfg: DictConfig) -> None:
    neptune_logger.experiment["source_files/train_config"].upload("train_config.py")

    lightning_model, train_loader, val_loader = train_setup(cfg)

    # Train model
    trainer = create_trainer(neptune_logger)
    trainer.fit(lit_model, train_loader, val_loader)
