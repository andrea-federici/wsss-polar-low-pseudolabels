from omegaconf import DictConfig

from src.train.setup import setup_training


def run(cfg: DictConfig) -> None:
    ts = setup_training(cfg)

    logger = ts.logger
    logger.experiment["source_files/train_config"].upload("train_config.py")

    # Train model
    ts.trainer.fit(ts.lightning_model, ts.train_loader, ts.val_loader)
