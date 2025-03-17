from dataclasses import dataclass

from torch.utils.data import DataLoader
import lightning.pytorch.callbacks as cb
from lightning.pytorch import LightningModule, Trainer
from lightning.pytorch.loggers import NeptuneLogger
from omegaconf import DictConfig

from src.data.data_loading import create_data_loaders
from src.utils.model_utils import (
    torch_model_getter,
    lightning_model_getter,
    criterion_getter, 
    optimizer_getter
)
from src.train.loggers import create_neptune_logger
from src.data.transforms import get_transform


@dataclass
class TrainSetup:
    lightning_model: LightningModule
    train_loader: DataLoader
    val_loader: DataLoader
    trainer: Trainer
    logger: NeptuneLogger


# TODO: add support for optuna
# TODO: add support for max_translations
def get_train_setup(cfg: DictConfig) -> TrainSetup:
    
    ## LOGGER ##

    neptune_logger = create_neptune_logger(
        cfg.neptune.project,
        cfg.neptune.api_kei
    )

    ## TORCH MODEL ##

    torch_model = torch_model_getter(cfg.torch_model, cfg.num_classes)

    ## DATA LOADERS ##

    train_loader, val_loader, _ = create_data_loaders(
        cfg.data_dir,
        cfg.batch_size,
        cfg.num_workers,
        get_transform(cfg, 'train'),
        get_transform(cfg, 'val'),
        dataset_type=cfg.mode.dataset_type,
    )

    ## LOSS AND OPTIMIZER ##

    criterion = criterion_getter(cfg.criterion)
    optimizer = optimizer_getter(cfg.optimizer, torch_model, cfg.learning_rate)

    ## LIGHTNING MODEL ##

    lightning_model = lightning_model_getter(
        cfg.lightning_model,
        torch_model,
        criterion,
        optimizer
    )

    ## CALLBACKS ##

    callbacks = []

    # Early Stopping
    callbacks.append(cb.EarlyStopping(
        monitor='val_loss',
        mode='min',
        patience=cfg.early_stopping.patience
    ))

    # Model Checkpoint
    callbacks.append(cb.ModelCheckpoint(
        monitor='val_loss',
        mode='min',
        save_top_k=1,
        dirpath=cfg.checkpoint.directory,
        filename=cfg.checkpoint.filename,
    ))

    # Learning Rate Monitor
    callbacks.append(cb.LearningRateMonitor(logging_interval='epoch'))

    ## TRAINER ##
    trainer = Trainer(
        logger = neptune_logger,
        max_epochs = cfg.max_epochs,
        callbacks=callbacks,
        accelerator=cfg.hardware.accelerator,
        devices=cfg.hardware.devices,
        check_val_every_n_epoch=1,
        enable_progress_bar=True,
        log_every_n_steps=1
    )

    return TrainSetup(
        lightning_model=lightning_model,
        train_loader=train_loader,
        val_loader=val_loader,
        trainer=trainer,
        logger=neptune_logger
    )