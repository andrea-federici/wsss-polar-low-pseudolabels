from dataclasses import dataclass

import lightning.pytorch.callbacks as cb
from lightning.pytorch import LightningModule, Trainer
from lightning.pytorch.loggers import NeptuneLogger
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from src.data.augmentation import to_aug_config, to_compose
from src.data.data_loaders import create_data_loaders
from src.models.configs import AdversarialErasingBaseConfig
from src.models.erase_strategies import HeatmapEraseStrategy, MaskEraseStrategy
from src.train.logger import create_neptune_logger
from src.utils.getters import (criterion_getter, lightning_model_getter,
                               lr_scheduler_getter, optimizer_getter,
                               torch_model_getter)


@dataclass
class TrainSetup:
    lightning_model: LightningModule
    train_loader: DataLoader
    val_loader: DataLoader
    trainer: Trainer
    logger: NeptuneLogger


def setup_training(cfg: DictConfig, *, iteration: int = None) -> TrainSetup:
    device = cfg.hardware.device

    ## LOGGER ##
    neptune_logger = create_neptune_logger(cfg.logger.project, cfg.logger.api_key)

    train_cfg = cfg.training
    ## TORCH MODEL ##
    torch_model = torch_model_getter(train_cfg.torch_model, train_cfg.num_classes, device=device)

    ## DATA LOADERS ##
    aug_config = to_aug_config(cfg.data.transforms)
    train_loader, val_loader, _ = create_data_loaders(
        cfg.data.directory,
        batch_size=train_cfg.batch_size,
        num_workers=cfg.hardware.num_workers,
        transform_train=to_compose(aug_config, cfg.mode.train_data_transform),
        transform_val=to_compose(aug_config, "val"),
        dataset_type=cfg.mode.dataset_type,
        pin_memory=True if device == "cuda" else False,
    )

    ## LOSS AND OPTIMIZER ##
    criterion = criterion_getter(train_cfg.criterion)
    optimizer = optimizer_getter(train_cfg.optimizer, torch_model, train_cfg.learning_rate)

    ## OPTIMIZER CONFIGURATION ##
    lr_scheduler = lr_scheduler_getter(
        train_cfg.lr_scheduler.name,
        optimizer,
        train_cfg.lr_scheduler.mode,
        train_cfg.lr_scheduler.patience,
        train_cfg.lr_scheduler.factor,
    )
    optimizer_config = {
        "optimizer": optimizer,
        "lr_scheduler": {
            "scheduler": lr_scheduler,
            "monitor": train_cfg.lr_scheduler.monitor,
            "interval": "epoch",
        },
    }

    ## LIGHTNING MODEL ##
    lightning_model_name = cfg.mode.lightning_model
    if lightning_model_name == "base":
        lightning_model = lightning_model_getter(
            lightning_model_name,
            torch_model,
            criterion=criterion,
            optimizer_config=optimizer_config,
            device=device,
        )
    elif lightning_model_name == "adversarial_erasing":
        if iteration is None:
            raise ValueError(
                "Iteration value must be provided with adversarial erasing"
            )
        if cfg.mode.erase_strategy == "heatmap":
            heatmaps_config = cfg.mode.train_config.heatmaps
            envelope_cfg = heatmaps_config.get("area_envelope", {})
            heatmap_erase_strategy = HeatmapEraseStrategy(
                fill_color=heatmaps_config.fill_color,
                base_dir=heatmaps_config.base_directory,
                heatmap_threshold=heatmaps_config.threshold,
                negative_load_strategy=heatmaps_config.negative_load_strategy,
                envelope_start=envelope_cfg.get("start_iteration", 2),
                envelope_scale=envelope_cfg.get("scale", 0.1),
            )
            model_config = AdversarialErasingBaseConfig(
                iteration=iteration,
                aug_config=aug_config,
                erase_strategy=heatmap_erase_strategy,
            )
        elif cfg.mode.erase_strategy == "mask":
            mask_erase_strategy = MaskEraseStrategy(
                base_dir=cfg.mode.train_config.heatmaps.base_directory,
                fill_color=cfg.mode.train_config.heatmaps.fill_color,
                negative_load_strategy=cfg.mode.train_config.heatmaps.negative_load_strategy,
            )
            model_config = AdversarialErasingBaseConfig(
                iteration=iteration,
                aug_config=aug_config,
                erase_strategy=mask_erase_strategy,
            )
        else:
            raise ValueError(
                f"Adversarial erasing strategy '{cfg.mode.erase_strategy}' not supported."
            )
        lightning_model = lightning_model_getter(
            lightning_model_name,
            torch_model,
            criterion=criterion,
            optimizer_config=optimizer_config,
            model_config=model_config,
            device=device,
        )
    else:
        raise ValueError(f"Lightning model '{lightning_model_name}' not supported.")

    ## CALLBACKS ##

    callbacks = []

    # Early Stopping
    callbacks.append(
        cb.EarlyStopping(
            monitor=train_cfg.early_stopping.monitor,
            mode=train_cfg.early_stopping.mode,
            patience=train_cfg.early_stopping.patience,
        )
    )

    # Model Checkpoint
    callbacks.append(
        cb.ModelCheckpoint(
            monitor=train_cfg.checkpoint.monitor,
            mode=train_cfg.checkpoint.mode,
            save_top_k=1,
            dirpath=train_cfg.checkpoint.directory,
            filename=train_cfg.checkpoint.name_prefix + str(iteration),
        )
    )

    # Learning Rate Monitor
    # This is not required by the learning rate scheduler, but it is useful
    # for logging the learning rate values at each step/epoch.
    callbacks.append(cb.LearningRateMonitor(logging_interval="epoch"))

    ## TRAINER ##
    trainer = Trainer(
        logger=neptune_logger,
        max_epochs=train_cfg.max_epochs,
        callbacks=callbacks,
        accelerator=cfg.hardware.accelerator,
        devices=cfg.hardware.devices,
        check_val_every_n_epoch=1,
        enable_progress_bar=True,
        log_every_n_steps=1,
    )

    return TrainSetup(
        lightning_model=lightning_model,
        train_loader=train_loader,
        val_loader=val_loader,
        trainer=trainer,
        logger=neptune_logger,
    )
