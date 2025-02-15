from lightning.pytorch import Trainer
from optuna.integration import PyTorchLightningPruningCallback

import train_config as tc
import callbacks as cb

def create_trainer(neptune_logger = None,
                   using_optuna = False,
                   optuna_trial = None):
    callbacks = cb.create_callbacks()

    if using_optuna:
        callbacks.append(
            PyTorchLightningPruningCallback(
                optuna_trial,
                monitor='val_loss'
            )
        )

    return Trainer(
        logger = neptune_logger,
        max_epochs = tc.max_epochs,
        callbacks=callbacks,
        accelerator=tc.accelerator,
        devices=1,
        check_val_every_n_epoch=1,
        enable_progress_bar=True,
        log_every_n_steps=1 # TODO: progress bars still update every 20 epochs
        #                       for some reason.
    )