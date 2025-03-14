import lightning.pytorch.callbacks as cb

import train_config as tc


def create_callbacks(checkpoint_filename: str = tc.cc_filename):
    # Early Stopping
    early_stopping = cb.EarlyStopping(
        monitor='val_loss',
        mode='min',
        patience=tc.es_patience
    )

    # Model Checkpoint
    model_checkpoint = cb.ModelCheckpoint(
        monitor='val_loss',
        mode='min',
        save_top_k=1,
        dirpath='checkpoints/',
        filename=checkpoint_filename,
    )

    # Learning Rate Monitor
    lr_monitor = cb.LearningRateMonitor(logging_interval='epoch')

    return [early_stopping, model_checkpoint, lr_monitor]

