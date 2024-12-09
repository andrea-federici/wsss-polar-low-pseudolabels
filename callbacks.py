import pytorch_lightning.callbacks as cb

import train_config as tc

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
    filename=tc.cc_filename
)

# Learning Rate Monitor
lr_monitor = cb.LearningRateMonitor(logging_interval='epoch')

