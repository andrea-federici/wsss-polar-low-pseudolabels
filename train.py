import argparse

import torch.nn as nn
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import NeptuneLogger

import train_config as tc
from data_loader import create_data_loaders
from models import XceptionModel
from model_container import ModelContainer
import callbacks as cb
from optimizers import adam

# Suppress the warning related to the creation of DataLoader using a high 
# number of num_workers
import warnings
warnings.filterwarnings('ignore', message='.*DataLoader will create.*')

parser = argparse.ArgumentParser()
parser.add_argument(
    '--data-dir',
    type=str,
    default=tc.default_data_dir,
    help=f'Path to the data directory (default: {tc.default_data_dir})',
)

args = parser.parse_args()
data_dir = args.data_dir

# Create data loaders
train_loader, val_loader, _ = create_data_loaders(tc.get_train_dir(data_dir),
                                                  tc.get_test_dir(data_dir),
                                                  tc.batch_size,
                                                  tc.num_workers,
                                                  tc.verbose)

torch_model = XceptionModel(num_classes=2)

criterion = nn.CrossEntropyLoss()
optimizer = adam(torch_model)

lit_model = ModelContainer(torch_model, criterion, optimizer).to(tc.device)

neptune_logger = NeptuneLogger(
    project="andreaf/polarlows",
    api_key=tc.neptune_api_key,
    log_model_checkpoints=True
)

neptune_logger.experiment["source_files/train_config"].upload("train_config.py")

# Train model
trainer = Trainer(
    logger = neptune_logger,
    max_epochs = tc.max_epochs,
    callbacks=[cb.early_stopping, cb.lr_monitor, cb.model_checkpoint],
    accelerator=tc.accelerator,
    devices=1,
    check_val_every_n_epoch=1,
    enable_progress_bar=True,
    log_every_n_steps=1 # TODO: progress bars still update every 20 epochs
    #                       for some reason.
)

trainer.fit(lit_model, train_loader, val_loader)

