
from models import XceptionModel
from train_setups import create_standard_setup
from trainers import create_trainer
import loggers


def run():
    neptune_logger = loggers.neptune_logger

    neptune_logger.experiment["source_files/train_config"].upload("train_config.py")

    torch_model = XceptionModel(num_classes=2)

    lit_model, train_loader, val_loader = create_standard_setup(torch_model)

    # Train model
    trainer = create_trainer(neptune_logger)
    trainer.fit(lit_model, train_loader, val_loader)
