
from src.models.torch import Xception
from src.train.setups import create_standard_setup
from src.train.components.trainers import create_trainer
from src.train.loggers import create_neptune_logger


def run():
    neptune_logger = create_neptune_logger()

    neptune_logger.experiment["source_files/train_config"].upload("train_config.py")

    torch_model = Xception(num_classes=2)

    lit_model, train_loader, val_loader = create_standard_setup(torch_model)

    # Train model
    trainer = create_trainer(neptune_logger)
    trainer.fit(lit_model, train_loader, val_loader)
