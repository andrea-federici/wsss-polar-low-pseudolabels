
from lightning.pytorch.loggers import NeptuneLogger

from train_config import (
    neptune_project,
    neptune_api_key, 
    log_model_checkpoints
)


def create_neptune_logger(
    log_model_checkpoints: bool = log_model_checkpoints
) -> NeptuneLogger:
    return NeptuneLogger(
        project=neptune_project,
        api_key=neptune_api_key,
        log_model_checkpoints=log_model_checkpoints,
        # mode="offline",
    )