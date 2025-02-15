
from lightning.pytorch.loggers import NeptuneLogger

import train_config as tc

neptune_logger = NeptuneLogger(
    project="andreaf/polarlows",
    api_key=tc.neptune_api_key,
    log_model_checkpoints=tc.log_model_checkpoints,
    # mode="offline",
)