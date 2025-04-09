from src.utils.neptune_utils import NeptuneLogger


def create_neptune_logger(
    project: str,
    api_key: str,
    # log_model_checkpoints: bool = False,
) -> NeptuneLogger:
    return NeptuneLogger(
        project_name=project,
        api_key=api_key,
        # log_model_checkpoints=log_model_checkpoints,
        # mode="offline",
    )