import torch.nn as nn
import torch.optim as optim
from lightning.pytorch import LightningModule
from torch.optim.lr_scheduler import LRScheduler, ReduceLROnPlateau

from src.models.configs import AdversarialErasingBaseConfig, BaseConfig
from src.models.lightning import (  # MaxTranslationsModel,
    AdversarialErasingModel,
    BaseModel,
)
from src.models.torch import Xception


def torch_model_getter(model_name: str, num_classes: int = 2) -> nn.Module:
    if model_name == "xception":
        return Xception(num_classes=num_classes)
    else:
        raise ValueError(
            f"Invalid model name: {model_name}. " f"Available options: ['xception']"
        )


# TODO: add support for max_translations
# TODO: fix checkpoint loading for other lightning models
def lightning_model_getter(
    lightning_model_name: str,
    torch_model: nn.Module,
    *,
    criterion: nn.Module,
    optimizer_config: dict,
    model_config: BaseConfig,
    stage: str = "train",
    checkpoint_path: str = None,
) -> LightningModule:

    assert stage in [
        "train",
        "predict",
    ], f"Invalid stage: {stage}. Available options: ['train', 'predict']"

    if stage == "train":
        if checkpoint_path is not None:
            print(
                "Checkpoint path is provided, but it will be ignored since stage is "
                "set to 'train'."
            )

    if stage == "predict":
        assert (
            checkpoint_path is not None
        ), "Checkpoint path is required for prediction."

    if lightning_model_name == "base":
        return BaseModel(torch_model, criterion, optimizer_config)
    elif lightning_model_name == "adversarial_erasing":
        if not isinstance(model_config, AdversarialErasingBaseConfig):
            raise TypeError(
                "Mode 'adversarial_erasing' requires an adversarial "
                "erasing configuration."
            )
        if stage == "train":
            return AdversarialErasingModel(
                model=torch_model,
                criterion=criterion,
                optimizer_config=optimizer_config,
                adver_config=model_config,
            )
        elif stage == "predict":
            return AdversarialErasingModel.load_from_checkpoint(
                checkpoint_path=checkpoint_path,
                model=torch_model,
                criterion=criterion,
                optimizer_config=optimizer_config,
                adver_config=model_config,
            )
    else:
        raise ValueError(
            f"Invalid lightning model name: {lightning_model_name}. "
            f"Available options: ['base', 'adversarial_erasing']"
        )


def criterion_getter(criterion_name: str):
    criterions = {
        "cross_entropy": nn.CrossEntropyLoss(),
    }

    if criterion_name not in criterions:
        raise ValueError(
            f"Invalid criterion name: {criterion_name}. "
            f"Available options: {list(criterions.keys())}"
        )

    return criterions[criterion_name]


def optimizer_getter(
    optimizer_name: str, torch_model: nn.Module, learning_rate: float
) -> optim.Optimizer:

    if optimizer_name == "adam":
        # model.parameters() returns an iterator over all model
        # parameters (weights and biases)
        # Each parameter is a tensor, and each tensor has a 'requires_grad'
        # attribute, which is True if the parameter should be updated during
        # backpropagation, and False otherwise.
        # filter() is a function that takes a function and an iterable as input.
        # It applies the function to each element of the iterable and returns
        # only the elements for which the function returns True.
        # This line filters out the parameters that should not be updated during
        # backpropagation (frozen layers)
        return optim.Adam(
            filter(lambda p: p.requires_grad, torch_model.parameters()),
            lr=learning_rate,
        )
    else:
        raise ValueError(
            f"Invalid optimizer name: {optimizer_name}. " f"Available options: ['adam']"
        )


def lr_scheduler_getter(
    lr_scheduler_name: str,
    optimizer: optim.Optimizer,
    mode: str,
    patience: int,
    factor: float,
) -> LRScheduler:
    if lr_scheduler_name == "reduce_lr_on_plateau":
        return ReduceLROnPlateau(optimizer, mode=mode, patience=patience, factor=factor)
    else:
        raise ValueError(
            f"Invalid lr_scheduler name: {lr_scheduler_name}. "
            f"Available options: ['reduce_lr_on_plateu']"
        )
