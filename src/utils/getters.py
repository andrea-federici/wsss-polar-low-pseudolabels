import torch.nn as nn
import torch.optim as optim
from lightning.pytorch import LightningModule
from torch.optim.lr_scheduler import LRScheduler, ReduceLROnPlateau

from src.models.configs import AdversarialErasingBaseConfig, BaseConfig
from src.models.lightning_wrappers import AdversarialErasingModel, BaseModel
from src.models.torch import Xception
from src.utils.constants import ADVERSARIAL_ERASING_MODEL, BASE_MODEL


def torch_model_getter(
    model_name: str, num_classes: int, *, device: str = "cpu"
) -> nn.Module:
    models = {
        "xception": Xception(num_classes=num_classes),
    }
    if model_name not in models:
        raise ValueError(
            f"Invalid model name: {model_name}. Available options: {list(models.keys())}"
        )
    model = models[model_name].to(device)
    return model


def lightning_model_getter(
    lightning_model_name: str,
    torch_model: nn.Module,
    *,
    criterion: nn.Module,
    optimizer_config: dict,
    model_config: BaseConfig = None,
    device: str = "cpu",
) -> LightningModule:
    if lightning_model_name == BASE_MODEL:
        return BaseModel(torch_model, criterion, optimizer_config).to(device)
    elif lightning_model_name == ADVERSARIAL_ERASING_MODEL:
        if not isinstance(model_config, AdversarialErasingBaseConfig):
            raise TypeError(
                f"Mode '{ADVERSARIAL_ERASING_MODEL}' requires an adversarial erasing "
                "configuration."
            )
        return AdversarialErasingModel(
            model=torch_model,
            criterion=criterion,
            optimizer_config=optimizer_config,
            adver_config=model_config,
        ).to(device)
    else:
        raise ValueError(
            f"Invalid lightning model name: {lightning_model_name}. "
            f"Available options: ['{BASE_MODEL}', '{ADVERSARIAL_ERASING_MODEL}']"
        )


def criterion_getter(criterion_name: str):
    criterions = {
        "cross_entropy": nn.CrossEntropyLoss(),
        "bce_with_logits": nn.BCEWithLogitsLoss(),
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
    optimizers = {
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
        "adam": optim.Adam(
            filter(lambda p: p.requires_grad, torch_model.parameters()),
            lr=learning_rate,
        )
    }

    if optimizer_name not in optimizers:
        raise ValueError(
            f"Invalid optimizer name: {optimizer_name}. "
            f"Available options: {list(optimizers.keys())}"
        )
    else:
        return optimizers[optimizer_name]


def lr_scheduler_getter(
    lr_scheduler_name: str,
    optimizer: optim.Optimizer,
    mode: str,
    patience: int,
    factor: float,
) -> LRScheduler:
    schedulers = {
        "reduce_lr_on_plateau": ReduceLROnPlateau(
            optimizer, mode=mode, patience=patience, factor=factor
        )
    }

    if lr_scheduler_name not in schedulers:
        raise ValueError(
            f"Invalid lr_scheduler name: {lr_scheduler_name}. "
            f"Available options: {list(schedulers.keys())}"
        )
    else:
        return schedulers[lr_scheduler_name]
