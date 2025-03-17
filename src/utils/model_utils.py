import torch.optim as optim
import torch.nn as nn


def criterion_getter(criterion_name: str):
    criterions = {
        "cross_entropy": nn.CrossEntropyLoss(),
    }

    if criterion_name not in criterions:
        raise ValueError(f"Invalid criterion name: {criterion_name}. "
                         f"Available options: {list(criterions.keys())}")

    return criterions[criterion_name]


def optimizer_getter(
        optimizer_name: str, 
        torch_model: nn.Module, 
        learning_rate: float) -> optim.Optimizer:
    
    if optimizer_name == 'adam':
        # model.parameters() returns an iterator over all model 
        # parameters (weights and biases)
        # Each parameter is a tensor, and each tensor has a 'requires_grad' 
        # attribute, which is True if the parameter should be updated during 
        # backpropagation, and False otherwise.
        # filter() is a function that takes a function and an iterable as input.
        # It applies the function to each element of the iterable and returns 
        # only the elements for which the function returns True.
        # This line filters out the parameters that should not be updated during 
        # backpropagation (frozen layers)
        return optim.Adam(
            filter(lambda p: p.requires_grad, torch_model.parameters()),
            lr=learning_rate
        )
    else:
        raise ValueError(f"Invalid optimizer name: {optimizer_name}. "
                         f"Available options: ['adam']")