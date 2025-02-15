import torch.optim as optim
from torch.nn import Module

from train_config import learning_rate

def adam(torch_model: Module, learning_rate: float = learning_rate):
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

