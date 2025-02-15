
import torch.nn as nn

from data_loader import create_data_loaders, create_data_loaders_max_tr
import train_config as tc
from optimizers import adam
from lit_model import LitModel
from iterative_training.lit_model_custom_transl import LitModelCustomTransl


def create_standard_setup(torch_model):
    train_loader, val_loader, criterion, optimizer = common_setup(torch_model)

    lit_model = LitModel(torch_model, criterion, optimizer).to(tc.device)

    return lit_model, train_loader, val_loader


def create_model_ckpt_setup(torch_model, lit_model_ckpt_path: str):
    train_loader, val_loader, criterion, optimizer = common_setup(torch_model)

    lit_model = LitModel.load_from_checkpoint(
        lit_model_ckpt_path,
        model=torch_model,
        criterion=criterion,
        optimizer=optimizer
    ).to(tc.device)

    return lit_model, train_loader, val_loader


def create_max_tr_model_ckpt_setup(torch_model, lit_model_ckpt_path: str, max_translations):
    train_loader, val_loader, criterion, optimizer = max_translations_setup(torch_model, max_translations)

    lit_model_ckpt = LitModel.load_from_checkpoint(
        lit_model_ckpt_path,
        model=torch_model,
        criterion=criterion,
        optimizer=optimizer
    ).to(tc.device)

    lit_model = LitModelCustomTransl(lit_model_ckpt.model, criterion, optimizer, max_translations).to(tc.device)

    return lit_model, train_loader, val_loader


# DEFINE BETTER NAMES FOR THE FOLLOWING FUNCTIONS (THEY SHOULD ALMOST BE PRIVATE, NOT MEANT TO BE USED FROM OUTSIDE)

def common_setup(torch_model):
    # Create data loaders
    train_loader, val_loader, _ = create_data_loaders(
        tc.train_dir(),
        tc.test_dir(),
        tc.batch_size,
        tc.num_workers,
        tc.verbose
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = adam(torch_model)

    return train_loader, val_loader, criterion, optimizer


def max_translations_setup(torch_model, max_translations):
    # Create data loaders
    train_loader, val_loader, _ = create_data_loaders_max_tr(
        tc.train_dir(),
        tc.test_dir(),
        tc.batch_size,
        tc.num_workers,
        max_translations,
        tc.verbose
    ) 

    criterion = nn.CrossEntropyLoss()
    # TODO: check that adam is intantiated again from scratch
    optimizer = adam(torch_model, learning_rate=tc.ft_learning_rate) # TODO: move parameter learning rate to function definition ?

    return train_loader, val_loader, criterion, optimizer