import os
import sys

from dotenv import load_dotenv
import torch
import hydra
from omegaconf import DictConfig, OmegaConf

from src.train.modes import (
    train_single,
    train_optuna,
    train_adv_er,
    finetune_max_tr
)


@hydra.main(version_base=None, config_path='conf', config_name='config')
def run(cfg: DictConfig) -> None:
    validate_hardware_config(cfg)
    load_dotenv()

    print(OmegaConf.to_yaml(cfg, resolve=True))

    cuda_available = torch.cuda.is_available()
    accelerator = 'gpu' if cuda_available else 'cpu'
    device = torch.device("cuda" if cuda_available else "cpu")
    print(f"Device: {device}. Accelerator: {accelerator}")

    torch.set_float32_matmul_precision(cfg.matmul_precision)

    mode = cfg.mode
    if mode == 'single':
        train_single.run(cfg)
    elif mode == 'optuna':
        train_optuna.run(cfg)
    elif mode == 'adversarial_erasing':
        train_adv_er.run(cfg)
    elif mode == 'max_translations':
        finetune_max_tr.run(cfg)


def validate_hardware_config(cfg):
    if cfg.hardware.accelerator = 'gpu' or cfg.hardware.accelerator = 'cuda':
        if not torch.cuda.is_available():
            print('Error: GPU was requested but is not available on this '
                'machine.')
            sys.exit(1)


if __name__ == "__main__":
    run()