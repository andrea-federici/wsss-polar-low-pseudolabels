import os

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
    load_dotenv()

    print(OmegaConf.to_yaml(cfg, resolve=True))

    torch.set_float32_matmul_precision(cfg.matmul_precision)

    mode = cfg.mode
    if mode == 'single':
        train_single.run()
    elif mode == 'optuna':
        train_optuna.run()
    elif mode == 'adversarial_erasing':
        train_adv_er.run()
    elif mode == 'max_translations':
        finetune_max_tr.run()


if __name__ == "__main__":
    run()