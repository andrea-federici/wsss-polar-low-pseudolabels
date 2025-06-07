import sys

import hydra
import torch
from dotenv import load_dotenv
from omegaconf import DictConfig, OmegaConf

from src.train.mode import train_adv_er, train_single  # train_optuna,; finetune_max_tr


@hydra.main(version_base=None, config_path="conf", config_name="config")
def run(cfg: DictConfig) -> None:
    validate_hardware_config(cfg)
    load_dotenv()

    # Mask sensitive information
    cfg_copy = OmegaConf.create(cfg)  # Create a mutable copy
    cfg_copy.neptune.api_key = "*****"  # Mask or remove the sensitive key
    print(OmegaConf.to_yaml(cfg_copy, resolve=True))

    torch.set_float32_matmul_precision(cfg.matmul_precision)

    mode = cfg.mode.name
    if mode == "single":
        train_single.run(cfg)
    # elif mode == 'optuna':
    #     train_optuna.run(cfg)
    elif mode == "adversarial_erasing":
        train_adv_er.run(cfg)
    # elif mode == 'max_translations':
    #     finetune_max_tr.run(cfg)
    else:
        raise ValueError(f"Invalid mode: {mode}")


def validate_hardware_config(cfg):
    if cfg.hardware.accelerator == "gpu" or cfg.hardware.accelerator == "cuda":
        if not torch.cuda.is_available():
            print("Error: GPU was requested but is not available on this " "machine.")
            sys.exit(1)
    print("Hardware configuration is valid.")


if __name__ == "__main__":
    run()
