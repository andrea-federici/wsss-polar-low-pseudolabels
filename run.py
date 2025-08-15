import hydra
import torch
from dotenv import load_dotenv
from omegaconf import DictConfig, OmegaConf

from src.train.mode import train_adv_er, train_single


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
    elif mode == "adversarial_erasing":
        train_adv_er.run(cfg)
    else:
        raise ValueError(f"Invalid mode: {mode}")


def validate_hardware_config(cfg):
    device = cfg.hardware.device
    accelerator = cfg.hardware.accelerator
    cuda_available = torch.cuda.is_available()

    if device not in ["cpu", "cuda"]:
        raise ValueError(
            f"Invalid device: {device}. " "Available options: ['cpu', 'cuda']"
        )

    if device == "cuda" and not cuda_available:
        raise ValueError(
            "CUDA device requested but CUDA is not available on this machine."
        )

    accel = "cuda" if accelerator in ["gpu", "cuda"] else "cpu"
    if accel == "cuda" and not cuda_available:
        raise ValueError(
            "GPU/CUDA accelerator requested but CUDA is not available on this machine."
        )

    print(
        f"Hardware configuration is valid. Using device: {device}, accelerator: {accelerator}"
    )


if __name__ == "__main__":
    run()
