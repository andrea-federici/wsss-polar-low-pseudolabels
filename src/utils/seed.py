"""Utility helpers for deterministic experiment seeding."""

from __future__ import annotations

import os
import random
import warnings
from typing import Callable, Optional

import numpy as np
import torch
from lightning.pytorch import seed_everything as lightning_seed_everything


def configure_seed(seed: int, *, deterministic: bool = True) -> torch.Generator:
    """Seed all known random number generators and return a torch generator.

    Args:
        seed: The base seed to use for python, numpy, torch and Lightning.
        deterministic: When ``True`` additional flags are toggled to favour
            deterministic CUDA kernels.

    Returns:
        A ``torch.Generator`` instance seeded with ``seed`` that can be used
        when constructing deterministic dataloaders or samplers.
    """

    if seed is None:
        raise ValueError("A numeric seed must be provided for deterministic runs.")

    os.environ["PYTHONHASHSEED"] = str(seed)

    # Lightning seeds python, numpy and torch (including CUDA) and optionally
    # configures dataloader workers for us.
    lightning_seed_everything(seed, workers=True)

    # Redundantly seed the core libraries to ensure reproducibility even when
    # Lightning's helper changes behaviour in the future.
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # Required for deterministic cuBLAS on Ampere+ GPUs. Setting the value
        # here is harmless on CPUs.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

        if hasattr(torch, "use_deterministic_algorithms"):
            try:
                torch.use_deterministic_algorithms(True, warn_only=True)  # type: ignore[arg-type]
            except TypeError:
                torch.use_deterministic_algorithms(True)  # type: ignore[call-arg]
            except RuntimeError as exc:
                warnings.warn(
                    f"Deterministic algorithms could not be enabled: {exc}",
                    RuntimeWarning,
                )

        # OpenCV maintains an internal RNG used by some operations. Guard the
        # import so that this module stays optional.
        try:
            import cv2  # type: ignore

            cv2.setRNGSeed(seed)
        except Exception:
            # Ignore missing OpenCV installations – it is an optional dependency
            # for this project.
            pass

    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


def make_worker_init_fn(seed: Optional[int]) -> Optional[Callable[[int], None]]:
    """Return a ``worker_init_fn`` that deterministically seeds workers."""

    if seed is None:
        return None

    def _seed_worker(worker_id: int) -> None:
        worker_seed = (seed + worker_id) % (2**32)
        random.seed(worker_seed)
        np.random.seed(worker_seed)
        torch.manual_seed(worker_seed)

    return _seed_worker


def make_torch_generator(seed: Optional[int]) -> Optional[torch.Generator]:
    """Create a seeded ``torch.Generator`` when a seed is supplied."""

    if seed is None:
        return None

    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator