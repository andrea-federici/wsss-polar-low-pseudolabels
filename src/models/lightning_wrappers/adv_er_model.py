import os
from typing import List

import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
from torchvision.utils import make_grid, save_image

import src.data.image_processing as image_processing
from src.models.configs import AdversarialErasingBaseConfig

from .base_model import BaseModel


class AdversarialErasingModel(BaseModel):
    def __init__(
        self,
        model: torch.nn.Module,
        *,
        criterion: torch.nn.Module,
        optimizer_config: dict,
        adver_config: AdversarialErasingBaseConfig,
        multi_label: bool = False,
        threshold: float = 0.5,
    ):
        super().__init__(
            model,
            criterion,
            optimizer_config,
            multi_label=multi_label,
            threshold=threshold,
        )
        self.current_iteration = adver_config.iteration
        self.aug_config = adver_config.aug_config
        self.erase_strategy = adver_config.erase_strategy

        if not self.aug_config.is_valid_for_adversarial_erasing():
            raise ValueError(
                "Augmentation configuration is not valid for adversarial erasing. "
                "Ensure that 'mean', 'std', 'translate_frac', 'degrees', and 'scale' "
                "are set."
            )

        print(self.aug_config)

    def _process_batch(
        self,
        images: torch.Tensor,
        labels: torch.Tensor,
        img_paths: List[str],
        do_augment: bool = False,
    ) -> torch.Tensor:
        out = []
        for img, label, img_path in zip(images, labels, img_paths):
            img = self.erase_strategy.erase(
                img,
                img_name=img_path,
                label=label,
                current_iteration=self.current_iteration,
            )

            if do_augment:
                img = self._apply_transform(img)

            out.append(img)

        # Convert list to batch tensor
        return torch.stack(out)

    def _apply_transform(self, img):
        img = image_processing.unnormalize_image_by_statistics(
            img, self.aug_config.mean, self.aug_config.std
        )

        # If translate_frac=0.2, and the image size is (500, 500), the possible
        # translation range is [-100, 100] in both x and y directions.
        dx = torch.randint(
            int(-self.aug_config.translate_frac * self.aug_config.resize_width),
            int(
                self.aug_config.translate_frac * self.aug_config.resize_width + 1
            ),  # +1 to include max
            size=(1,),
        ).item()
        dy = torch.randint(
            int(-self.aug_config.translate_frac * self.aug_config.resize_height),
            int(
                self.aug_config.translate_frac * self.aug_config.resize_height + 1
            ),  # +1 to include max
            size=(1,),
        ).item()

        angle = torch.randint(
            -self.aug_config.degrees, self.aug_config.degrees + 1, size=(1,)
        ).item()

        min_s, max_s = self.aug_config.scale
        scale = torch.empty(1).uniform_(min_s, max_s).item()

        img = F.affine(
            img,
            angle=angle,
            translate=(dx, dy),
            scale=scale,
            shear=[0.0, 0.0],
            fill=[0],  # black in normalized image (gray-ish in original image)
        )

        flips = []

        if self.aug_config.horizontal_flip:
            flips.append(transforms.RandomHorizontalFlip(p=0.5))

        if self.aug_config.vertical_flip:
            flips.append(transforms.RandomVerticalFlip(p=0.5))

        flips_transform = transforms.Compose(flips)
        img = flips_transform(img)

        cj_params = {}
        if self.aug_config.brightness is not None:
            cj_params["brightness"] = self.aug_config.brightness
        if self.aug_config.contrast is not None:
            cj_params["contrast"] = self.aug_config.contrast
        if self.aug_config.saturation is not None:
            cj_params["saturation"] = self.aug_config.saturation
        if self.aug_config.hue is not None:
            cj_params["hue"] = self.aug_config.hue
        if cj_params:
            jitter = transforms.ColorJitter(**cj_params)
            img = jitter(img)

        img = image_processing.normalize_image_by_statistics(
            img, self.aug_config.mean, self.aug_config.std
        )

        if self.aug_config.random_erasing:
            re = transforms.RandomErasing(
                p=self.aug_config.random_erasing.p,
                scale=self.aug_config.random_erasing.scale,
            )
            img = re(img)

        return img

    def _maybe_save_images(
        self, images, img_paths, batch_idx, stage: str, every: int = 10
    ):
        if batch_idx % every == 0:
            save_dir = f"out/debug/images/{stage}"
            os.makedirs(save_dir, exist_ok=True)
            num_samples = min(len(images), 16)

            # Normalize images to [0, 1]
            mins = images.amin(dim=(1, 2, 3), keepdim=True)
            maxs = images.amax(dim=(1, 2, 3), keepdim=True)
            denom = (maxs - mins).clamp_min(1e-6)  # Avoid division by zero
            norm = (images - mins) / denom
            norm = norm.clamp(0, 1)  # Ensure values are in [0, 1]

            for i in range(num_samples):
                save_image(
                    norm[i],
                    os.path.join(
                        save_dir,
                        f"iter_{self.current_iteration}_batch_{batch_idx}_{img_paths[i].removesuffix('.png')}.png",
                    ),
                )

    def _maybe_save_grid(self, images, batch_idx, stage: str, every: int = 10):
        if batch_idx % every == 0:
            save_dir = f"out/debug/grids/{stage}"
            os.makedirs(save_dir, exist_ok=True)
            num_samples = min(len(images), 16)  # Ensure we don't exceed batch size
            sample_grid = make_grid(images[:num_samples], nrow=4, normalize=True)
            save_image(
                sample_grid,
                os.path.join(
                    save_dir, f"iter_{self.current_iteration}_batch_{batch_idx}.png"
                ),
            )

    def training_step(self, batch, batch_idx):
        images, labels, img_paths = batch
        processed_images = self._process_batch(
            images, labels, img_paths, do_augment=True
        )
        # Save images and grid only for the first epoch
        if self.current_epoch == 0:
            self._maybe_save_images(
                processed_images, img_paths, batch_idx, stage="train", every=20
            )
            self._maybe_save_grid(processed_images, batch_idx, stage="train", every=20)
        loss = super()._process_step("train", processed_images, labels)
        return loss

    def validation_step(self, batch, batch_idx):
        images, labels, img_paths = batch
        processed_images = self._process_batch(
            images, labels, img_paths, do_augment=False
        )
        if self.current_epoch == 0:
            self._maybe_save_images(
                processed_images, img_paths, batch_idx, stage="val", every=20
            )
            self._maybe_save_grid(processed_images, batch_idx, stage="val", every=20)
        loss = super()._process_step("val", processed_images, labels)
        return loss
