import os

import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
from omegaconf import DictConfig
from torchvision.utils import make_grid, save_image

from src.data.image_processing import (
    erase_region_using_heatmap,
    normalize_image_by_statistics,
    unnormalize_image_by_statistics,
)
from src.models.lightning import BaseModel
from src.train.helper import load_accumulated


class AdversarialErasingModel(BaseModel):
    def __init__(
        self,
        model: torch.nn.Module,
        *,
        criterion: torch.nn.Module,
        optimizer_config: dict,
        current_iteration: int,
        train_config: DictConfig,
        transforms_config: dict,
    ):
        super().__init__(model, criterion, optimizer_config)
        self.current_iteration = current_iteration
        self.base_heatmaps_dir = train_config.heatmaps.base_directory
        self.train_config = train_config
        self.transforms_config = transforms_config

    def training_step(self, batch, batch_idx):
        images, labels, img_paths = batch

        erased_images = []
        for img, label, img_path in zip(images, labels, img_paths):
            if self.current_iteration > 0:
                accumulated_heatmap = load_accumulated(
                    self.base_heatmaps_dir, img_path, label, self.current_iteration - 1
                )

                img = erase_region_using_heatmap(
                    img.unsqueeze(0),
                    accumulated_heatmap,
                    threshold=self.train_config.heatmaps.threshold,
                    fill_color=self.train_config.heatmaps.fill_color,
                ).squeeze(0)

            image_width = self.transforms_config.image_width
            image_height = self.transforms_config.image_height
            mean = self.transforms_config.normalization.mean
            std = self.transforms_config.normalization.std

            erased_img = unnormalize_image_by_statistics(img, mean, std)

            aug_config = self.train_config.aug

            # If translate_frac=0.2, and the image size is (500, 500), the possible
            # translation range is [-100, 100] in both x and y directions.
            max_translation_fraction = aug_config.translate_frac
            dx = torch.randint(
                int(-max_translation_fraction * image_width),
                int(max_translation_fraction * image_width + 1),  # +1 to include max
                size=(1,),
            ).item()
            dy = torch.randint(
                int(-max_translation_fraction * image_height),
                int(max_translation_fraction * image_height + 1),  # +1 to include max
                size=(1,),
            ).item()

            max_rotation = aug_config.degrees
            angle = torch.randint(-max_rotation, max_rotation + 1, size=(1,)).item()

            max_scale_fraction = aug_config.scale
            scale = 1.0 + (torch.rand(1).item() - 0.5) * max_scale_fraction * 2

            erased_img = F.affine(
                erased_img,
                angle=angle,
                translate=(dx, dy),
                scale=scale,
                shear=[0.0, 0.0],
                fill=[0],  # black in normalized image (gray-ish in original image)
            )

            flips = []

            if aug_config.horizontal_flip:
                flips.append(transforms.RandomHorizontalFlip(p=0.5))

            if aug_config.vertical_flip:
                flips.append(transforms.RandomVerticalFlip(p=0.5))

            flips_transform = transforms.Compose(flips)
            erased_img = flips_transform(erased_img)

            erased_img = normalize_image_by_statistics(erased_img, mean, std)

            erased_images.append(erased_img)

        # Convert list to batch tensor
        erased_images = torch.stack(erased_images)

        # Save a sample batch every N iterations
        if batch_idx % 20 == 0:
            save_dir = "out/debug_images"
            os.makedirs(save_dir, exist_ok=True)

            num_samples = min(len(erased_images), 16)  # Ensure we don't exceed
            # batch size
            sample_grid = make_grid(erased_images[:num_samples], nrow=4, normalize=True)
            save_image(
                sample_grid,
                os.path.join(
                    save_dir, f"it_{self.current_iteration}_batch_{batch_idx}.png"
                ),
            )

            # print(f"Saved sample images from batch {batch_idx}")

        loss = super().process_step("train", erased_images, labels)
        return loss

    def validation_step(self, batch, batch_idx):
        images, labels, img_paths = batch

        erased_images = []
        for img, label, img_path in zip(images, labels, img_paths):
            if self.current_iteration > 0:
                accumulated_heatmap = load_accumulated(
                    self.base_heatmaps_dir, img_path, label, self.current_iteration - 1
                )

                img = erase_region_using_heatmap(
                    img.unsqueeze(0),
                    accumulated_heatmap,
                    threshold=self.train_config.heatmaps.threshold,
                    fill_color=self.train_config.heatmaps.fill_color,
                ).squeeze(0)

            erased_images.append(img)

        # Convert list to batch tensor
        erased_images = torch.stack(erased_images)

        # Save a sample batch every N iterations
        if batch_idx % 20 == 0:
            save_dir = "out/val_debug_images"
            os.makedirs(save_dir, exist_ok=True)

            num_samples = min(len(erased_images), 16)  # Ensure we don't exceed
            # batch size
            sample_grid = make_grid(erased_images[:num_samples], nrow=4, normalize=True)
            save_image(
                sample_grid,
                os.path.join(
                    save_dir, f"it_{self.current_iteration}_batch_{batch_idx}.png"
                ),
            )

        loss = super().process_step("val", erased_images, labels)
        return loss
