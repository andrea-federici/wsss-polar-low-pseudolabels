import os

import torch
from torchvision.utils import make_grid, save_image
import torchvision.transforms as transforms
import torchvision.transforms.functional as F

from src.models.lightning.base_model import BaseModel
from src.data.image_processing import (
    adversarial_erase,
    normalize_image_by_statistics,
    unnormalize_image_by_statistics
)
from src.train.helpers.adv_er_helper import load_accumulated_heatmap


class AdversarialErasingModel(BaseModel):
    def __init__(
        self, 
        model: torch.nn.Module, 
        criterion: torch.nn.Module, 
        optimizer: torch.optim.Optimizer, 
        current_iteration: int, 
        base_heatmaps_dir: str, 
        threshold: float = 0.5, 
        fill_color: float = 0
    ):
        super().__init__(model, criterion, optimizer)
        self.current_iteration = current_iteration
        self.base_heatmaps_dir = base_heatmaps_dir
        self.threshold = threshold
        self.fill_color = fill_color
        

    def training_step(self, batch, batch_idx):
        images, labels, img_paths = batch

        erased_images = []
        for img, label, img_path in zip(images, labels, img_paths):
            if self.current_iteration > 0:
                accumulated_heatmap = load_accumulated_heatmap(
                    self.base_heatmaps_dir, img_path, label, self.current_iteration, self.heatmap_files
                )

                img = adversarial_erase(
                    img.unsqueeze(0), 
                    accumulated_heatmap,
                    threshold=self.threshold,
                    fill_color=self.fill_color
                ).squeeze(0)

            erased_img = unnormalize_image_by_statistics(img, mean, std)

            max_dx = resized_image_res[0] / 2
            max_dy = resized_image_res[1] / 2

            max_translation_fraction = 0.2
            dx = torch.randint(
                int(-max_translation_fraction*max_dx), 
                int(max_translation_fraction*max_dx+1), 
                size=(1,)
            ).item()
            dy = torch.randint(
                int(-max_translation_fraction*max_dy), 
                int(max_translation_fraction*max_dy+1), 
                size=(1,)
            ).item()

            max_rotation = 20
            angle = torch.randint(-max_rotation, max_rotation+1, size=(1,)) \
                .item()

            max_scale_fraction = 0.1
            scale = 1.0 + (torch.rand(1).item()-0.5) * max_scale_fraction * 2

            erased_img = F.affine(
                erased_img,
                angle=angle,
                translate=(dx, dy),
                scale=scale,
                shear=[0.0, 0.0],
                fill=[0] # black
            )

            random_transforms = transforms.Compose([
                transforms.RandomApply(
                    [transforms.RandomHorizontalFlip()], 
                    p=0.5
                ),
                transforms.RandomApply(
                    [transforms.RandomVerticalFlip()], 
                    p=0.5
                )
            ])
            erased_img = random_transforms(erased_img)

            erased_img = normalize_image_by_statistics(erased_img, mean, std)

            erased_images.append(erased_img)

        # Convert list to batch tensor
        erased_images = torch.stack(erased_images)

        # Save a sample batch every N iterations
        if batch_idx % 20 == 0:
            save_dir = "debug_images"
            os.makedirs(save_dir, exist_ok=True)

            num_samples = min(len(erased_images), 16) # Ensure we don't exceed 
            # batch size
            sample_grid = make_grid(
                erased_images[:num_samples], 
                nrow=4,
                normalize=True
            )
            save_image(sample_grid, os.path.join(
                save_dir, 
                f"it_{self.current_iteration}_batch_{batch_idx}.png"
            ))

            # print(f"Saved sample images from batch {batch_idx}")

        loss = super().process_step('train', erased_images, labels)
        return loss

    def validation_step(self, batch, batch_idx):
        images, labels, _ = batch
        loss = super().process_step('val', images, labels)
        return loss
