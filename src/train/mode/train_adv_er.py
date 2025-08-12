import os
from datetime import datetime
from typing import List, Tuple, Union

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

import src.attributions.gradcam as gradcam
import src.data.augmentation as aug
from src.data.adversarial_erasing_io import (
    load_accumulated_heatmap,
    load_accumulated_mask,
)
from src.data.custom_datasets import ImageFilenameDataset
from src.data.data_loaders import create_class_dataloader
from src.data.image_processing import (
    erase_region_using_heatmap,
    erase_region_using_mask,
)
from src.post.masks import generate_masks, generate_negative_masks
from src.train.setup import get_train_setup
from src.utils.neptune_utils import NeptuneLogger


def run(cfg: DictConfig) -> None:
    heatmaps_config = cfg.mode.train_config.heatmaps

    data_dir = cfg.data.directory
    train_dir = os.path.join(data_dir, "train")

    base_heatmaps_dir = heatmaps_config.base_directory
    training_size = (
        cfg.transforms.resize_height,
        cfg.transforms.resize_width,
    )  # (H, W)
    threshold = heatmaps_config.threshold

    max_iterations = cfg.mode.train_config.max_iterations

    for iteration in range(0, max_iterations):
        ts = get_train_setup(cfg, iteration=iteration)

        logger = ts.logger
        lightning_model = ts.lightning_model

        logger.experiment["iteration"] = iteration
        logger.experiment[f"start_time"] = datetime.now().isoformat()

        current_heatmaps_dir = os.path.join(base_heatmaps_dir, f"iteration_{iteration}")
        os.makedirs(current_heatmaps_dir, exist_ok=True)

        # Remember that Lightning automatically moves the model to "cuda" if available,
        # and then moves the model back to "cpu" after training, even if the model was
        # on the "cuda" device before calling the fit() function.
        ts.trainer.fit(lightning_model, ts.train_loader, ts.val_loader)

        # Generate new heatmaps for next iteration
        heatmap_load_transform = aug.to_compose(
            aug.AugConfig(
                resize_width=cfg.data.original_width,  # Remember to use original size
                resize_height=cfg.data.original_height,  # Same as above
                mean=cfg.transforms.normalization.mean,
                std=cfg.transforms.normalization.std,
            ),
            "val",
        )
        _generate_and_save_heatmaps(
            lightning_model.model,
            data_dir=train_dir,
            transform=heatmap_load_transform,
            base_heatmaps_dir=base_heatmaps_dir,
            iteration=iteration,
            save_dir=current_heatmaps_dir,
            target_size=training_size,
            super_sizes=heatmaps_config.super_sizes,
            threshold=threshold,
            fill_color=heatmaps_config.fill_color,
            logger=logger,
            device=cfg.hardware.device,
        )

        logger.experiment[f"end_time"] = datetime.now().isoformat()

        logger.experiment.stop()

    # Generate masks
    mask_dir = cfg.mode.masks.save_directory
    remove_background = cfg.mode.masks.remove_background
    negative_dir = os.path.join(train_dir, "neg")

    os.makedirs(mask_dir, exist_ok=True)

    for iteration in range(0, max_iterations):
        ## BINARY ##
        # Visualizable (just for debugging)
        generate_masks(
            base_heatmaps_dir=base_heatmaps_dir,
            mask_dir=f"{mask_dir}/binary/iteration_{iteration}/vis",
            mask_size=training_size,
            threshold=threshold,
            type="binary",
            iteration=iteration,
            remove_background=remove_background,
            vis=True,
        )

        # Non-visualizable (for training)
        generate_masks(
            base_heatmaps_dir=base_heatmaps_dir,
            mask_dir=f"{mask_dir}/binary/iteration_{iteration}/non_vis",
            mask_size=training_size,
            threshold=threshold,
            type="binary",
            iteration=iteration,
            remove_background=remove_background,
            vis=False,
        )

        ## MULTI-CLASS ##
        # Visualizable (just for debugging)
        generate_masks(
            base_heatmaps_dir=base_heatmaps_dir,
            mask_dir=f"{mask_dir}/multiclass/iteration_{iteration}/vis",
            mask_size=training_size,
            threshold=threshold,
            type="multiclass",
            iteration=iteration,
            remove_background=remove_background,
            vis=True,
        )

        # Non-visualizable (for training)
        generate_masks(
            base_heatmaps_dir=base_heatmaps_dir,
            mask_dir=f"{mask_dir}/multiclass/iteration_{iteration}/non_vis",
            mask_size=training_size,
            threshold=threshold,
            type="multiclass",
            iteration=iteration,
            remove_background=remove_background,
            vis=False,
        )

        ## NEGATIVE ##
        # Negative images are only generated for the non-visualizable folder

        # Binary
        generate_negative_masks(
            negative_images_dir=negative_dir,
            mask_dir=f"{mask_dir}/binary/iteration_{iteration}/non_vis",
            mask_size=training_size,
        )

        # Multi-class
        generate_negative_masks(
            negative_images_dir=negative_dir,
            mask_dir=f"{mask_dir}/multiclass/iteration_{iteration}/non_vis",
            mask_size=training_size,
        )


# TODO: while the entire adversarial erasing pipeline is flexible and can be used with
# either heatmap-based or mask-based erasing, the current implementation of this function
# only supports heatmap-based erasing. Mask-based erasing is not implemented yet.
def _generate_and_save_heatmaps(
    model: torch.nn.Module,
    *,
    data_dir: str,
    transform: transforms.Compose,
    base_heatmaps_dir: str,
    iteration: int,
    save_dir: str,
    target_size: Tuple[int, int],  # (H, W)
    super_sizes: List[int],
    threshold: float,
    fill_color: Union[int, float],
    logger: NeptuneLogger,
    target_class: int = 1,
    batch_size: int = 32,
    num_workers: int = 4,
    device: str = "cpu",
):
    was_training = model.training
    model.eval()
    model.zero_grad()

    model = model.to(device)

    try:
        dataloader = create_class_dataloader(
            data_dir=data_dir,
            transform=transform,
            target_class=target_class,
            batch_size=batch_size,
            num_workers=num_workers,
            dataset_type="adversarial_erasing",
            shuffle=False,
        )
        os.makedirs(save_dir, exist_ok=True)

        count = 0
        neg_count = 0

        with torch.no_grad():
            # Loop through batches
            for batch_idx, (images, labels, img_paths) in enumerate(
                tqdm(dataloader, desc="Generating Heatmaps")
            ):
                images = images.to(device)

                # Loop through images in batch
                for i, (image, label, img_path) in enumerate(
                    zip(images, labels, img_paths)
                ):
                    assert label == 1, "Label should be 1 here."
                    count += 1
                    img = image.unsqueeze(0)

                    if iteration > 0:
                        accumulated_heatmap = load_accumulated_heatmap(
                            base_heatmaps_dir,
                            img_path,
                            label,
                            iteration - 1,
                        )

                        img = erase_region_using_heatmap(
                            img, accumulated_heatmap, threshold, fill_color
                        )

                    heatmap = gradcam.generate_super_heatmap(
                        model,
                        img,
                        target_size=target_size,
                        sizes=super_sizes,
                        target_class=1,
                    )

                    img_overlay = gradcam.overlay_heatmap(img, heatmap)

                    pred = torch.argmax(model(img)).item()

                    # Log heatmap and image overlay to Neptune, just for batch 0
                    if batch_idx == 0:
                        logger.log_tensor_img(
                            img_overlay, name=f"heatmap_{img_path}_pred_{pred}"
                        )

                    # Resize image to training size in order to do inference
                    img = F.interpolate(
                        img, size=target_size, mode="bilinear", align_corners=False
                    )

                    # Generate transparent heatmap if prediction is negative
                    if pred == 0:
                        print(f"Iteration: {iteration}. Negative: {img_path}")
                        neg_count += 1
                        heatmap = torch.zeros_like(heatmap)

                    # Save heatmap
                    img_name = os.path.splitext(os.path.basename(img_path))[0]
                    _save_heatmap(heatmap, save_dir, img_name, pred)

        print(f"Count: {count}, Negative Count: {neg_count}")

    finally:
        if was_training:
            model.train()


def _save_heatmap(
    heatmap: torch.Tensor, save_dir: str, img_name: str, pred: int
) -> None:
    """
    Save the heatmap to the specified directory in both .pt and .png formats.
    The .pt file contains the tensor used by the pipeline, while the .png file is just
    for debugging purposes.

    Args:
        heatmap (torch.Tensor): The heatmap tensor to save. The heatmap is expected to
            be of shape (H, W), normalized in the range [0, 1].
        save_dir (str): Directory where the heatmap will be saved.
        img_name (str): Name of the image to which the heatmap corresponds.
        pred (int): The predicted class for the image. It will be appended to the .png
            filename for debugging purposes.

    Returns:
        None
    """
    assert heatmap.dim() == 2, "Heatmap must be a 2D tensor."
    assert (
        heatmap.min() >= 0 and heatmap.max() <= 1
    ), "Heatmap tensor must be normalized in the range [0, 1]."
    assert os.path.isdir(save_dir), f"Save directory {save_dir} does not exist."

    # Ensure the save directory exists
    os.makedirs(save_dir, exist_ok=True)

    # Save .pt file
    heatmap_filename_pt = os.path.join(save_dir, f"{img_name}.pt")
    torch.save(heatmap, heatmap_filename_pt)

    # Save .png file, just for debugging purposes
    heatmap_filename_png = os.path.join(save_dir, f"{img_name}_pred_{pred}.png")
    heatmap_np = heatmap.detach().cpu().numpy()
    plt.imsave(heatmap_filename_png, heatmap_np, cmap="jet")
