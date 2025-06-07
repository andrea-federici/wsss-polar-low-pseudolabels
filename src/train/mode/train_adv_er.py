import os
from datetime import datetime

import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from tqdm import tqdm

import src.data.augmentation as aug
from adv_perturb import captum_hurricane_occlusion
from src.attributions.gradcam import GradCAM
from src.data.adversarial_erasing_io import (
    load_accumulated_heatmap,
    load_accumulated_mask,
)
from src.data.custom_datasets import ImageFilenameDataset
from src.data.image_processing import (
    erase_region_using_heatmap,
    erase_region_using_mask,
)
from src.train.setup import get_train_setup
from src.utils.neptune_utils import NeptuneLogger


def run(cfg: DictConfig) -> None:
    # Load generate_and_save function using a getter, given name from config

    for iteration in range(0, cfg.mode.train_config.max_iterations):
        ts = get_train_setup(cfg, iteration=iteration)

        logger = ts.logger
        lightning_model = ts.lightning_model

        logger.experiment["iteration"] = iteration
        logger.experiment[f"start_time"] = datetime.now().isoformat()

        heatmaps_config = cfg.mode.train_config.heatmaps

        current_heatmaps_dir = os.path.join(
            heatmaps_config.base_directory, f"iteration_{iteration}"
        )
        os.makedirs(current_heatmaps_dir, exist_ok=True)

        ts.trainer.fit(lightning_model, ts.train_loader, ts.val_loader)

        aug_heatmap_gen = aug.to_aug_config(cfg.transforms)
        aug_heatmap_gen.resize_width = 800
        aug_heatmap_gen.resize_height = 800
        # Generate new heatmaps for next iteration
        train_val_data = ImageFilenameDataset(
            os.path.join(cfg.data_dir, "train"),
            transform=aug.to_compose(aug_heatmap_gen, "val"),
        )
        generate_and_save_heatmaps(
            lightning_model,
            train_val_data,
            heatmaps_config.base_directory,
            iteration,
            current_heatmaps_dir,
            cfg.num_workers,
            heatmaps_config.threshold,
            heatmaps_config.fill_color,
            logger=logger,
        )

        logger.experiment[f"end_time"] = datetime.now().isoformat()

        logger.experiment.stop()


def generate_and_save_heatmaps(
    model,
    dataset,
    base_heatmaps_dir,
    current_iteration,
    save_dir,
    num_workers,
    threshold,
    fill_color,
    logger: NeptuneLogger,
):
    model.eval()
    device = next(model.parameters()).device  # Get device from model

    # TODO: can we use the create_data_loaders function with the only_positives flag?
    dataloader = DataLoader(
        dataset, batch_size=32, shuffle=False, num_workers=num_workers
    )
    os.makedirs(save_dir, exist_ok=True)

    gradcam = GradCAM(model.model, device)

    total_count = 0
    neg_count = 0

    with torch.no_grad():
        # TODO: remove enumerate and batch_idx
        for batch_idx, (images, labels, img_paths) in enumerate(
            tqdm(dataloader, desc="Generating Heatmaps")
        ):
            images = images.to(device)

            for i, (image, label, img_path) in enumerate(
                zip(images, labels, img_paths)
            ):
                if label.item() == 1:
                    total_count += 1
                    img = image.unsqueeze(0)

                    if current_iteration > 0:
                        accumulated_heatmap = load_accumulated_heatmap(
                            base_heatmaps_dir, img_path, label, current_iteration - 1
                        )

                        img = erase_region_using_heatmap(
                            img, accumulated_heatmap, threshold, fill_color
                        )

                        # accumulated_mask = load_accumulated_mask(
                        #     base_masks_dir=base_heatmaps_dir,
                        #     img_name=img_path,
                        #     label=label,
                        #     iteration=current_iteration - 1,
                        # )

                        # img = erase_region_using_mask(
                        #     img,
                        #     accumulated_mask,
                        #     fill_color=fill_color,
                        # )

                    # heatmap = gradcam.generate_heatmap(img, target_class=1)
                    heatmap = gradcam.generate_super_heatmap(
                        img,
                        target_size=(512, 512),
                        sizes=[512, 512 * 2, 512 * 3],
                        target_class=1,
                    )
                    # mask = captum_hurricane_occlusion(
                    #     x=img,
                    #     classifier=model.model,
                    #     hurricane_class=1,
                    #     patch_size=128,
                    #     stride=32,
                    #     top_k=32 * 32 * 8,
                    #     # threshold=0.85,
                    #     device="cuda",
                    # )

                    img_overlay = gradcam.overlay_heatmap(img, heatmap)

                    # TODO: make this better
                    if batch_idx == 40 or batch_idx == 41:
                        logger.log_tensor_img(img_overlay, name=f"heatmap_{img_path}")

                    # resize image to training size in order to do inference
                    img = F.interpolate(
                        img, size=(512, 512), mode="bilinear", align_corners=False
                    )

                    # do inference and if pred is negative generate transparent heatmap
                    pred = torch.argmax(model(img)).item()
                    if pred == 0:
                        print(f"Iteration: {current_iteration}. Negative: {img_path}")
                        neg_count += 1
                        heatmap = torch.zeros_like(heatmap)

                    # Save heatmap as a .pt file
                    heatmap_filename = os.path.join(
                        save_dir,
                        os.path.splitext(os.path.basename(img_path))[0] + ".pt",
                    )
                    torch.save(heatmap, heatmap_filename)

    print(f"Negative count: {neg_count}. Total: {total_count}")
