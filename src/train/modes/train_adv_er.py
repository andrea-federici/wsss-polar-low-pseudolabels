import os
from datetime import datetime

import torch
from torch.utils.data import DataLoader
from omegaconf import DictConfig
from tqdm import tqdm

from src.attributions.gradcam import GradCAM
from src.data.image_processing import adversarial_erase
from src.train.setups import get_train_setup
from src.train.helpers.adv_er_helper import load_accumulated_heatmap
from src.data.custom_datasets import ImageFilenameDataset
from src.data.transforms import get_transform


def run(cfg: DictConfig) -> None:
    for iteration in range(0, cfg.mode.max_iterations):
        ts = get_train_setup(cfg, iteration=iteration)

        logger = ts.logger
        lightning_model = ts.lightning_model

        logger.experiment["source_files/train_config"] \
            .upload("train_config.py")
        
        logger.experiment["source_files/train_adversarial_erasing"] \
            .upload("train_adversarial_erasing.py")
        
        logger.experiment["iteration"] = iteration

        logger.experiment[f"start_time"] = \
            datetime.now().isoformat() 

        current_heatmaps_dir = os.path.join(
            cfg.mode.heatmaps.base_directory, f"iteration_{iteration}"
        )
        os.makedirs(current_heatmaps_dir, exist_ok=True)

        ts.trainer.fit(lightning_model, ts.train_loader, ts.val_loader)

        # Generate new heatmaps for next iteration
        train_val_data = ImageFilenameDataset(
            os.path.join(cfg.data_dir, "train"),
            transform=get_transform(cfg, "val")
        )
        generate_and_save_heatmaps(
            lightning_model, 
            train_val_data, 
            cfg.mode.heatmaps.directory,
            iteration,
            current_heatmaps_dir
        )

        logger.experiment[f"end_time"] \
            = datetime.now().isoformat()
        
        logger.experiment.stop()


def generate_and_save_heatmaps(
    model, 
    dataset, 
    base_heatmaps_dir, 
    current_iteration, 
    save_dir,
    num_workers,
    threshold,
    fill_color
):
    model.eval()
    device = next(model.parameters()).device # Get device from model

    dataloader = DataLoader(
        dataset, 
        batch_size=32, 
        shuffle=False, 
        num_workers=num_workers
    )
    os.makedirs(save_dir, exist_ok=True)

    gradcam = GradCAM(model.model, device)

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
                    img = image.unsqueeze(0)

                    if current_iteration > 0:
                        accumulated_heatmap = load_accumulated_heatmap(
                            base_heatmaps_dir,
                            img_path,
                            label,
                            current_iteration
                        )

                        img = adversarial_erase(
                            img.unsqueeze(0), 
                            accumulated_heatmap,
                            threshold,
                            fill_color
                        ).squeeze(0)

                    heatmap = gradcam.generate_heatmap(
                        img, 
                        target_class=1
                    )

                    # Save heatmap as a .pt file
                    heatmap_filename = os.path.join(
                        save_dir, 
                        os.path.basename(img_path) + ".pt"
                    )
                    torch.save(heatmap, heatmap_filename)