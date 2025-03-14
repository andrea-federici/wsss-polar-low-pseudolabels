import os
from datetime import datetime

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from train.loggers import create_neptune_logger
import train_config as tc
from models import Xception
from train.components.trainers import create_trainer
from src.models.lightning.adv_er_model import LitModelAdversarialErasing
from attributions.gradcam import GradCAM
from src.data.data_loading import create_data_loaders
from data.image_processing import adversarial_erase
from src.train.helpers.adv_er_helper import load_accumulated_heatmap
from train.components.optimizers import adam
from custom_datasets import ImageFilenameDataset


def run():
    base_heatmaps_dir = 'heatmaps/'
    max_iterations = 10

    for iteration in range(0, max_iterations):
        neptune_logger = create_neptune_logger()

        neptune_logger.experiment["source_files/train_config"] \
            .upload("train_config.py")
        
        neptune_logger.experiment["source_files/train_adversarial_erasing"] \
            .upload("train_adversarial_erasing.py")
        
        neptune_logger.experiment["iteration"] = iteration

        neptune_logger.experiment[f"start_time"] = \
            datetime.now().isoformat() 

        current_heatmaps_dir = os.path.join(
            base_heatmaps_dir, f"iteration_{iteration}"
        )
        os.makedirs(current_heatmaps_dir, exist_ok=True)

        torch_model = Xception()

        train_loader, val_loader, _ = create_data_loaders(
            tc.train_dir(),
            tc.test_dir(),
            tc.batch_size,
            tc.num_workers,
            transform_train=tc.transform_prep,
            dataset_type='adversarial_erase'
        )

        criterion = torch.nn.CrossEntropyLoss()
        optimizer = adam(torch_model, learning_rate=tc.learning_rate)

        lit_model = LitModelAdversarialErasing(
            torch_model, 
            criterion=criterion,
            optimizer=optimizer,
            current_iteration=iteration,
            base_heatmaps_dir=base_heatmaps_dir,
            threshold=0.75,
            fill_color=0
        )

        trainer = create_trainer(
            neptune_logger, 
            checkpoint_filename=f"adv_iteration_{iteration}"
        )
        trainer.fit(lit_model, train_loader, val_loader)

        # Generate new heatmaps for next iteration
        train_val_data = ImageFilenameDataset(
            tc.train_dir(), 
            transform=tc.transform_prep
        )
        generate_and_save_heatmaps(
            lit_model, 
            train_val_data, 
            base_heatmaps_dir,
            iteration,
            current_heatmaps_dir
        )

        neptune_logger.experiment[f"end_time"] \
            = datetime.now().isoformat()
        
        neptune_logger.experiment.stop()


def generate_and_save_heatmaps(model, dataset, base_heatmaps_dir, current_iteration, save_dir, batch_size=32):
    model.eval()
    device = next(model.parameters()).device # Get device from model

    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=tc.num_workers
    )
    os.makedirs(save_dir, exist_ok=True)

    gradcam = GradCAM(model.model, device)

    with torch.no_grad():
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
                            threshold=0.75, # TODO: put this in config
                            fill_color=0 # TODO: put this in config
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