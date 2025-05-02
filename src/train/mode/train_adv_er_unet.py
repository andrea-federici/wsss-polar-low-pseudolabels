import os
from datetime import datetime

import torch
from torch.utils.data import DataLoader
from omegaconf import DictConfig
from tqdm import tqdm

from src.train.setup import get_train_setup
from src.train.helper import adversarial_erase, load_accumulated_heatmap
from src.data.custom_datasets import ImageFilenameDataset
from src.data.transforms import get_transform


def run(cfg: DictConfig) -> None:
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

        # Generate new heatmaps for next iteration
        train_val_data = ImageFilenameDataset(
            os.path.join(cfg.data_dir, "train"),
            transform=get_transform(cfg.transforms, "val"),
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
        )

        logger.experiment[f"end_time"] = datetime.now().isoformat()

        logger.experiment.stop()


def generate_and_save_heatmaps(
    lit_classifier,
    dataset,
    base_heatmaps_dir,
    current_iteration,
    save_dir,
    num_workers,
    threshold,
    fill_color,
):
    classifier = lit_classifier.model

    # load weights / checkpoint into classifier ...
    classifier.eval()
    for p in classifier.parameters():
        p.requires_grad = False

    image_size = (512, 512)

    mean = [0.2872, 0.2872, 0.4595]
    std = [0.1806, 0.1806, 0.2621]

    # dataset + dataloader
    from torchvision import transforms

    transform_list = [
        transforms.RandomAffine(
            degrees=30,
            translate=(0.3, 0.3),
            scale=(0.9, 1.1),
            fill=0,
        ),
        transforms.Resize(image_size),
    ]

    # Random flips
    transform_list.append(transforms.RandomHorizontalFlip(p=0.5))

    transform_list.append(transforms.RandomVerticalFlip(p=0.5))

    transform_list.extend(
        [transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)]
    )

    transform_train = transforms.Compose(transform_list)

    transform_val = transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )

    from src.data.data_loading import create_data_loaders

    train_loader, val_loader, _ = create_data_loaders(
        "data",
        batch_size=4,
        num_workers=8,
        transform_train=transform_train,
        transform_val=transform_val,
        only_positives=True,
    )

    # init LightningModule
    from src.models.lightning.mask_head import MaskerLightning

    mask_trainer = MaskerLightning(classifier=classifier, mask_threshold=0.5)

    # Callbacks
    # Early stopping
    from pytorch_lightning.callbacks import EarlyStopping

    early_stop = EarlyStopping(
        monitor="val/loss",
        patience=5,
        verbose=True,
        mode="min",
    )

    # Learning rate monitor
    from pytorch_lightning.callbacks import LearningRateMonitor

    lr_monitor = LearningRateMonitor(logging_interval="step")

    callbacks = [early_stop, lr_monitor]
    callbacks = [lr_monitor]

    # fit
    from pytorch_lightning import Trainer

    trainer = Trainer(
        max_epochs=20,
        accelerator="gpu",
        devices=1,
        callbacks=callbacks,
        enable_checkpointing=True,
        log_every_n_steps=1,
        enable_progress_bar=True,
    )
    trainer.fit(mask_trainer, train_loader, val_loader)

    classifier.eval()
    device = next(classifier.parameters()).device  # Get device from model

    dataloader = DataLoader(
        dataset, batch_size=32, shuffle=False, num_workers=num_workers
    )
    print(f"Num batches: {len(dataloader)}")
    print(f"Num samples: {len(dataset)}")
    os.makedirs(save_dir, exist_ok=True)

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
                            base_heatmaps_dir, img_path, label, current_iteration - 1
                        )

                        img = adversarial_erase(
                            img, accumulated_heatmap, threshold, fill_color
                        )

                    heatmap = mask_trainer(img)

                    heatmap = heatmap.squeeze(0).squeeze(0)
                    # print(f"Heatmap shape: {heatmap.shape}")

                    heatmap_erase = 1 - heatmap

                    # Save heatmap as a .pt file
                    heatmap_filename = os.path.join(
                        save_dir,
                        os.path.splitext(os.path.basename(img_path))[0] + ".pt",
                    )
                    torch.save(heatmap_erase, heatmap_filename)
