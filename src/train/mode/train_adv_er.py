import glob
import os
import warnings
from datetime import datetime
from typing import List, Optional, Tuple, Union

import cv2
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from torchvision import transforms
from tqdm import tqdm

import src.attributions.cam as cam
import src.attributions.gradcam as gradcam
import src.data.augmentation as aug
from src.data.adversarial_erasing_io import (
    _area_targeted_envelope,
    load_accumulated_heatmap,
    load_tensor,
)
from src.data.data_loaders import create_class_dataloader
from src.data.image_processing import erase_region_using_heatmap
from src.post.masks import generate_masks, generate_negative_masks
from src.train.setup import get_train_setup
from src.utils.neptune_utils import NeptuneLogger


def run(cfg: DictConfig) -> None:
    heatmaps_config = cfg.mode.train_config.heatmaps

    data_dir = cfg.data.directory
    dataset_type = cfg.mode.dataset_type
    train_dir = (
        data_dir if dataset_type == "pascal_voc" else os.path.join(data_dir, "train")
    )
    splits = ["train"]
    val_dir_path = os.path.join(data_dir, "val")
    if dataset_type == "pascal_voc" or os.path.isdir(val_dir_path):
        splits.append("val")
    else:
        warnings.warn(
            f"Validation directory '{val_dir_path}' not found. Skipping 'val' split."
        )

    base_heatmaps_dir = heatmaps_config.base_directory
    training_size = (
        cfg.data.transforms.resize_height,
        cfg.data.transforms.resize_width,
    )  # (H, W)
    threshold = heatmaps_config.threshold

    envelope_cfg = heatmaps_config.get("area_envelope", {})
    envelope_start = envelope_cfg.get("start_iteration", 2)
    envelope_scale = envelope_cfg.get("scale", 0.1)

    max_iterations = cfg.mode.train_config.max_iterations

    for iteration in range(0, max_iterations):
        ts = get_train_setup(cfg, iteration=iteration)

        logger = ts.logger
        lightning_model = ts.lightning_model

        logger.experiment["iteration"] = iteration
        logger.experiment["start_time"] = datetime.now().isoformat()

        # Remember that Lightning automatically moves the model to "cuda" if available,
        # and then moves the model back to "cpu" after training, even if the model was
        # on the "cuda" device before calling the fit() function.
        ts.trainer.fit(lightning_model, ts.train_loader, ts.val_loader)

        # Generate new heatmaps for next iteration
        heatmap_load_transform = aug.to_compose(
            aug.AugConfig(
                resize_width=cfg.data.original_width,  # Remember to use original size
                resize_height=cfg.data.original_height,  # Same as above
                mean=cfg.data.transforms.normalization.mean,
                std=cfg.data.transforms.normalization.std,
            ),
            "val",
        )
        class_range = (
            range(cfg.training.num_classes) if dataset_type == "pascal_voc" else [1]
        )
        current_heatmaps_dir = os.path.join(base_heatmaps_dir, f"iteration_{iteration}")
        os.makedirs(current_heatmaps_dir, exist_ok=True)
        for split in splits:
            split_data_dir = (
                data_dir
                if dataset_type == "pascal_voc"
                else os.path.join(data_dir, split)
            )
            for cls in class_range:
                _generate_and_save_heatmaps(
                    lightning_model.model,
                    data_dir=split_data_dir,
                    transform=heatmap_load_transform,
                    base_heatmaps_dir=base_heatmaps_dir,
                    iteration=iteration,
                    envelope_start=envelope_start,
                    envelope_scale=envelope_scale,
                    save_dir=current_heatmaps_dir,
                    target_size=training_size,
                    super_sizes=heatmaps_config.get("super_sizes", []),
                    attribution=heatmaps_config.attribution,
                    threshold=threshold,
                    fill_color=heatmaps_config.fill_color,
                    logger=logger,
                    device=cfg.hardware.device,
                    target_class=cls,
                    dataset_type=dataset_type,
                    split=split,
                )

        if iteration > 0:
            sample_paths = sorted(glob.glob(os.path.join(train_dir, "pos", "*.png")))[
                :10
            ]
            debug_dir = os.path.join(
                "out", "debug", "envelopes", f"iteration_{iteration}"
            )
            _save_envelope_debug_images(
                base_heatmaps_dir=base_heatmaps_dir,
                img_paths=sample_paths,
                iteration=iteration,
                threshold=threshold,
                target_size=training_size,
                envelope_start=envelope_start,
                envelope_scale=envelope_scale,
                save_dir=debug_dir,
            )

        logger.experiment["end_time"] = datetime.now().isoformat()

        logger.experiment.stop()

    # Generate masks
    mask_dir = cfg.mode.masks.save_directory
    remove_background = cfg.mode.masks.remove_background

    area_cfg = cfg.mode.train_config.heatmaps.get("area_envelope", {})
    envelope_start = area_cfg.get("start_iteration", 2)
    envelope_scale = area_cfg.get("scale", 0.1)

    os.makedirs(mask_dir, exist_ok=True)
    negative_dir = (
        os.path.join(data_dir, "train", "neg") if dataset_type != "pascal_voc" else None
    )

    for iteration in range(0, max_iterations):
        ## BINARY ##
        # Visualizable (just for debugging)
        generate_masks(
            base_heatmaps_dir=base_heatmaps_dir,
            mask_dir=f"{mask_dir}/binary/iteration_{iteration}/vis",
            mask_size=training_size,
            threshold=threshold,
            type="binary" if dataset_type != "pascal_voc" else "voc",
            iteration=iteration,
            remove_background=remove_background,
            vis=True,
            envelope_start=envelope_start,
            envelope_scale=envelope_scale,
        )

        # Non-visualizable (for training)
        generate_masks(
            base_heatmaps_dir=base_heatmaps_dir,
            mask_dir=f"{mask_dir}/binary/iteration_{iteration}/non_vis",
            mask_size=training_size,
            threshold=threshold,
            type="binary" if dataset_type != "pascal_voc" else "voc",
            iteration=iteration,
            remove_background=remove_background,
            vis=False,
            envelope_start=envelope_start,
            envelope_scale=envelope_scale,
        )

        if dataset_type != "pascal_voc":
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
                envelope_start=envelope_start,
                envelope_scale=envelope_scale,
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
                envelope_start=envelope_start,
                envelope_scale=envelope_scale,
            )

            ## NEGATIVE ##
            # Negative images are only generated for the non-visualizable folder
            if negative_dir and os.path.isdir(negative_dir):
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
    envelope_start: int,
    envelope_scale: float,
    save_dir: str,
    target_size: Tuple[int, int],  # (H, W)
    super_sizes: List[int],
    threshold: float,
    fill_color: Union[int, float],
    logger: NeptuneLogger,
    attribution: str = "gradcam",
    target_class: int = 1,
    batch_size: int = 32,
    num_workers: int = 4,
    device: str = "cpu",
    dataset_type: str = "adversarial_erasing",
    split: str = "train",
):
    was_training = model.training
    model.eval()
    model.zero_grad()

    model = model.to(device)

    try:
        # TODO: here I am passing split=split even for datasets that don't support it. Examine
        # whether it causes problems.
        dataloader = create_class_dataloader(
            data_dir=data_dir,
            transform=transform,
            target_class=target_class,
            batch_size=batch_size,
            num_workers=num_workers,
            dataset_type=dataset_type,
            shuffle=False,
            split=split,
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
                for img_idx, (image, label, img_path) in enumerate(
                    zip(images, labels, img_paths)
                ):
                    if dataset_type == "pascal_voc":
                        assert label[target_class] == 1, "Image missing target class"
                        lbl_val = 1
                    else:
                        assert label == 1, "Label should be 1 here."
                        lbl_val = label
                    count += 1
                    img = image.unsqueeze(0)

                    if iteration > 0:
                        accumulated_heatmap = load_accumulated_heatmap(
                            base_heatmaps_dir,
                            img_path,
                            lbl_val,
                            iteration - 1,
                            threshold=threshold,
                            envelope_start=envelope_start,
                            envelope_scale=envelope_scale,
                            target_class=target_class,
                        )

                        img = erase_region_using_heatmap(
                            img, accumulated_heatmap, threshold, fill_color
                        )

                    if attribution.lower() == "gradcam":
                        heatmap, intermediates = gradcam.generate_super_heatmap(
                            model,
                            img,
                            target_size=target_size,
                            sizes=super_sizes,
                            target_class=target_class,
                            return_intermediates=True,
                        )
                    elif attribution.lower() == "cam":
                        heatmap = cam.generate_heatmap(
                            model, img, target_class=target_class
                        )
                        intermediates = {}
                    else:
                        raise ValueError(
                            f"Unsupported attribution method: {attribution}"
                        )

                    img_overlay = gradcam.overlay_heatmap(img, heatmap)

                    img_name = os.path.splitext(os.path.basename(img_path))[0]

                    # Resize image to training size in order to do inference
                    resized_img = F.interpolate(
                        img, size=target_size, mode="bilinear", align_corners=False
                    )

                    # Do inference
                    logits = model(resized_img)
                    probs = torch.sigmoid(logits)
                    pos_prob = probs[0, target_class].item()
                    pred = int(pos_prob >= 0.2)  # TODO: avoid hard-coding

                    # Calculate necessity drop
                    next_img = erase_region_using_heatmap(
                        resized_img, heatmap, threshold=threshold, fill_color=fill_color
                    )
                    next_logits = model(next_img)[0]
                    next_pos_logit = next_logits[target_class].item()
                    drop = logits[0][target_class].item() - next_pos_logit

                    suffix = f"{pred}_{pos_prob:.3f}_{drop:.3f}"

                    # Log heatmap and image overlay to Neptune, just for batch 0
                    if batch_idx == 0 and img_idx < 5:
                        logger.log_tensor_img(
                            img_overlay,
                            name=f"heatmap_cls{target_class}_{img_name}_{suffix}",
                        )
                        for s in sorted(intermediates.keys()):
                            hm = intermediates[s].unsqueeze(0).unsqueeze(0).cpu()
                            logger.log_tensor_img(
                                hm,
                                name=f"intermediates/heatmap_cls{target_class}_{img_name}_{s}_{suffix}",
                            )

                    # Generate transparent heatmap if prediction is negative
                    if pred == 0:
                        print(
                            f"Iteration: {iteration}. Negative: {img_path} class {target_class}"
                        )
                        neg_count += 1
                        heatmap = torch.zeros_like(heatmap)

                    # Save heatmap
                    save_png = (
                        True if img_idx == 0 else False
                    )  # Only save for first image in batch
                    _save_heatmap(
                        heatmap,
                        save_dir,
                        img_name,
                        pred,
                        target_class=target_class,
                        save_png=save_png,
                    )

        # TODO: I could group the metrics in a subfolder
        correct_count = count - neg_count
        accuracy_pct = 100.0 * correct_count / float(count)
        logger.experiment["Count"] = count
        logger.experiment["Negative Count"] = neg_count
        logger.experiment["Accuracy"] = accuracy_pct
        print(f"Count: {count}, Negative Count: {neg_count}, Accuracy: {accuracy_pct}")

    finally:
        if was_training:
            model.train()


def _save_heatmap(
    heatmap: torch.Tensor,
    save_dir: str,
    img_name: str,
    pred: int,
    *,
    target_class: Optional[int] = None,
    save_png: bool = False,
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
    assert heatmap.min() >= 0 and heatmap.max() <= 1, (
        "Heatmap tensor must be normalized in the range [0, 1]."
    )
    assert os.path.isdir(save_dir), f"Save directory {save_dir} does not exist."

    # Ensure the save directory exists
    os.makedirs(save_dir, exist_ok=True)

    # Adjust filename for class-aware datasets
    base_name = (
        f"{img_name}_cls{target_class}" if target_class is not None else img_name
    )

    # Save .pt file
    heatmap_filename_pt = os.path.join(save_dir, f"{base_name}.pt")
    torch.save(heatmap, heatmap_filename_pt)

    if save_png:
        # Save .png file, just for debugging purposes
        heatmap_filename_png = os.path.join(save_dir, f"{base_name}_pred_{pred}.png")
        heatmap_np = heatmap.detach().cpu().numpy()
        plt.imsave(heatmap_filename_png, heatmap_np, cmap="jet")


def _save_envelope_debug_images(
    *,
    base_heatmaps_dir: str,
    img_paths: List[str],
    iteration: int,
    threshold: float,
    target_size: Tuple[int, int],
    envelope_start: int,
    envelope_scale: float,
    save_dir: str,
) -> None:
    """Save composite plots illustrating mask accumulation and envelope constraints.

    Each saved figure contains one subplot per iteration up to ``iteration-1``.
    For subplot ``t``, the image is overlaid with the cumulative mask through
    iteration ``t`` (inclusive), the raw heatmap from iteration ``t+1``, and the
    border of the area-targeted envelope computed from the current mask.

    Args:
        base_heatmaps_dir: Directory containing per-iteration heatmaps.
        img_paths: List of image file paths to visualize.
        iteration: Current iteration (the latest heatmaps correspond to this index).
        threshold: Threshold used to binarize accumulated heatmaps.
        target_size: Size ``(H, W)`` to which images are resized for visualization.
        envelope_start: Iteration at which the envelope constraint begins.
        envelope_scale: Scale factor controlling envelope dilation.
        save_dir: Directory where the debug figures will be saved.
    """

    os.makedirs(save_dir, exist_ok=True)

    for img_path in img_paths:
        img_name = os.path.splitext(os.path.basename(img_path))[0]
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (target_size[1], target_size[0]))

        fig, axes = plt.subplots(1, iteration, figsize=(5 * iteration, 5))
        if iteration == 1:
            axes = [axes]

        for t in range(iteration):
            cumulative = load_accumulated_heatmap(
                base_heatmaps_dir,
                img_name,
                1,
                t,
                threshold=threshold,
                envelope_start=envelope_start,
                envelope_scale=envelope_scale,
            )
            mask = cumulative > threshold

            next_heatmap = load_tensor(base_heatmaps_dir, t + 1, img_name)

            ax = axes[t]
            ax.imshow(img)
            ax.imshow(mask.cpu(), cmap="Reds", alpha=0.4)
            ax.imshow(next_heatmap.cpu(), cmap="jet", alpha=0.4)
            if t >= envelope_start:
                envelope = _area_targeted_envelope(mask, envelope_scale)
                ax.contour(envelope.cpu(), levels=[0.5], colors="yellow", linewidths=1)
            ax.set_title(f"iter {t} -> {t + 1}")
            ax.axis("off")

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"{img_name}.png"))
        plt.close(fig)
