import csv
import glob
import math
import os
from datetime import datetime
from itertools import chain
from typing import List, Optional, Tuple, cast

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from lightning.pytorch.callbacks import ModelCheckpoint
from omegaconf import DictConfig
from pytorch_grad_cam.metrics.cam_mult_image import \
    CamMultImageConfidenceChange
from pytorch_grad_cam.metrics.road import (ROADLeastRelevantFirst,
                                           ROADMostRelevantFirst)
from pytorch_grad_cam.utils.model_targets import ClassifierOutputSoftmaxTarget
from torchvision import transforms
from tqdm import tqdm

import src.attributions.gradcam as gradcam
import src.data.augmentation as aug
from src.data.adversarial_erasing_io import (_area_targeted_envelope,
                                             load_accumulated_heatmap,
                                             load_tensor)
from src.data.data_loaders import create_class_dataloader
from src.data.image_processing import (convert_to_np_array,
                                       erase_region_using_heatmap)
from src.post.masks import generate_masks, generate_negative_masks
from src.train.setup import setup_training
from src.utils.misc import resize_heatmap
from src.utils.neptune_utils import NeptuneLogger

# default precision if a field isn't listed below
_DEFAULT_DECIMALS = 4

# per-field precision (tweak as you like)
_FLOAT_PRECISION = {
    "pos_prob": 4,
    "keep_prob": 4,
    "MRF": 4, "LRF": 4, "Combined": 4,
    "rel_MRF": 4, "rel_LRF": 4,
    "logprob_pos": 4,
    "margin": 3,
    "cam_area_ratio": 4, "cam_largest_comp_ratio": 4, "cam_entropy": 4,
    "part_ratio": 4, "gini": 4, "rg_norm": 4, "compactness": 4, "hull_fill": 4,
}

def _format_row_for_csv(row: dict) -> dict:
    out = {}
    for k, v in row.items():
        if isinstance(v, (float, np.floating)):
            d = _FLOAT_PRECISION.get(k, _DEFAULT_DECIMALS)
            fv = round(float(v), d)
            # kill negative zero
            if fv == 0.0:
                fv = 0.0
            out[k] = fv
        else:
            out[k] = v
    return out


import math

import cv2
import numpy as np


def participation_ratio(cam: np.ndarray) -> float:
    """Intensity-based 'effective area' in [1/(HW), 1]; higher = more spread."""
    x = cam.astype(np.float64)
    x = x / (x.sum() + 1e-12)
    return float((x.sum()**2) / (np.square(x).sum() + 1e-12))

def gini_intensity(cam: np.ndarray) -> float:
    """Inequality of intensities in [0,1]; higher = more concentrated (less spread)."""
    x = cam.ravel().astype(np.float64)
    x = x[x>0]
    if x.size == 0: return 0.0
    x.sort()
    n = x.size
    cum = np.cumsum(x)
    return float((n + 1 - 2*(cum.sum()/cum[-1])) / n)

def weighted_radius_of_gyration(cam: np.ndarray) -> float:
    """Std distance to the intensity-weighted centroid, normalized by image diag."""
    H, W = cam.shape
    y, x = np.mgrid[0:H, 0:W]
    w = cam.astype(np.float64)
    w_sum = w.sum() + 1e-12
    cx = (w * x).sum() / w_sum
    cy = (w * y).sum() / w_sum
    r2 = (((x - cx)**2 + (y - cy)**2) * w).sum() / w_sum
    r = math.sqrt(r2)
    diag = math.sqrt(H*H + W*W)
    return float(r / (diag + 1e-12))  # in [0,~0.5]

def shape_compactness(mask: np.ndarray) -> float:
    """4πA/P^2 in (0,1]; circle=1. Needs a binary mask."""
    mask_u8 = (mask>0).astype(np.uint8)
    A = float(mask_u8.sum())
    if A == 0: return 0.0
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    P = sum(cv2.arcLength(c, True) for c in contours)
    if P == 0: return 0.0
    return float(4*math.pi*A/(P*P))

def hull_fill_ratio(mask: np.ndarray) -> float:
    """Area / convex hull area in (0,1]; closer to 1 = less concave."""
    mask_u8 = (mask>0).astype(np.uint8)
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours: return 0.0
    A = float(mask_u8.sum())
    hull = cv2.convexHull(np.vstack(contours))
    hullA = float(cv2.contourArea(hull))
    if hullA <= 0: return 0.0
    return float(A / hullA)


def _cam_entropy(cam_np: np.ndarray) -> float:
    """Shannon entropy of the CAM distribution, normalized to [0,1]."""
    p = cam_np.astype(np.float64)
    p = p / (p.sum() + 1e-12)
    ent = -(p * np.log(p + 1e-12)).sum()
    return float(ent / math.log(p.size + 1e-12))

def _mask_stats(cam_np: np.ndarray, percentile: int):
    """Stats for the top-p% CAM region."""
    thr = np.quantile(cam_np, percentile / 100.0)
    mask = (cam_np >= thr).astype(np.uint8)
    area_ratio = float(mask.mean())
    num_labels, labels = cv2.connectedComponents(mask, connectivity=8)
    num_components = int(max(0, num_labels - 1))
    if num_components > 0:
        sizes = np.bincount(labels.ravel())[1:]
        largest_comp_ratio = float(sizes.max() / sizes.sum())
    else:
        largest_comp_ratio = 0.0
    return area_ratio, num_components, largest_comp_ratio, mask.astype(np.float32)

def _append_row(csv_path: str, row: dict):
    row = _format_row_for_csv(row)
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    write_header = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            w.writeheader()
        w.writerow(row)


def run(cfg: DictConfig) -> None:
    heatmaps_config = cfg.mode.train_config.heatmaps

    data_dir = cfg.data.directory
    train_dir = os.path.join(data_dir, "train")

    base_heatmaps_dir = heatmaps_config.base_directory
    training_size = (
        cfg.data.transforms.resize_height,
        cfg.data.transforms.resize_width,
    )  # (H, W)
    threshold = heatmaps_config.threshold
    fill_color = heatmaps_config.fill_color

    envelope_cfg = heatmaps_config.get("area_envelope", {})
    envelope_start = envelope_cfg.get("start_iteration", 2)
    envelope_scale = envelope_cfg.get("scale", 0.1)

    max_iterations = cfg.mode.train_config.max_iterations

    if not cfg.get("skip_training", False):
        for iteration in range(0, max_iterations):
            train_setup = setup_training(cfg, iteration=iteration)

            logger = train_setup.logger
            lightning_model = train_setup.lightning_model

            logger.experiment["iteration"] = iteration
            logger.experiment["start_time"] = datetime.now().isoformat()

            current_heatmaps_dir = os.path.join(base_heatmaps_dir, f"iteration_{iteration}")
            os.makedirs(current_heatmaps_dir, exist_ok=True)

            # Remember that Lightning automatically moves the model to "cuda" if available,
            # and then moves the model back to "cpu" after training, even if the model was
            # on the "cuda" device before calling the fit() function.
            trainer = train_setup.trainer
            trainer.fit(lightning_model, train_setup.train_loader, train_setup.val_loader)

            # # Load best model checkpoint
            assert trainer.checkpoint_callback is not None, "Checkpoint callback not configured."
            ckpt_cb = cast(ModelCheckpoint, trainer.checkpoint_callback)
            best_ckpt_path = ckpt_cb.best_model_path
            assert best_ckpt_path and os.path.isfile(best_ckpt_path), f"Missing: {best_ckpt_path}"
            best_lit = type(lightning_model).load_from_checkpoint(best_ckpt_path)
            best_lit.eval()

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
            base_seed = cfg.get("seed", None)
            _generate_and_save_heatmaps(
                best_lit.model,
                data_dir=train_dir,
                transform=heatmap_load_transform,
                base_heatmaps_dir=base_heatmaps_dir,
                iteration=iteration,
                envelope_start=envelope_start,
                envelope_scale=envelope_scale,
                save_dir=current_heatmaps_dir,
                target_size=training_size,
                threshold=threshold,
                fill_color=fill_color,
                logger=logger,
                device=cfg.hardware.device,
                seed=None if base_seed is None else base_seed + iteration,
            )

            if iteration > 0:
                patterns = ("*.png", "*.jpg", "*.jpeg")
                sample_paths = sorted(chain.from_iterable(
                    glob.glob(os.path.join(train_dir, "pos", pat)) for pat in patterns
                ))[:20]

                debug_dir = os.path.join(
                    "out", "debug", "envelopes", f"iteration_{iteration}"
                )
                _save_envelope_debug_images(
                    base_heatmaps_dir=base_heatmaps_dir,
                    img_paths=sample_paths,
                    iteration=iteration,
                    threshold=threshold,
                    fill_color=fill_color,
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
    negative_dir = os.path.join(train_dir, "neg")

    area_cfg = cfg.mode.train_config.heatmaps.get("area_envelope", {})
    envelope_start = area_cfg.get("start_iteration", 2)
    envelope_scale = area_cfg.get("scale", 0.1)

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
            envelope_start=envelope_start,
            envelope_scale=envelope_scale,
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
            envelope_start=envelope_start,
            envelope_scale=envelope_scale,
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
    threshold: float,
    fill_color: int,
    logger: NeptuneLogger,
    target_class: int = 1,
    device: str = "cpu",
    seed: Optional[int] = None,
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
            batch_size=32,
            num_workers=4,
            dataset_type="adversarial_erasing",
            shuffle=False,
            seed=seed,
        )
        os.makedirs(save_dir, exist_ok=True)

        count = 0
        neg_count = 0

        targets = [ClassifierOutputSoftmaxTarget(1)]
        percentile = 80
        mrf = ROADMostRelevantFirst(percentile)
        lrf = ROADLeastRelevantFirst(percentile)

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
                            threshold=threshold,
                            envelope_start=envelope_start,
                            envelope_scale=envelope_scale,
                        )

                        img = erase_region_using_heatmap(
                            img, accumulated_heatmap, threshold, fill_color
                        )

                    with torch.enable_grad():
                        heatmap = gradcam.generate_heatmap(
                            model,
                            img,
                            target_class=target_class,
                        )
                        heatmap = resize_heatmap(heatmap, target_size)
                    
                    # heatmap = torch.from_numpy(grayscale_cam)

                    img_overlay = gradcam.overlay_heatmap(img, heatmap)

                    img_name = os.path.splitext(os.path.basename(img_path))[0]

                    # Resize image to training size in order to do inference
                    resized_img = F.interpolate(
                        img, size=target_size, mode="bilinear", align_corners=False
                    )

                    # Do inference
                    logits = model(resized_img)
                    probs = torch.softmax(logits, dim=1)
                    pos_prob = probs[0, target_class].item()
                    pred = int(torch.argmax(logits).item())

                    # Log heatmap and image overlay to Neptune, just for batch 0
                    if batch_idx in [0, 1, 2]:
                        # --- after logits/probs/pred are computed ---

                        # Paths for local logging
                        metrics_dir = os.path.join(base_heatmaps_dir, "metrics")
                        iter_csv = os.path.join(metrics_dir, f"iteration_{iteration}.csv")
                        all_csv  = os.path.join(metrics_dir, "features_all.csv")

                        # Convert heatmap to numpy in [0,1]
                        heatmap_np = heatmap.detach().cpu().numpy().astype("float32")
                        heatmap_np = (heatmap_np - heatmap_np.min()) / (heatmap_np.max() - heatmap_np.min() + 1e-6)

                        # ROAD targets (probability space for interpretability)
                        # targets_road = [ClassifierOutputSoftmaxTarget(target_class)]

                        # # ROAD scores @ your chosen percentile (you already instantiated mrf/lrf above)
                        # with torch.inference_mode():
                        #     mrf_score = mrf(resized_img, heatmap_np[None], targets_road, model).item()
                        #     lrf_score = lrf(resized_img, heatmap_np[None], targets_road, model).item()
                        # combined = (lrf_score - mrf_score) / 2.0

                        # # Relative drops (guard against saturation)
                        # rel_mrf = max(0.0, -mrf_score) / max(1e-6, pos_prob)
                        # rel_lrf = max(0.0, -lrf_score) / max(1e-6, pos_prob)

                        # # Margin and log-prob features
                        # logprob_pos = torch.log_softmax(logits, dim=1)[0, target_class].item()
                        # z = logits[0]
                        # z_pos = z[target_class].item()
                        # z_others = torch.cat([z[:target_class], z[target_class+1:]]) if z.numel() > 1 else z.new_tensor([-float("inf")])
                        # margin = (z_pos - z_others.max().item()) if z_others.numel() > 0 else z_pos

                        # # CAM stats @ same percentile
                        # area_ratio, num_comp, largest_comp_ratio, top_mask = _mask_stats(heatmap_np, percentile)
                        # cam_entropy = _cam_entropy(heatmap_np)

                        # # --- Spread / compactness metrics ---
                        # part_ratio = participation_ratio(heatmap_np)            # higher = more spread
                        # gini = gini_intensity(heatmap_np)                       # higher = more concentrated
                        # rg_norm = weighted_radius_of_gyration(heatmap_np)       # higher = more spread
                        # compactness = shape_compactness(top_mask)               # closer to 1 = round/compact
                        # hfill = hull_fill_ratio(top_mask)                       # closer to 1 = less concave

                        # # Sufficiency (keep-only top region): use inverse mask with CamMultImageConfidenceChange
                        # # This returns a confidence change; by convention in this lib it's (s_orig - s_pert).
                        # cmicc = CamMultImageConfidenceChange()
                        # with torch.inference_mode():
                        #     drop_keep = cmicc(resized_img, (1.0 - top_mask)[None, ...], targets_road, model, return_visualization=False)
                        # drop_keep = float(drop_keep)  # ≈ s_orig - s_pert (if your version returns s_pert - s_orig, flip sign)
                        # keep_prob = float(pos_prob - drop_keep)
                        # keep_prob = max(0.0, min(1.0, keep_prob))  # clamp to [0,1]

                        # # Build and save the row
                        # row = {
                        #     "img_id": img_name,
                        #     "iteration": iteration,
                        #     "pred": pred,
                        #     "pos_prob": float(pos_prob),
                        #     "logprob_pos": float(logprob_pos),
                        #     "margin": float(margin),
                        #     "MRF": float(mrf_score),
                        #     "LRF": float(lrf_score),
                        #     "Combined": float(combined),
                        #     "rel_MRF": float(rel_mrf),
                        #     "rel_LRF": float(rel_lrf),
                        #     "keep_prob": float(keep_prob),
                        #     "cam_area_ratio": float(area_ratio),
                        #     "cam_num_components": int(num_comp),
                        #     "cam_largest_comp_ratio": float(largest_comp_ratio),
                        #     "cam_entropy": float(cam_entropy),
                        #     "part_ratio": float(part_ratio),
                        #     "gini": float(gini),
                        #     "rg_norm": float(rg_norm),
                        #     "compactness": float(compactness),
                        #     "hull_fill": float(hfill),
                        #     "percentile": int(percentile),
                        #     "path": img_path,
                        # }
                        # _append_row(iter_csv, row)
                        # _append_row(all_csv, row)

                        # suffix = f"{pred}_pp={pos_prob:.2f}_mrf={mrf_score:.2f}_lrf={lrf_score:.2f}_comb={combined:.2f}"
                        suffix = f"{pred}_pp={pos_prob:.2f}"
                        logger.log_tensor_img(
                            img_overlay, name=f"heatmap_{img_name}_{suffix}"
                        )

                    # Generate transparent heatmap if prediction is negative
                    if pred == 0:
                        print(f"Iteration: {iteration}. Negative: {img_path}")
                        neg_count += 1
                        heatmap = torch.zeros_like(heatmap)

                    # Save heatmap
                    _save_heatmap(heatmap, save_dir, img_name, pred)

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
    assert heatmap.min() >= 0 and heatmap.max() <= 1, (
        "Heatmap tensor must be normalized in the range [0, 1]."
    )
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


def _save_envelope_debug_images(
    *,
    base_heatmaps_dir: str,
    img_paths: List[str],
    iteration: int,
    threshold: float,
    fill_color: int,
    target_size: Tuple[int, int],
    envelope_start: int,
    envelope_scale: float,
    save_dir: str,
) -> None:
    os.makedirs(save_dir, exist_ok=True)

    for img_path in img_paths:
        img_name = os.path.splitext(os.path.basename(img_path))[0]
        img_bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img_bgr is None:
            continue
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_rgb = cv2.resize(img_rgb, (target_size[1], target_size[0]))
        img_t = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0  # (C,H,W)
        img_t = img_t.unsqueeze(0)  # (1,C,H,W),

        fig, axes = plt.subplots(1, iteration, figsize=(5 * iteration, 5))
        if iteration == 1:
            axes = [axes]

        for t in range(iteration):
            cumulative = load_accumulated_heatmap(
                base_heatmaps_dir,
                img_name,
                label=1,
                iteration=t,
                threshold=threshold,
                envelope_start=envelope_start,
                envelope_scale=envelope_scale,
            )

            img_t = erase_region_using_heatmap(
                img_t, cumulative, threshold, fill_color
            )

            mask = cumulative > threshold

            next_heatmap = load_tensor(base_heatmaps_dir, t + 1, img_name)
            overlay_t = gradcam.overlay_heatmap(img_t, next_heatmap)
            if overlay_t.dim() == 4:
                overlay_t = overlay_t.squeeze(0)
            overlay_np = overlay_t.cpu().permute(1, 2, 0).clamp(0.0, 1.0).numpy()

            ax = axes[t]
            ax.imshow(overlay_np)
            if t >= envelope_start:
                envelope = _area_targeted_envelope(mask, envelope_scale)
                ax.contour(envelope.cpu(), levels=[0.5], colors="yellow", linewidths=1)
            ax.set_title(f"iter {t} -> {t + 1}")
            ax.axis("off")

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"{img_name}.png"))
        plt.close(fig)
