import os

import lightning.pytorch as pl
import numpy as np
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
import torch.nn.functional as F
from skimage import measure, morphology
from torchvision.utils import make_grid, save_image


def total_variation(mask: torch.Tensor) -> torch.Tensor:
    """Simple isotropic TV loss on Bx1xHxW mask."""
    tv_h = torch.abs(mask[:, :, 1:, :] - mask[:, :, :-1, :]).mean()
    tv_w = torch.abs(mask[:, :, :, 1:] - mask[:, :, :, :-1]).mean()
    return tv_h + tv_w


def minmax_norm(tensor):
    # tensor: B×C×H×W
    flat = tensor.view(tensor.size(0), -1)
    mins = flat.min(dim=1)[0].view(-1, 1, 1, 1)
    maxs = flat.max(dim=1)[0].view(-1, 1, 1, 1)
    return (tensor - mins) / (maxs - mins + 1e-8)


def connectivity_loss(mask: torch.Tensor, k: int = 3) -> torch.Tensor:
    """
    mask: B×1×H×W in [0,1], where 1=erase region
    k: window size (odd), e.g. 3 or 5
    """
    # a k×k box filter
    weight = torch.ones(1, 1, k, k, device=mask.device) / (k * k)
    # each pixel’s local-average of mask
    local_avg = F.conv2d(mask, weight, padding=k // 2)
    # penalize pixels with mask=1 but low local support
    # (1 - local_avg) is near 1 if the neighborhood is empty
    # multiply by mask so only “erased” pixels get penalized
    loss = ((1 - local_avg) * mask).mean()
    return loss


def budget_loss(erase: torch.Tensor, p: float) -> torch.Tensor:
    """
    erase: B×1×H×W ∈ [0,1], where 1 means pixel is erased
    p: max fraction of pixels you’re allowed to erase (0<p<1)
    """
    B, _, H, W = erase.shape
    # total erase‐mass per image
    erase_mass = erase.view(B, -1).sum(dim=1)  # (B,)
    # budget in pixels
    max_mass = p * (H * W)
    # hinge: only positive when erase_mass > max_mass
    violation = torch.clamp(erase_mass - max_mass, min=0.0)
    # average over batch
    return violation.mean() / (H * W)  # normalize if you like


def clean_mask(
    mask: np.ndarray, min_size: int = 100, closing_radius: int = 5
) -> np.ndarray:
    """
    Post-process a binary mask to keep only the largest blob,
    remove small speckles, and optionally close holes.

    Args:
      mask           : H×W numpy array of dtype bool or {0,1}
      min_size       : remove any connected component smaller than this (in pixels)
      closing_radius : radius for the disk structuring element used in binary_closing

    Returns:
      clean          : H×W numpy array (0 or 1) of the cleaned mask
    """

    # 1) Ensure boolean
    binary = mask.astype(bool)

    # 2) Remove tiny speckles
    cleaned = morphology.remove_small_objects(binary, min_size=min_size)

    # 3) Close small holes / gaps (makes the blob more convex)
    selem = morphology.disk(closing_radius)
    closed = morphology.binary_closing(cleaned, selem)

    # 4) Keep only the single largest connected component
    labels = measure.label(closed)
    if labels.max() == 0:
        return closed.astype(np.uint8)

    # Compute area of each label, ignore background label 0
    areas = [(labels == lab).sum() for lab in range(1, labels.max() + 1)]
    largest_lab = np.argmax(areas) + 1

    largest_blob = labels == largest_lab

    return largest_blob.astype(np.uint8)


class MaskerLightning(pl.LightningModule):
    def __init__(
        self,
        classifier: nn.Module,
        mask_threshold: float,
        lr: float = 5e-3,
    ):
        """
        LightningModule to train a single-channel mask via an SMP U-Net decoder.

        Args:
          classifier: a frozen classifier (Xception) that takes B×3×H×W → logits
          lr: learning rate for the mask head
          lambda_l1: weight for L1 sparsity on (1-mask)
          lambda_tv: weight for total-variation smoothness on mask
        """
        super().__init__()
        self.save_hyperparameters(ignore=["classifier"])

        # 1) Frozen classifier
        self.classifier = classifier.eval()
        for p in self.classifier.parameters():
            p.requires_grad = False

        # 2) SMP Unet: encoder frozen, train decoder+head only
        self.masker = smp.Unet(
            encoder_name="xception",
            encoder_weights="imagenet",
            in_channels=3,
            classes=1,
            activation="sigmoid",
        )

        for name, p in self.masker.encoder.named_parameters():
            p.requires_grad = False

        self.mask_threshold = mask_threshold
        self.lr = lr
        self.lambda_conf = 1.0
        self.lambda_l1 = 1.0
        self.lambda_tv = 1.0
        self.lambda_compact = 1.0
        self.lambda_spread = 1.0
        self.lambda_budget = 0.0

    def spread_loss(self, mask: torch.Tensor) -> torch.Tensor:
        # Invert the mask
        erase = 1.0 - mask  # Now 0=keep, 1=erase

        _, _, H, W = erase.shape
        device = erase.device

        # Build coordinate grids
        # ys has shape (1,1,H,1): value at [0,0,i,0] = i
        ys = torch.arange(H, device=device).view(1, 1, H, 1).float()
        # xs has shape (1,1,1,W): value at [0,0,0,j] = j
        xs = torch.arange(W, device=device).view(1, 1, 1, W).float()

        # Compute area of erasure per image
        area = erase.sum(dim=[2, 3], keepdim=True) + 1e-6  # B×1×1×1

        # Compute centroid coords
        mu_y = (ys * erase).sum(dim=[2, 3], keepdim=True) / area
        mu_x = (xs * erase).sum(dim=[2, 3], keepdim=True) / area

        # Per-pixel squared distance from centroid
        var = ((ys - mu_y) ** 2 + (xs - mu_x) ** 2) * erase

        # Sum those distances and normalize by area
        spread_per_image = var.sum(dim=[2, 3]) / area.squeeze()  # shape: (B,)

        return spread_per_image.mean()  # Average across batch

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mask_cont = self.masker(x)
        mask_hard = (mask_cont > 0.5).float()
        # ST‑estimator: forward uses mask_hard, backward uses mask_cont
        mask = mask_hard.detach() - mask_cont.detach() + mask_cont
        return mask

    def training_step(self, batch, batch_idx):
        x, y = batch  # x: input images (B×3×H×W), y: ground-truth class labels (B,)

        mask = self(x)

        # Perturb using hard mask
        # TODO: use dataset mean/std to normalize x
        bkg = x.mean(dim=[2, 3], keepdim=True)  # B×3×1×1
        x_pert = mask * x + (1.0 - mask) * bkg

        # Run classifier on perturbed image
        logits = self.classifier(x_pert)
        probs = F.softmax(logits, dim=1)
        conf = probs[:, 1].mean()  # average positive-class confidence

        # Regularizers
        l1 = torch.mean(1.0 - mask)  # encourage mask ≈1 (minimal erasure)
        tv = total_variation(mask)
        spread = self.spread_loss(mask)
        area = torch.sum(mask) + 1e-6
        compact = tv / area

        budget_l = budget_loss(1 - mask, p=0.1)

        loss_terms = {
            "conf": self.lambda_conf * conf,
            "l1": self.lambda_l1 * l1,
            "tv": self.lambda_tv * tv,
            "compact": self.lambda_compact * (compact * 1e6),
            "spread": self.lambda_spread * (spread * 1e-5),
            "budget": self.lambda_budget * budget_l,
        }
        loss = sum(loss_terms.values())
        self.log_dict({f"train/{k}": v for k, v in loss_terms.items()}, prog_bar=True)

        # 6) Save debug images every 20 batches
        if batch_idx % 5 == 0 and self.global_rank == 0:
            print(f"Epoch {self.current_epoch}, batch {batch_idx}")
            print("Loss components:")
            for k, v in loss_terms.items():
                print(f"  {k}: {v:.4f}")

            save_dir = "out/debug_images"
            os.makedirs(save_dir, exist_ok=True)

            x_vis = minmax_norm(x)
            x_pert_vis = minmax_norm(x_pert)

            # Pick up to 16 samples
            num = min(x_vis.size(0), 16)

            # Make grids
            grid_orig = make_grid(x_vis[:num], nrow=4, normalize=False)
            grid_pert = make_grid(x_pert_vis[:num], nrow=4, normalize=False)
            grid_mask = make_grid(
                mask[:num], nrow=4, normalize=True
            )  # mask already [0,1]

            masks_np = (
                mask[:num, 0].detach().cpu().numpy() > self.mask_threshold
            ).astype(np.uint8)
            cleaned_list = []
            for m in masks_np:
                cm = clean_mask(
                    m,
                    min_size=int(0.01 * m.size),  # drop <1% area
                    closing_radius=10,  # tune as needed
                )
                cleaned_list.append(cm)

            cleaned_tensor = torch.from_numpy(np.stack(cleaned_list))  # num×H×W
            cleaned_tensor = (
                cleaned_tensor.unsqueeze(1).float().to(mask.device)
            )  # num×1×H×W
            grid_clean = make_grid(cleaned_tensor, nrow=4, normalize=False)

            # Save
            epoch = self.current_epoch
            save_image(grid_orig, f"{save_dir}/e{epoch}_b{batch_idx}_orig.png")
            save_image(grid_pert, f"{save_dir}/e{epoch}_b{batch_idx}_pert.png")
            save_image(grid_mask, f"{save_dir}/e{epoch}_b{batch_idx}_mask.png")
            save_image(
                grid_clean,
                os.path.join(save_dir, f"e{epoch}_b{batch_idx}_cleaned_mask.png"),
            )

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch

        # 0) keep only positives
        pos_idx = (y == 1).nonzero(as_tuple=True)[0]
        if pos_idx.numel() == 0:
            # nothing to compute
            return None

        x = x[pos_idx]
        y = y[pos_idx]

        # 1) predict & binarize mask
        mask = self(x)

        # 2) perturb input
        bkg = x.mean(dim=[2, 3], keepdim=True)
        x_pert = mask * x + (1.0 - mask) * bkg

        # 3) classifier on perturbed
        # print("VAL")
        logits = self.classifier(x_pert)
        # print(f"Logits: {logits}")
        probs = F.softmax(logits, dim=1)
        # print(f"Probs: {probs}")
        conf = probs[torch.arange(len(y)), y].mean() * 2.0
        # print(f"Conf: {conf}")

        # 4) regularizers
        l1 = torch.mean(1.0 - mask)
        tv = total_variation(mask)
        area = torch.sum(mask) + 1e-6
        compact = tv / area

        # 5) loss
        loss = conf * 10.0 + self.lambda_l1 * l1 + self.lambda_tv * tv + compact * 1e2

        # 6) log scalars
        self.log("val/loss", loss, prog_bar=True, on_epoch=True)
        self.log("val/conf", conf, prog_bar=True, on_epoch=True)
        self.log("val/l1", l1, on_epoch=True)
        self.log("val/tv", tv, on_epoch=True)
        self.log("val/compact", compact, on_epoch=True)

        return loss

    def compute_decoder_weight_mag(self):
        # Gather absolute values of *all* trainable decoder weights
        mags = []
        for p in self.masker.decoder.parameters():
            if p.requires_grad:
                mags.append(p.data.abs().mean())
        return torch.stack(mags).mean()

    def on_train_epoch_end(self):
        # call after your existing train‐epoch logging
        avg_w_mag = self.compute_decoder_weight_mag()
        self.log("train/decoder_avg_weight_mag", avg_w_mag, prog_bar=True)

    def configure_optimizers(self):
        # only the masker parameters (decoder+head) are trainable
        return torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.masker.parameters()),
            lr=self.lr,
        )
        # return torch.optim.AdamW(
        #     filter(lambda p: p.requires_grad, self.masker.parameters()),
        #     lr=self.lr,
        #     weight_decay=1e-4,  # L2 penalty on all decoder weights
        # )
