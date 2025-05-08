import os
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
from torchvision.transforms import GaussianBlur
from torchvision.utils import save_image
from tqdm import tqdm


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


def spread_loss(mask: torch.Tensor) -> torch.Tensor:
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


# def find_hurricane_mask(
#     x: torch.Tensor,
#     classifier: torch.nn.Module,
#     hurricane_class: int,
#     *,
#     num_steps: int = 300,
#     lr: float = 0.1,
#     coarse_size: int = 16,
#     lambda_l1: float = 1e-3,
#     lambda_tv: float = 1e-2,
#     lambda_bin: float = 1e-2,
#     lambda_conn: float = 1e-2,
#     lambda_spread: float = 1e-2,
#     blur_sigma: float = 10.0,
#     jitter: int = 4,
#     device: torch.device = None,
#     save_dir: str = None,
#     save_interval: int = 100,
# ) -> Tuple[torch.Tensor, Dict[str, List[float]]]:
#     """
#     Given:
#       x:             a B×3×H×W batch of SAR images (normalized)
#       classifier:    your frozen model returning logits
#       hurricane_class: index of the “positive” class
#     Returns:
#       m:             B×1×H×W mask in [0,1], tight on the hurricane region

#     Minimizes:
#       λ1·‖1−m‖₁ + λ2·TV(m) + f_c(Φ(x;m))
#     where Φ(x;m) = m⊙x + (1−m)⊙b, and b is a Gaussian-blurred background.
#     """
#     if device is None:
#         device = x.device

#     x = x.to(device)

#     losses = {
#         "total": [],
#         "score": [],
#         "l1": [],
#         "tv": [],
#         # "bin": [],
#         # "conn": [],
#         # "spread": [],
#     }

#     B, C, H, W = x.shape
#     classifier = classifier.eval().to(device)
#     for p in classifier.parameters():
#         p.requires_grad = False

#     # 1) Initialize mask = all-ones (no erasure)
#     # m = torch.ones((B, 1, H, W), device=device, requires_grad=True)
#     m_small = torch.ones(
#         (B, 1, coarse_size, coarse_size), device=device, requires_grad=True
#     )

#     # 2) Precompute blurred background b = GaussianBlur(x)
#     # blur = GaussianBlur(kernel_size=int(4 * blur_sigma + 1), sigma=blur_sigma)
#     # b = blur(x.cpu()).to(device)  # B×3×H×W

#     # 2) Fixed fill color
#     fill = torch.tensor([0.0, 0.0, 0.0], device=device)  # black
#     b = fill.view(1, 3, 1, 1).expand(B, -1, H, W)

#     optimizer = torch.optim.Adam([m_small], lr=lr)

#     # Optional: prepare save dir
#     if save_dir is not None:
#         os.makedirs(save_dir, exist_ok=True)
#         # we'll unnormalize with these if you used Normalize(mean,std)
#         mean = torch.tensor([0.2872, 0.2872, 0.4595], device=device).view(1, 3, 1, 1)
#         std = torch.tensor([0.1806, 0.1806, 0.2621], device=device).view(1, 3, 1, 1)

#     for step in tqdm(range(num_steps)):
#         optimizer.zero_grad()

#         m = F.interpolate(m_small, size=(H, W), mode="nearest")

#         # 3) Jitter (optional): shift image+mask by up to jitter pixels
#         if jitter > 0:
#             dy = torch.randint(-jitter, jitter + 1, (1,), device=device).item()
#             dx = torch.randint(-jitter, jitter + 1, (1,), device=device).item()
#             x_j = torch.roll(x, shifts=(dy, dx), dims=(2, 3))
#             b_j = torch.roll(b, shifts=(dy, dx), dims=(2, 3))
#             m_j = torch.roll(m, shifts=(dy, dx), dims=(2, 3))
#         else:
#             x_j, b_j, m_j = x, b, m

#         # 3a) Binarize for perturbation
#         m_j_hard = (m_j > 0.5).float()  # exactly 0 or 1
#         # 3b) Straight-through trick: forward uses m_hard, backward uses m
#         m_ste = m_j_hard.detach() - m.detach() + m

#         # 4) Perturb with the hard mask
#         x_pert = m_j * x_j + (1 - m_j) * b_j
#         logits = classifier(x_pert)  # B×num_classes
#         score = F.softmax(logits, dim=1)[:, hurricane_class]
#         score = score.mean()

#         # —— Save some perturbed images ——
#         if save_dir is not None and step % save_interval == 0:
#             # take the first image in batch
#             img_pert = x_pert[0]  # 3×H×W
#             # unnormalize back to [0,1] if you normalized originally:
#             img_vis = img_pert * std + mean
#             img_vis = img_vis.clamp(0, 1)
#             # save as PNG
#             save_image(img_vis, os.path.join(save_dir, f"pert_step{step:03d}.png"))

#         # 5) Regularizers
#         # 5a) Binarization loss: zero only at {0,1}, max at 0.5
#         bin_loss = (m_small * (1 - m_small)).mean()  # encourages m→{0,1}
#         l1 = (1 - m_small).abs().mean()
#         # isotropic TV
#         tv = (
#             torch.abs(m_small[:, :, 1:, :] - m_small[:, :, :-1, :]).mean()
#             + torch.abs(m_small[:, :, :, 1:] - m_small[:, :, :, :-1]).mean()
#         )
#         # 5b) Connectivity loss: penalize isolated pixels
#         conn = connectivity_loss(1 - m_small, k=5)
#         spread = spread_loss(m_small) * 1e-5
#         loss = (
#             lambda_l1 * l1
#             + lambda_tv * tv
#             + score
#             # + bin_loss * lambda_bin
#             # + conn * lambda_conn
#             # + spread * lambda_spread
#         )

#         # Update losses
#         losses["total"].append(loss.item())
#         losses["score"].append(score.item())
#         losses["l1"].append(l1.item())
#         losses["tv"].append(tv.item())
#         # losses["bin"].append(bin_loss.item())
#         # losses["conn"].append(conn.item())
#         # losses["spread"].append(spread.item())

#         # Print losses
#         if step % 10 == 0:
#             print(
#                 f"Step {step:03d}: "
#                 f"loss={loss.item():.4f}, "
#                 f"score={score.item():.4f}, "
#                 f"l1={l1.item():.4f}, "
#                 f"tv={tv.item():.4f}, "
#                 # f"bin={bin_loss.item():.4f}, "
#                 # f"conn={conn.item():.4f}, ",
#                 # f"spread={spread.item():.4f}",
#             )

#         # 6) Backprop & clamp
#         loss.backward()
#         optimizer.step()
#         with torch.no_grad():
#             m.clamp_(0.0, 1.0)

#         m_final = F.interpolate(m_small, size=(H, W), mode="nearest")

#     return (m_final > 0.5).float(), losses


def find_hurricane_mask(
    x: torch.Tensor,
    classifier: torch.nn.Module,
    hurricane_class: int,
    *,
    num_steps: int = 500,
    lr: float = 0.05,
    coarse_size: int = 32,
    lambda_l1: float = 0.1,  # Controls erasure vs. score tradeoff
    lambda_tv: float = 0.2,
    blur_sigma: float = 15.0,
    init_strategy: str = "random",
    device: torch.device = None,
) -> torch.Tensor:
    """
    Returns binary mask where 1 = keep (preserve hurricane), 0 = erase
    Goal: Find smallest erased regions (0s) that minimize classifier confidence
    """
    # ------ Setup ------
    if device is None:
        device = x.device

    x = x.to(device)

    B, C, H, W = x.shape
    classifier = classifier.eval().to(device)

    # Precompute blurred background
    blur = GaussianBlur(kernel_size=int(6 * blur_sigma) + 1, sigma=blur_sigma)
    with torch.no_grad():
        b = blur(x.cpu()).to(device)  # Use blurred background for erased regions

    # ------ Mask Initialization ------
    if init_strategy == "ones":
        logits = torch.zeros(
            B, 1, coarse_size, coarse_size, device=device
        )  # Sigmoid(0)=0.5
    else:  # "random"
        logits = torch.randn(B, 1, coarse_size, coarse_size, device=device) * 0.5
    logits = logits.requires_grad_(True)

    optimizer = torch.optim.Adam([logits], lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, num_steps)

    # ------ Optimization Loop ------
    for step in range(num_steps):
        optimizer.zero_grad()

        # 1. Generate smooth mask with annealed temperature
        temp = max(0.3, 1.0 - step / num_steps * 0.8)  # From 1.0 → 0.2
        m_coarse = torch.sigmoid(logits / temp)  # (B,1,Hc,Wc)

        # 2. Upsample to full resolution
        m = F.interpolate(m_coarse, (H, W), mode="bilinear", align_corners=False)

        # 3. Apply perturbation: m=1 keeps pixels, m=0 uses blurred background
        x_pert = m * x + (1 - m) * b

        # 4. Classifier score to MINIMIZE
        logits_pred = classifier(x_pert)
        score = F.softmax(logits_pred, dim=1)[:, hurricane_class].mean()

        # 5. Regularization losses
        l1_loss = (1 - m_coarse).mean()  # Penalize erasure (0s in mask)
        tv_loss = (torch.abs(m_coarse[:, :, 1:, :] - m_coarse[:, :, :-1, :]).mean()) + (
            torch.abs(m_coarse[:, :, :, 1:] - m_coarse[:, :, :, :-1]).mean()
        )

        # 6. Total loss (MINIMIZE score + regularization)
        loss = score + lambda_l1 * l1_loss + lambda_tv * tv_loss

        # 7. Backprop & update
        loss.backward()
        optimizer.step()
        scheduler.step()

    # ------ Final thresholded mask ------
    m_final = (
        F.interpolate(torch.sigmoid(logits), (H, W), mode="bilinear") > 0.2
    ).float()
    return m_final


import matplotlib.pyplot as plt
from captum.attr import Occlusion
from src.data.image_processing import plot_to_pil_image, normalize_image_to_range


def captum_hurricane_occlusion(
    x: torch.Tensor,
    classifier: torch.nn.Module,
    hurricane_class: int,
    patch_size: int = None,
    stride: int = None,
    baseline: float = 0.5,
    threshold: float = 0.5,
    save_dir: str = None,
    device: torch.device = None,
) -> torch.Tensor:
    """
    Only supports batch size = 1. Returns a [1, 1, H, W] heatmap in [0, 1].
    """
    assert x.ndim == 4 and x.shape[0] == 1, "Input must be shape [1, 3, H, W]."
    if device is None:
        device = x.device
    x = x.to(device)

    classifier = classifier.eval().to(device)
    for p in classifier.parameters():
        p.requires_grad = False

    def model_forward(inp):
        logits = classifier(inp)
        return logits[:, hurricane_class]

    occ = Occlusion(model_forward)

    print(f"Shape of x: {x.shape}")

    attributions = occ.attribute(
        inputs=x,
        sliding_window_shapes=(1, patch_size, patch_size),
        strides=(1, stride, stride),
        baselines=baseline,
    )

    print(f"Shape of attributions: {attributions.shape}")

    a = attributions[0, 0]  # [H, W]
    a = a - a.min()
    a = a / (a.max() + 1e-8)
    heatmap = a.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]

    mask = (heatmap >= threshold).float()  # [1, 1, H, W]

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        img = x[0]  # [3, H, W]
        m = mask[0]  # [1, H, W]
        img_np = normalize_image_to_range(img.cpu().permute(1, 2, 0).numpy())
        m_np = m.cpu().permute(1, 2, 0).numpy()

        fig, axs = plt.subplots(1, 2, figsize=(8, 4))
        axs[0].imshow(img_np)
        axs[0].set_title("Original Image")
        axs[0].axis("off")

        axs[1].imshow(m_np, cmap="gray", vmin=0, vmax=1)
        axs[1].set_title("Binary Mask")
        axs[1].axis("off")

        plt.tight_layout()

        plot_pil = plot_to_pil_image(fig)
        plot_pil.save(os.path.join(save_dir, "sidebyside.png"))

    return mask
