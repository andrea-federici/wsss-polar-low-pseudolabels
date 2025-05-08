import os

import torch
from dotenv import load_dotenv
from torchvision import transforms
from torchvision.utils import save_image

from src.data.data_loaders import create_data_loaders
from src.models.lightning import BaseModel
from src.models.lightning.mask_head import MaskerLightning
from src.models.torch import Xception
from src.train.logger import create_neptune_logger
from src.utils.getters import criterion_getter, lr_scheduler_getter, optimizer_getter

load_dotenv()

neptune_logger = create_neptune_logger(
    "andreaf/polarlows", os.getenv("NEPTUNE_API_TOKEN")
)

# freeze & prepare your Xception classifier first
torch_model = Xception(num_classes=2)

criterion = criterion_getter("cross_entropy")
optimizer = optimizer_getter("adam", torch_model, 1e-4)

lr_scheduler = lr_scheduler_getter(
    "reduce_lr_on_plateau",
    optimizer,
    mode="min",
    patience=5,
    factor=0.5,
)
optimizer_config = {
    "optimizer": optimizer,
    "lr_scheduler": {
        "scheduler": lr_scheduler,
        "monitor": "val/loss",
        "interval": "epoch",
    },
}

lit_model = BaseModel.load_from_checkpoint(
    "out/checkpoints/adver_it0.ckpt",
    model=torch_model,
    criterion=criterion,
    optimizer_config=optimizer_config,
)


classifier = lit_model.model

# load weights / checkpoint into classifier ...
classifier.eval()
for p in classifier.parameters():
    p.requires_grad = False

image_size = (512, 512)

mean = [0.2872, 0.2872, 0.4595]
std = [0.1806, 0.1806, 0.2621]

# dataset + dataloader
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

transform_list.extend([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])

transform_train = transforms.Compose(transform_list)

transform_val = transforms.Compose(
    [
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ]
)

train_loader, val_loader, _ = create_data_loaders(
    "data",
    batch_size=1,
    num_workers=8,
    transform_train=transform_train,
    transform_val=transform_val,
    only_positives=True,
    shuffle_train=False,
)

# init LightningModule
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


from adv_perturb import captum_hurricane_occlusion, find_hurricane_mask

out_dir = "out/perturbed_masks"
os.makedirs(out_dir, exist_ok=True)

from tqdm import tqdm

# After you pull x from the loader, x is on CPU by default,
# but you want mean/std on CPU as well for this un-normalize:
mean_tensor = torch.tensor(mean, dtype=torch.float32).view(1, 3, 1, 1)
std_tensor = torch.tensor(std, dtype=torch.float32).view(1, 3, 1, 1)

# Loop through dataloader
for batch_idx, (x, y) in enumerate(tqdm(train_loader)):
    # mask = find_hurricane_mask(
    #     x,
    #     lit_model.model,
    #     hurricane_class=1,
    #     num_steps=200,  # 300
    #     lr=0.2,
    #     lambda_l1=0.5,
    #     lambda_tv=1.0,
    #     blur_sigma=20,
    #     device="cuda",
    #     # save_dir="out/train_perturbed_masks",
    #     # save_interval=100,
    # )

    mask = captum_hurricane_occlusion(
        x,
        lit_model.model,
        hurricane_class=1,
        patch_size=64,
        stride=16,
        save_dir="out/occlusion",
        device="cuda",
    )

    mask = mask.cpu()

    # 2) bring images back to [0,1] for saving
    x_vis = x.cpu() * std_tensor + mean_tensor
    x_vis = x_vis.clamp(0, 1)

    B, _, H, W = x_vis.shape
    for j in range(B):
        img = x_vis[j]  # 3×H×W
        # m = (mask[j] > 0.5).float()  # 1×H×W
        m = mask[j]  # 1×H×W
        m3 = m.repeat(3, 1, 1)  # 3×H×W
        pair = torch.cat([img, m3], dim=2)  # 3×H×(2W)

        filename = f"img_{batch_idx:03d}_{j:03d}.png"
        save_image(pair, os.path.join(out_dir, filename))

    # Save losses plots
    # fig, ax = plt.subplots(figsize=(6, 6))
    # ax.plot(losses["total"], label="total")
    # ax.plot(losses["score"], label="score")
    # ax.plot(losses["l1"], label="l1")
    # ax.plot(losses["tv"], label="tv")
    # # ax.plot(losses["bin"], label="bin")
    # # ax.plot(losses["conn"], label="conn")
    # # ax.plot(losses["spread"], label="spread")
    # ax.legend()
    # ax.set_xlabel("steps")
    # ax.set_ylabel("loss")
    # ax.set_title("Losses")
    # fig.savefig(os.path.join(out_dir, f"losses_{batch_idx:03d}.png"))
    # plt.close(fig)

    if batch_idx == 20:
        break

# fit
# trainer = Trainer(
#     max_epochs=50,
#     accelerator="gpu",
#     devices=1,
#     callbacks=callbacks,
#     logger=neptune_logger,
#     enable_checkpointing=True,
#     log_every_n_steps=1,
#     enable_progress_bar=True,
#     profiler="simple",
#     deterministic=True,
# )
# trainer.fit(mask_trainer, train_loader, val_loader)

# mask_trainer.eval().to("cuda")

# # 2) Image transform (same as training)
# infer_transform = transforms.Compose(
#     [
#         transforms.Resize(image_size),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=mean, std=std),
#     ]
# )

# # 3) Paths to test images
# sample_paths = ["data/train/pos/0aa613_20181003T134110_20181003T134210_mos_rgb.png"]

# # 4) Output folder
# os.makedirs("output/masks", exist_ok=True)

# # 5) Inference and saving
# for img_path in sample_paths:
#     # a) Load and transform image
#     img = Image.open(img_path).convert("RGB")
#     x = infer_transform(img).unsqueeze(0).to("cuda")  # B=1

#     # b) Predict mask
#     with torch.no_grad():
#         mask = mask_trainer(x)  # B×1×H×W
#     mask = mask.squeeze().cpu().numpy()  # H×W

#     # c) Unnormalize original image for saving
#     unnorm = transforms.Normalize(
#         mean=[-m / s for m, s in zip(mean, std)],
#         std=[1 / s for s in std],
#     )
#     img_tensor = unnorm(x.squeeze().cpu())
#     img_np = img_tensor.permute(1, 2, 0).clamp(0, 1).numpy()  # H×W×C

#     # d) Overlay: mask in inferno colormap, semi-transparent
#     fig, ax = plt.subplots(figsize=(6, 6))
#     ax.imshow(img_np)
#     ax.imshow(mask, cmap="inferno", alpha=0.5)
#     ax.axis("off")

#     # e) File names
#     base = os.path.splitext(os.path.basename(img_path))[0]
#     orig_path = f"output/masks/{base}_orig.png"
#     mask_path = f"output/masks/{base}_mask.png"
#     overlay_path = f"output/masks/{base}_overlay.png"

#     # f) Save
#     img.save(orig_path)
#     plt.imsave(mask_path, mask, cmap="gray")
#     fig.savefig(overlay_path, bbox_inches="tight", pad_inches=0)
#     plt.close(fig)

#     print(f"Saved: {orig_path}, {mask_path}, {overlay_path}")
