import os

from dotenv import load_dotenv
import torch
from pytorch_lightning import Trainer
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

from src.data.data_loading import create_data_loaders
from src.models.torch import Xception
from src.models.lightning import BaseModel
from src.models.lightning.mask_head import MaskerLightning
from src.train.logger import create_neptune_logger
from src.utils.getters import (
    criterion_getter,
    optimizer_getter,
    lr_scheduler_getter,
)

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
    batch_size=16,
    num_workers=8,
    transform_train=transform_train,
    transform_val=transform_val,
)

# init LightningModule
mask_trainer = MaskerLightning(classifier=classifier)

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
trainer = Trainer(
    max_epochs=50,
    accelerator="gpu",
    devices=1,
    callbacks=callbacks,
    logger=neptune_logger,
    enable_checkpointing=True,
    log_every_n_steps=1,
    enable_progress_bar=True,
    profiler="simple",
    deterministic=True,
)
trainer.fit(mask_trainer, train_loader, val_loader)

mask_trainer.eval().to("cuda")

# 2) Image transform (same as training)
infer_transform = transforms.Compose(
    [
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ]
)

# 3) Paths to test images
sample_paths = ["data/train/pos/0aa613_20181003T134110_20181003T134210_mos_rgb.png"]

# 4) Output folder
os.makedirs("output/masks", exist_ok=True)

# 5) Inference and saving
for img_path in sample_paths:
    # a) Load and transform image
    img = Image.open(img_path).convert("RGB")
    x = infer_transform(img).unsqueeze(0).to("cuda")  # B=1

    # b) Predict mask
    with torch.no_grad():
        mask = mask_trainer(x)  # B×1×H×W
    mask = mask.squeeze().cpu().numpy()  # H×W

    # c) Unnormalize original image for saving
    unnorm = transforms.Normalize(
        mean=[-m / s for m, s in zip(mean, std)],
        std=[1 / s for s in std],
    )
    img_tensor = unnorm(x.squeeze().cpu())
    img_np = img_tensor.permute(1, 2, 0).clamp(0, 1).numpy()  # H×W×C

    # d) Overlay: mask in inferno colormap, semi-transparent
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(img_np)
    ax.imshow(mask, cmap="inferno", alpha=0.5)
    ax.axis("off")

    # e) File names
    base = os.path.splitext(os.path.basename(img_path))[0]
    orig_path = f"output/masks/{base}_orig.png"
    mask_path = f"output/masks/{base}_mask.png"
    overlay_path = f"output/masks/{base}_overlay.png"

    # f) Save
    img.save(orig_path)
    plt.imsave(mask_path, mask, cmap="gray")
    fig.savefig(overlay_path, bbox_inches="tight", pad_inches=0)
    plt.close(fig)

    print(f"Saved: {orig_path}, {mask_path}, {overlay_path}")
