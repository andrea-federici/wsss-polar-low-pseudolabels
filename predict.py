import hydra
import torch
import torch.nn.functional as F
from dotenv import load_dotenv
from omegaconf import DictConfig, OmegaConf
from PIL import Image
from torchvision.utils import make_grid, save_image

from src.data.augmentation import get_transform
from src.data.image_processing import erase_region_using_heatmap
from src.train.helper import load_accumulated
from src.train.setup import get_predict_setup


@hydra.main(version_base=None, config_path="conf", config_name="config")
def run(cfg: DictConfig) -> None:
    load_dotenv()

    # Mask sensitive information
    cfg_copy = OmegaConf.create(cfg)  # Create a mutable copy
    cfg_copy.neptune.api_key = "*****"  # Mask or remove the sensitive key
    print(OmegaConf.to_yaml(cfg_copy, resolve=True))

    lightning_model = get_predict_setup("out/checkpoints/adver_it0.ckpt", cfg)
    lightning_model.eval()

    # Load image
    image_path = "data/train/pos/0aa613_20181003T134110_20181003T134210_mos_rgb.png"
    # image_path = "data/train/neg/0ee435_20190401T142401_20190401T142605_mos_rgb.png"

    image = Image.open(image_path).convert("RGB")

    transform = get_transform(cfg.transforms, "test")

    # Apply the transform to the image
    transformed_image = transform(image).to("cuda")

    adv_img = erase_region_using_heatmap(
        transformed_image.unsqueeze(0),
        load_accumulated(
            "out/heatmaps",
            img_name=image_path,
            label=1,
            iteration=3,
        ),
        threshold=0.75,
    )

    # Plot the image
    save_image(make_grid(adv_img), "out/adv_img.png")

    print(transformed_image.shape)
    logits = lightning_model(adv_img)
    print(f"Logits: {logits}")
    print(f"Probabilities: {torch.softmax(logits, dim=1)}")
    print(f"Prediction: {torch.argmax(logits, dim=1).item()}")


if __name__ == "__main__":
    run()
