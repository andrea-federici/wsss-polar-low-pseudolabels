from torchvision import transforms
from omegaconf import DictConfig

def get_transform(cfg: DictConfig, stage: str) -> transforms.Compose:

    transforms_config = cfg.transforms

    image_size = (
        transforms_config.image_width,
        transforms_config.image_height
    )

    mean = transforms_config.normalization.mean
    std = transforms_config.normalization.std

    if stage == 'train':
        transform_list = [
            transforms.RandomAffine(
                degrees=transforms_config.augmentation.degrees,
                translate=tuple(transforms_config.augmentation.translate_frac),
                scale=tuple(transforms_config.augmentation.scale),
                fill=0
            ),
            transforms.Resize(image_size)
        ]

        if transforms_config.augmentation.horizontal_flip:
            transform_list.append(transforms.RandomHorizontalFlip())
        
        if transforms_config.augmentation.vertical_flip:
            transform_list.append(transforms.RandomVerticalFlip())
        
        transform_list.extend([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
        
        transform = transforms.Compose(transform_list)
    elif stage in ['val', 'test']:
        transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
    else:
        raise ValueError(f"Invalid stage: {stage}. Expected one of 'train', "
                         f"'val', 'test'.")
    
    return transform