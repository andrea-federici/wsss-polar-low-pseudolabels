import torch
from torch import nn
import torch.optim as optim
import torchvision.transforms.functional as F

from lit_model import LitModel
from image_processing import normalize_image_by_mean_std, unnormalize_image_by_statistics
from iterative_training.iterative_training_utils import calculate_max_transl_fractions
from train_config import resized_image_res, mean, std


class LitModelCustomTransl(LitModel):

    def __init__(self, model: nn.Module, criterion: nn.Module, optimizer: optim.Optimizer, max_translations_dict = None):
        super().__init__(model, criterion, optimizer)
        self.max_translations_dict = max_translations_dict
        self.max_transl_fractions = calculate_max_transl_fractions(max_translations_dict, resized_image_res)

    
    def custom_aug(self, image, max_transl):
        if max_transl is not None:
            # Apply random translation within the limits
            delta_x = torch.randint(-max_transl['left'], max_transl['right']+1, size=(1,)).item()
            delta_y = torch.randint(-max_transl['up'], max_transl['down']+1, size=(1,)).item()
        else:
            h, w = image.shape[1], image.shape[2]
            delta_x = torch.randint(low=int(-self.max_transl_fractions['left']*w), high=int(self.max_transl_fractions['right']*w)+1, size=(1,)).item()
            delta_y = torch.randint(low=int(-self.max_transl_fractions['up']*h), high=int(self.max_transl_fractions['down']*h)+1, size=(1,)).item()

        image = unnormalize_image_by_statistics(image, mean, std)

        translated_image = F.affine(
            image,
            angle=0,
            translate=(delta_x, delta_y),
            scale=1.0,
            shear=[0.0, 0.0],
            fill=[0]
        )

        max_rotation = 20
        angle = torch.randint(-max_rotation, max_rotation+1, size=(1,)).item()
        rotated_image = F.affine(
            translated_image,
            angle=angle,
            translate=(0, 0),
            scale=1.0,
            shear=[0.0, 0.0],
            fill=[0]
        )

        if torch.rand(1).item() < 0.5:
            rotated_image = F.hflip(rotated_image)

        if torch.rand(1).item() < 0.5:
            rotated_image = F.vflip(rotated_image)

        rotated_image = normalize_image_by_mean_std(rotated_image, mean, std)

        return rotated_image


    def training_step(self, batch, batch_idx):
        images, labels, max_transl_dict = batch # Unpack the batch
        
        transformed_images = [
            self.custom_aug(img, max_transl) for img, max_transl in zip(images, max_transl_dict)
        ]
        transformed_images = torch.stack(transformed_images)
        
        loss = super().process_step('train', transformed_images, labels)
        return loss


    def validation_step(self, batch, batch_idx):
        images, labels, _ = batch
        loss = super().process_step('val', images, labels)
        return loss
