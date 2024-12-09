
import numpy as np

import torch
from captum.attr import LRP
from captum.attr._core.lrp import EpsilonRule


# Custom pass-through rule for Identity layers
class IdentityRule(EpsilonRule):
    def apply(self, attributions, inputs, outputs):
        return attributions

# Subclass LRP to include custom rules
class CustomLRP(LRP):
    def _check_and_attach_rules(self):
        # Call parent method to set up default rules
        super()._check_and_attach_rules()
        
        # Attach custom IdentityRule for Identity layers
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Identity):
                self.rule_dict[module] = IdentityRule()


def replace_flatten_with_identity(model):
    # Create a list of module names to modify
    modules_to_replace = [
        name for name, module in model.named_modules()
        if isinstance(module, torch.nn.Flatten)
    ]
    # Replace each module safely
    for module_name in modules_to_replace:
        # Access the parent module
        parent_module = model
        sub_names = module_name.split('.')
        for sub_name in sub_names[:-1]:
            parent_module = getattr(parent_module, sub_name)
        # Replace the module
        setattr(parent_module, sub_names[-1], torch.nn.Identity())
    return model


def generate_heatmap(model: torch.nn.Module, input_image: torch.Tensor, target_class: int = None, device = 'cpu'):
    model.eval()

    input_image = input_image.to(device) # Move image to the same device as the model

    lrp = LRP(model)

    if target_class is None:
        # Get the predicted class for the input image
        logits = model(input_image)
        target_class = torch.argmax(logits, dim=1).item()
    
    attr = lrp.attribute(input_image, target=target_class)

    attr = attr.squeeze().detach().cpu().numpy() # Convert to NumPy format
    heatmap = np.maximum(attr, 0) # Remove negative values in the heatmap
    norm_heatmap = heatmap / np.max(heatmap) if np.max(heatmap) != 0 else 1 # Normalize the heatmap between 0 and 1

    return norm_heatmap


# ---------------------------- TESTING ----------------------------

import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim

from models import XceptionModel
from model_container import ModelContainer
from data_utils import load_and_transform_image, pick_random_image
from image_utility import translate_image, normalize_image, convert_to_np_array
from train_config import transform_prep


torch_model = XceptionModel(num_classes=2)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(filter(lambda p: p.requires_grad, torch_model.parameters()), lr=0.001)

model = ModelContainer.load_from_checkpoint('checkpoints/xception-xaug.ckpt', model=torch_model, criterion=criterion, optimizer=optimizer)

image_path = pick_random_image('data/train', 'pos')
# image_path = 'data/train/pos/5d6964_20160902T082858_20160902T083102_mos_rgb.png'
print(image_path)
image = load_and_transform_image(image_path, transform_prep)
heatmap = generate_heatmap(model, image)

plt.imshow(heatmap)
plt.show()

