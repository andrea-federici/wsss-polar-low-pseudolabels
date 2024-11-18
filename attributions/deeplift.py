
import numpy as np
import torch
from captum.attr import DeepLift

from train_config import device

def generate_heatmap(model: torch.nn.Module, input_image: torch.Tensor, baseline=None, target_class=-1):
    model.eval()

    input_image = input_image.to(device)

    if baseline is None:
        baseline = torch.zeros_like(input_image)
    
    # Initialize DeepLift
    deeplift = DeepLift(model)

    # If the target class is not specified, use the predicted class
    if target_class == -1:
        logits = model(input_image)
        predicted_class = torch.argmax(logits, dim=1).item()
        target_class = predicted_class

    attributions = deeplift.attribute(
        input_image,
        baselines=baseline,
        target=target_class
    )

    # Convert the attributions to a NumPy array
    attributions = attributions.squeeze().detach().cpu().numpy()
    
    # Normalize the attributions

    attributions = np.sum(attributions, axis=0) # Sum across channels
    
    # attributions = attributions.mean(dim=1).squeeze()
    attributions = np.maximum(attributions, 0) # Remove negative values
    norm_attributions = attributions / attributions.max() # Normalize between 0 and 1

    return norm_attributions