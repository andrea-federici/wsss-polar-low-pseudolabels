
import cv2

import numpy as np
from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score
)
import torch
from torch import nn
import torch.optim as optim
from torchvision.transforms import ToPILImage
import pytorch_lightning as pl
from captum.attr import (
    DeepLift,
    IntegratedGradients
)

from verbose_util import init_verbose


class ModelContainer(pl.LightningModule):

    def __init__(self, model: nn.Module, criterion: nn.Module, optimizer: optim.Optimizer = None, verbose=False):
        super(ModelContainer, self).__init__()
        self.model = model
        self.criterion = criterion # Loss function
        self.optimizer = optimizer # Defaults to Adam if not given as input
        self.verboseprint = init_verbose(verbose) # Initialize the verboseprint function
        self.train_losses = [] # Stores train losses for each epoch
        self.train_outputs = [] # Stores training outputs across steps
        self.val_losses = [] # Stores validation losses for each epoch
        self.val_outputs = [] # Stores validation outputs across steps


    # The 'forward' method is called automatically when the model is invoked on input data
    def forward(self, x):
        return self.model(x) # self.model(x) calls the 'forward' method of the contained model


    # Called for each batch of data
    def training_step(self, batch, batch_idx):
        images, labels = batch # Unpack the batch
        logits = self(images) # Forward pass to get predictions
        loss = self.criterion(logits, labels) # Calculate loss

        preds = torch.argmax(logits, dim=1)
        self.train_outputs.append({
            'preds': preds.cpu().numpy(),
            'labels': labels.cpu().numpy(),
            'loss': loss.item()
        })

        return loss # Return the loss for optimization


    # Called at the end of every training epoch to aggregate metrics and print them
    def on_train_epoch_end(self):
        all_preds = [] # Collects all predictions from the epoch
        all_labels = [] # Collects all labels from the epoch
        total_loss = 0.0 # Tracks total loss across batches

        for output in self.train_outputs:
            all_preds.extend(output['preds'])
            all_labels.extend(output['labels'])
            total_loss += output['loss']

        avg_loss = total_loss / len(self.train_outputs) # Average loss for the epoch
        precision = precision_score(all_labels, all_preds, average='binary')
        recall = recall_score(all_labels, all_preds, average='binary')
        f1 = f1_score(all_labels, all_preds, average='binary')

        self.train_losses.append(avg_loss) # Store the average loss for this epoch

        print(f'Epoch {self.current_epoch} - Training:')
        print(f'Loss: {avg_loss:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}')

        self.train_outputs.clear() # Clear the outputs for the next epoch


    def validation_step(self, batch, batch_idx):
        images, labels = batch
        logits = self(images)
        loss = self.criterion(logits, labels) # Calculate validation loss

        preds = torch.argmax(logits, dim=1)

        # Store the step outputs for later use in the epoch end
        self.val_outputs.append({
            'preds': preds.cpu().numpy(),
            'labels': labels.cpu().numpy(),
            'loss': loss.item()
        })

        self.log('val_loss', loss, prog_bar=True)


    # At the end of each validation epoch, this method aggregates the outputs from all validation batches
    def on_validation_epoch_end(self):
        all_preds = []
        all_labels = []
        total_loss = 0.0

        for output in self.val_outputs:
            all_preds.extend(output['preds'])
            all_labels.extend(output['labels'])
            total_loss += output['loss']

        avg_loss = total_loss / len(self.val_outputs)
        precision = precision_score(all_labels, all_preds, average='binary')
        recall = recall_score(all_labels, all_preds, average='binary')
        f1 = f1_score(all_labels, all_preds, average='binary')

        self.val_losses.append(avg_loss) # Store average validation loss for the epoch

        print(f"Epoch {self.current_epoch} - Validation:")
        print(f"Loss: {avg_loss:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")

        self.val_outputs.clear() # Clear validation outputs for the next epoch


    def generate_integrated_gradients_heatmap(self, input_image, target_class=None, n_steps=50, baseline=None):
        self.eval()

        input_image = input_image.to(self.device) # Move image to the same device as the model

        if baseline is None:
            baseline = torch.zeros_like(input_image) # Use a black image as the baseline is no baseline is provided

        ig = IntegratedGradients(self.forward)

        # Get predicted class
        logits = self(input_image)
        predicted_class = torch.argmax(logits, dim=1).item()

        # If the target class is not specified, use the predicted class
        if target_class is None:
            target_class = predicted_class

        attributions, _ = ig.attribute(
            input_image,
            baselines=baseline,
            target=target_class,
            n_steps=n_steps,
            return_convergence_delta=True
        )

        attributions = attributions.squeeze().detach().cpu().numpy() # Convert to NumPy

        # Aggregate attributions across channels
        attributions = np.mean(attributions, axis=0)

        # Normalize the attributions to [0, 1]
        norm_attributions = (attributions - np.min(attributions)) / (np.max(attributions) - np.min(attributions) + 1e-8) # Normalize the attributions

        # Enhance contrast
        # norm_attributions = cv2.equalizeHist((norm_attributions * 255).astype(np.uint8)) / 255.0

        return norm_attributions, predicted_class
    

    def generate_deeplift_heatmap(self, input_image, baseline=None, target_class=None):
        self.verboseprint('Generating DeepLift heatmap...')
        
        self.eval()

        input_image = input_image.to(self.device)

        if baseline is None:
            baseline = torch.zeros_like(input_image)
        
        self.verboseprint('Input image shape:', input_image.shape)
        self.verboseprint('Baseline shape:', baseline.shape)
        
        # Initialize DeepLift
        deeplift = DeepLift(self.model)

        # If the target class is not specified, use the predicted class
        if target_class is None:
            self.verboseprint('Target class not specified. Using predicted class...')
            logits = self(input_image)
            self.verboseprint('Logits:', logits)
            predicted_class = torch.argmax(logits, dim=1).item()
            target_class = predicted_class
        
        self.verboseprint(f'Target class: {target_class}')

        attributions = deeplift.attribute(
            input_image,
            baselines=baseline,
            target=target_class
        )

        self.verboseprint(f'Attributions shape: {attributions.shape}')
        self.verboseprint(f'Attributions min: {attributions.min()}')
        self.verboseprint(f'Attributions max: {attributions.max()}')
        self.verboseprint(f'Attributions mean: {attributions.mean()}')

        # Convert the attributions to a NumPy array
        attributions = attributions.squeeze().detach().cpu().numpy()
        
        # Normalize the attributions

        attributions = np.sum(attributions, axis=0) # Sum across channels
        self.verboseprint(f'Attributions after summation: {attributions}')
        
        # attributions = attributions.mean(dim=1).squeeze()
        attributions = np.maximum(attributions, 0) # Remove negative values
        norm_attributions = attributions / attributions.max() # Normalize between 0 and 1

        return norm_attributions


    def overlay_green_heatmap(self, original_image, heatmap, predicted_class, target_class=None, alpha: float = 0.6, percentile_neg: float = 97, percentile_pos: float = 98, gaussian_blur_size: int = 15, verbose=False):
        image_np = np.array(original_image) # Convert the PIL image to a NumPy array

        heatmap_resized = cv2.resize(heatmap, (image_np.shape[1], image_np.shape[0]))

        # If the target class is not specified, use the predicted class
        if target_class is None:
            target_class = predicted_class

        # Set the percentile based on the target class
        if target_class == 0:
            percentile = percentile_neg
        else:
            percentile = percentile_pos

        # Flatten the heatmap
        heatmap_flattened = heatmap_resized.flatten()

        # Calculate the threshold based on the percentile
        threshold = np.percentile(heatmap_flattened, percentile)

        if verbose:
            print(f'Threshold for class {target_class} (percentile {percentile}): {threshold}')

        # Apply threshold to focus on high attribution regions
        heatmap_thresholded = np.where(heatmap_resized >= threshold, heatmap_resized, 0)

        # Create a green-only heatmap
        green_heatmap = np.zeros_like(image_np)
        green_heatmap[:, :, 1] = (heatmap_thresholded * 255).astype(np.uint8) # Green channel

        # Apply a small Gaussian blur to make dots more visible
        green_heatmap = cv2.GaussianBlur(green_heatmap, (gaussian_blur_size, gaussian_blur_size), 0)

        image_with_heatmap = cv2.addWeighted(green_heatmap, alpha, image_np, 1 - alpha, 0) # Overlay the heatmap on the original image

        image_with_heatmap_pil = ToPILImage()(image_with_heatmap) # Convert the NumPy array back to PIL image

        return image_with_heatmap_pil


    def configure_optimizers(self):
        if self.optimizer is None:
            # If no optimizer is provided, use Adam as default
            self.optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=0.001)
        return self.optimizer
