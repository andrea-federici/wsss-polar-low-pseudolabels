
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
from captum.attr import LayerGradCam

class ModelContainer(pl.LightningModule):
    def __init__(self, model: nn.Module, criterion: nn.Module, optimizer: optim.Optimizer = None):
        super(ModelContainer, self).__init__()
        self.model = model
        self.criterion = criterion # Loss function
        self.optimizer = optimizer # Defaults to Adam if not given as input
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
    
    def get_last_conv_layer(self):
        for layer in reversed(self.model.feature_extractor):
            if isinstance(layer, nn.Conv2d):
                return layer
        raise ValueError("No convolutional layer found in the feature extractor.")

    def generate_gradcam_heatmap(self, input_image, layer=None):
        # If no layer is specified, use the last layer of the feature extractor
        if layer is None:
            layer = self.get_last_conv_layer()

        gradcam = LayerGradCam(self.model, layer) # Initialize GradCAM method

        self.eval()

        input_image = input_image.to(self.device) # Move image to the same device as the model

        logits = self(input_image)
        target_class = torch.argmax(logits, dim=1).item() # Get the predicted class

        attr = gradcam.attribute(input_image, target_class) # Compute the GradCAM attributions
        attr = attr.squeeze().detach().cpu().numpy() # Convert to NumPy format

        heatmap = np.maximum(attr, 0) # Remove negative values in the heatmap
        norm_heatmap = heatmap / np.max(heatmap) if np.max(heatmap) != 0 else 1 # Normalize the heatmap

        return norm_heatmap

    # Visualize the GradCAM overlay on the original image
    def overlay_gradcam_heatmap(self, original_image, heatmap, alpha: float = 0.4):
        image_np = np.array(original_image) # Convert the PIL image to a NumPy array

        heatmap_resized = cv2.resize(heatmap, (image_np.shape[1], image_np.shape[0])) # Resize the heatmap to match the original image size

        heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET) # Apply color map to heatmap
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB) # Convert color space from BGR (Blue, Green, Red) to RGB

        image_with_heatmap = cv2.addWeighted(heatmap_colored, alpha, image_np, 1 - alpha, 0) # Overlay the heatmap on the original image

        image_with_heatmap_pil = ToPILImage()(image_with_heatmap) # Convert the NumPy array back to PIL image

        return image_with_heatmap_pil

    def configure_optimizers(self):
        if self.optimizer is None:
            # If no optimizer is provided, use Adam as default
            self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        return self.optimizer
