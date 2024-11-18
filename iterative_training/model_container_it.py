
import numpy as np
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score
)
import torch
from torch import nn
import torch.optim as optim
import pytorch_lightning as pl
import torchvision.transforms.functional as F


class ModelContainerIt(pl.LightningModule):

    def __init__(self, model: nn.Module, criterion: nn.Module, optimizer: optim.Optimizer, custom_max_translations = None):
        super(ModelContainerIt, self).__init__()
        self.model = model
        self.criterion = criterion # Loss function
        self.optimizer = optimizer # Optimizer
        self.max_translations = custom_max_translations
        self.train_losses = [] # Stores train losses for each epoch
        self.train_outputs = [] # Stores training outputs across steps
        self.val_losses = [] # Stores validation losses for each epoch
        self.val_outputs = [] # Stores validation outputs across steps

    
    def custom_transform(self, image, max_translations):
        if max_translations is not None:
            # Apply random translation within the limits
            x_tr = torch.randint(-max_translations['left'], max_translations['right']+1, size=(1,)).item()
            y_tr = torch.randint(-max_translations['up'], max_translations['down']+1, size=(1,)).item()
        else:
            h, w = image.shape[1], image.shape[2]
            x_tr = torch.randint(low=int(-0.35*w), high=int(0.35*w)+1, size=(1,)).item()
            y_tr = torch.randint(low=int(-0.35*h), high=int(0.35*h)+1, size=(1,)).item()

        translated_image = F.affine(
            image,
            angle=0,
            translate=(x_tr, y_tr),
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

        return rotated_image


    # The 'forward' method is called automatically when the model is invoked on input data
    def forward(self, x):
        return self.model(x) # self.model(x) calls the 'forward' method of the contained model


    # Called for each batch of data
    def training_step(self, batch, batch_idx):
        images, labels, max_translations = batch # Unpack the batch
        
        transformed_images = [
            self.custom_transform(img, max_tr) for img, max_tr in zip(images, max_translations)
        ]
        transformed_images = torch.stack(transformed_images).to(self.device)
        labels = labels.to(self.device)

        logits = self(transformed_images) # Forward pass to get predictions
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
        images, labels, _ = batch
        images, labels = images.to(self.device), labels.to(self.device)
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

        return loss


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


    def configure_optimizers(self):
        if self.optimizer is None:
            # self.model.parameters() returns an iterator over all model parameters (weights and biases)
            # Each parameter is a tensor, and each tensor has a 'requires_grad' attribute, which is True if the parameter should be updated during backpropagation, and False otherwise.
            # filter() is a function that takes a function and an iterable as input. It applies the function to each element of the iterable and returns only the elements for which the function returns True.
            # This line filters out the parameters that should not be updated during backpropagation (frozen layers)
            self.optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=0.001)
        return self.optimizer


