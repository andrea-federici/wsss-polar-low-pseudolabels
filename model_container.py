import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)
import torch
from torch import nn
import torch.optim as optim
from pytorch_lightning import LightningModule

from optimizer_configs import reduce_lr_on_plateau

class ModelContainer(LightningModule):

    def __init__(self, model: nn.Module, criterion: nn.Module, optimizer: optim.Optimizer):
        super(ModelContainer, self).__init__()
        self.model = model
        self.criterion = criterion # Loss function
        self.optimizer = optimizer # Defaults to Adam if not given as input

        self.train_outputs = [] # Stores training outputs across steps
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

        self.log('train_loss', loss, prog_bar=True)

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
        accuracy = accuracy_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds, average='binary')
        recall = recall_score(all_labels, all_preds, average='binary')
        f1 = f1_score(all_labels, all_preds, average='binary')

        self.log('train/epoch_loss', avg_loss)
        self.log('train/epoch_accuracy', accuracy)
        self.log('train/epoch_precision', precision)
        self.log('train/epoch_recall', recall)
        self.log('train/epoch_f1', f1)

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
        accuracy = accuracy_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds, average='binary')
        recall = recall_score(all_labels, all_preds, average='binary')
        f1 = f1_score(all_labels, all_preds, average='binary')

        self.log('val/epoch_loss', avg_loss)
        self.log('val/epoch_accuracy', accuracy)
        self.log('val/epoch_precision', precision)
        self.log('val/epoch_recall', recall)
        self.log('val/epoch_f1', f1)

        print(f"Epoch {self.current_epoch} - Validation:")
        print(f"Loss: {avg_loss:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")

        self.val_outputs.clear() # Clear validation outputs for the next epoch


    def configure_optimizers(self):
        return reduce_lr_on_plateau(self.optimizer)
