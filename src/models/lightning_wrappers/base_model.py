import torch
from lightning.pytorch import LightningModule
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch import nn


class BaseModel(LightningModule):

    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        optimizer_config: dict,
        *,
        multi_label: bool = False,
        threshold: float = 0.5,
    ):
        super().__init__()
        self.model = model
        self.criterion = criterion  # Loss function
        self.optimizer_config = optimizer_config  # Optimizer configuration
        self.multi_label = multi_label
        self.threshold = threshold

        # This is needed so that we don't need to pass all the hyperparameters when
        # loading the checkpoint
        self.save_hyperparameters()

        self.train_outputs = []  # Stores training outputs across steps
        self.val_outputs = []  # Stores validation outputs across steps

    # The 'forward' method is called automatically when the model is invoked on input data
    def forward(self, x):
        return self.model(
            x
        )  # self.model(x) calls the 'forward' method of the contained model

    def _process_step(self, stage: str, images, labels):
        if stage not in {"train", "val"}:
            raise ValueError(f"Invalid stage: {stage}. Must be 'train' or 'val'.")

        batch_size = images.size(0)

        logits = self(images)  # Forward pass to get predictions
        loss = self.criterion(logits, labels)  # Calculate loss

        if self.multi_label:
            preds = (torch.sigmoid(logits) > self.threshold).int()
        else:
            preds = torch.argmax(logits, dim=1)

        output = {
            "preds": preds.cpu().tolist(),
            "labels": labels.cpu().tolist(),
            "loss": loss.item(),
        }

        # Append to the appropriate output list
        outputs_attr = f"{stage}_outputs"
        if hasattr(self, outputs_attr):
            getattr(self, outputs_attr).append(output)
        else:
            raise AttributeError(f"The attribute '{outputs_attr}' does not exist.")

        self.log(
            f"{stage}/batch_loss",
            loss,
            on_step=True,
            on_epoch=True,
            batch_size=batch_size,
            prog_bar=True,
        )

        return loss

    def _process_epoch_end(self, stage: str):
        if stage not in {"train", "val"}:
            raise ValueError(f"Invalid stage: {stage}. Must be 'train' or 'val'.")

        outputs_attr = f"{stage}_outputs"
        if not hasattr(self, outputs_attr):
            raise AttributeError(f"The attribute '{outputs_attr}' does not exist.")

        outputs = getattr(self, outputs_attr)

        all_preds = []  # Collects all predictions from the epoch
        all_labels = []  # Collects all labels from the epoch
        total_loss = 0.0  # Tracks total loss across batches

        for output in outputs:
            all_preds.extend(output["preds"])
            all_labels.extend(output["labels"])
            total_loss += output["loss"]

        avg_loss = total_loss / len(outputs)  # Average loss for the epoch
        if self.multi_label:
            accuracy = accuracy_score(all_labels, all_preds)
            precision = precision_score(
                all_labels, all_preds, average="micro", zero_division=0
            )
            recall = recall_score(
                all_labels, all_preds, average="micro", zero_division=0
            )
            f1 = f1_score(
                all_labels, all_preds, average="micro", zero_division=0
            )
        else:
            accuracy = accuracy_score(all_labels, all_preds)
            precision = precision_score(all_labels, all_preds, average="binary")
            recall = recall_score(all_labels, all_preds, average="binary")
            f1 = f1_score(all_labels, all_preds, average="binary")

        self.log(f"{stage}/epoch_loss", avg_loss, on_epoch=True, prog_bar=True)
        self.log(f"{stage}/epoch_accuracy", accuracy, on_epoch=True, prog_bar=True)
        self.log(f"{stage}/epoch_precision", precision, on_epoch=True, prog_bar=True)
        self.log(f"{stage}/epoch_recall", recall, on_epoch=True, prog_bar=True)
        self.log(f"{stage}/epoch_f1", f1, on_epoch=True, prog_bar=True)

        if stage == "val":
            self.log("val_loss", avg_loss, on_epoch=True, prog_bar=True)

        print(f"Epoch {self.current_epoch} - {stage.capitalize()}:")
        print(
            f"Loss: {avg_loss:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}"
        )

        outputs.clear()  # Clear the outputs for the next epoch

    # Called for each batch of data
    def training_step(self, batch, batch_idx):
        images, labels = batch  # Unpack the batch
        loss = self._process_step("train", images, labels)
        return loss  # Return the loss for optimization

    # Called at the end of every training epoch to aggregate metrics and print them
    def on_train_epoch_end(self):
        self._process_epoch_end("train")

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        loss = self._process_step("val", images, labels)
        return loss

    # At the end of each validation epoch, this method aggregates the outputs from all validation batches
    def on_validation_epoch_end(self):
        self._process_epoch_end("val")

    def configure_optimizers(self):
        return self.optimizer_config
