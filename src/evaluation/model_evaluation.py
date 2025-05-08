import torch
from tqdm import tqdm


def evaluate_model(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device = "cpu",
):
    """
    Evaluates the given model on the provided dataloader.

    Args:
        model (torch.nn.Module): The PyTorch model to evaluate.
        dataloader (torch.utils.data.DataLoader): The dataloader for the validation/test dataset.
        device (torch.device): The device on which to perform evaluation (e.g., 'cpu' or 'cuda').

    Returns:
        all_preds (list): List of predictions for the entire dataset.
        all_labels (list): List of true labels for the entire dataset.
    """
    # Set model to evaluation mode
    model.eval()
    all_preds = []
    all_labels = []

    # Disable gradient computation
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating model"):
            images, labels = batch
            images, labels = images.to(device), labels.to(device)

            # Forward pass
            logits = model(images)

            # Get predictions
            preds = torch.argmax(logits, dim=1)

            # Store predictions and labels
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return all_preds, all_labels
