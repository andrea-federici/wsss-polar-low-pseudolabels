from typing import Optional

import torch


def generate_heatmap(
    model: torch.nn.Module,
    image: torch.Tensor,
    *,
    target_class: Optional[int] = None,
    layer: Optional[torch.nn.Module] = None,
) -> torch.Tensor:
    """Generate a standard Class Activation Mapping (CAM) heatmap.

    Args:
        model (torch.nn.Module): Model to explain. Must expose a final
            classification layer and optionally a ``get_last_conv_layer``
            method to locate the last convolutional layer when ``layer`` is
            not provided.
        image (torch.Tensor): Input image tensor of shape ``(1, C, H, W)``.
        target_class (int, optional): Class index for which CAM is computed.
            If ``None``, the model's predicted class is used.
        layer (torch.nn.Module, optional): Convolutional layer whose feature
            maps are used to compute CAM. If ``None``, the last convolutional
            layer returned by ``model.get_last_conv_layer()`` is used.

    Returns:
        torch.Tensor: Heatmap tensor of shape ``(H, W)`` with values
            normalised to ``[0, 1]``.

    Raises:
        ValueError: If ``image`` does not have four dimensions or if the
            classification layer cannot be located.
    """
    if image.dim() != 4:
        raise ValueError(
            f"Expected image to be a 4D tensor (1, C, H, W), but got shape {image.shape}"
        )

    was_training = model.training
    model.eval()
    model.zero_grad()

    try:
        if layer is None:
            layer = model.get_last_conv_layer()
        assert layer is not None, "No convolutional layer found. Pass `layer` explicitly."

        # hook to capture the feature maps
        features = {}

        def forward_hook(_, __, output):
            features["value"] = output.detach()

        handle = layer.register_forward_hook(forward_hook)

        # forward pass
        device = next(model.parameters()).device
        image = image.to(device)
        logits = model(image)

        if target_class is None:
            target_class = int(torch.argmax(logits, dim=1).item())

        # locate final linear layer to obtain class weights
        fc_layer = None
        for module in model.modules():
            if isinstance(module, torch.nn.Linear) and module.out_features == logits.shape[1]:
                fc_layer = module
        if fc_layer is None:
            raise ValueError("Could not locate final linear classification layer for CAM.")

        # feature maps from the conv layer
        fmap = features["value"].squeeze(0)
        weight = fc_layer.weight[target_class].detach()

        cam = torch.sum(weight[:, None, None] * fmap, dim=0)
        cam = torch.clamp(cam, min=0)

        # normalise to [0, 1]
        cam_min = cam.min()
        cam_max = cam.max()
        denom = (cam_max - cam_min).clamp_min(1e-6)
        cam_norm = (cam - cam_min) / denom
        return cam_norm

    finally:
        if "handle" in locals():
            handle.remove()
        if was_training:
            model.train()
