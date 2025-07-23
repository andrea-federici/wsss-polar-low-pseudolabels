import torch


def get_layer_from_name(model: torch.nn.Module, layer_name: str) -> torch.nn.Module:
    """
    Retrieve a specific layer from a PyTorch model by its name.

    Args:
        model (torch.nn.Module): The PyTorch model containing the layers.
        layer_name (str): The name of the layer to retrieve.

    Returns:
        torch.nn.Module or None: The requested layer if found; otherwise,
        None.
    """
    target_layer = None
    if layer_name:
        # Get all named layers, along with their names, and store them in a
        # dictionary. The key is the name of the layer, and the value is the
        # layer itself.
        named_modules = dict(model.named_modules())

        # Fetch the layer by name from the dictionary
        target_layer = named_modules.get(layer_name)

        if target_layer:
            return target_layer

    return None
