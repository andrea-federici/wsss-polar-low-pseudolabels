from torch.optim.lr_scheduler import ReduceLROnPlateau

# TODO: add metric to monitor and mode to the function signature
def reduce_lr_on_plateau(optimizer, lr_patience, lr_factor):
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=lr_patience, factor=lr_factor)
    return {
        'optimizer': optimizer,
        'lr_scheduler': {
            'scheduler': scheduler,
            'monitor': 'val_loss',
            'interval': 'epoch'
        }
    }
