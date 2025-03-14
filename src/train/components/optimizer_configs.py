from torch.optim.lr_scheduler import ReduceLROnPlateau

from train_config import lr_patience, lr_factor

def reduce_lr_on_plateau(optimizer):
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=lr_patience, factor=lr_factor)
    return {
        'optimizer': optimizer,
        'lr_scheduler': {
            'scheduler': scheduler,
            'monitor': 'val_loss',
            'interval': 'epoch'
        }
    }
