import argparse

import torch

import train_config as tc
import train.modes.train_single
import train.modes.train_optuna
import train.modes.finetune_iterative
import train.modes.train_adversarial_erasing

# Suppress the warning related to the creation of DataLoader using a high 
# number of num_workers
import warnings
warnings.filterwarnings('ignore', message='.*DataLoader will create.*')


if __name__ == "__main__":
    parser = argparse.ArgumentParser() # TODO: change _ to - for all arguments and their options
    parser.add_argument(
        '--data-dir',
        type=str,
        default=tc.default_data_dir,
        help=f'Path to the data directory (default: {tc.default_data_dir})',
    )
    parser.add_argument(
        '--train_mode', # TODO: change to train-mode
        type=str,
        choices=tc.possible_train_modes,
        default=tc.default_train_mode,
        help=f'Training mode (Possible options: {tc.possible_train_modes}. \
            Default: {tc.default_train_mode})',
    )

    args = parser.parse_args()
    tc.default_data_dir = args.data_dir
    train_mode = args.train_mode

    # Enable Tensor Cores for faster matrix multiplications, trading some 
    # precision for performance.
    # - 'high' gives the best speed-up but with reduced precision.
    # - 'medium' balances precision and performance.
    # - 'highest' (default) maintains full precision but may be slower.
    torch.set_float32_matmul_precision('medium')

    if train_mode == 'single':
        train_single.run()
    elif train_mode == 'single_optuna': 
        train_optuna.run()
    elif train_mode == 'ft_iterative':
        finetune_iterative.run('checkpoints/base_model.ckpt') # TODO: remove hard-coding -> use options
    elif train_mode == 'adv':
        train_adversarial_erasing.run()
    else:
        raise ValueError(f'Invalid training mode: {train_mode}')
