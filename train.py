import argparse

import train_config as tc
import train_single
import train_optuna
import finetune_iterative

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

    if train_mode == 'single':
        train_single.run()
    elif train_mode == 'single_optuna': 
        train_optuna.run()
    elif train_mode == 'ft_iterative':
        finetune_iterative.run('checkpoints/base_model.ckpt') # TODO: remove hard-coding -> use options
    else:
        raise ValueError(f'Invalid training mode: {train_mode}')
