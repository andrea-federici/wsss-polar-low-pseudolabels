
import optuna

from models import XceptionModel
from train_setups import create_standard_setup
from trainers import create_trainer
import loggers
import train_config as tc


def run():
    # Create Optuna study
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=50, n_jobs=1) # Setting n_jobs=1
    # makes the studies run sequentially and not simultaneously.

    # Print best hyperparameters
    print(f'Best hyperparameters: {study.best_params}')


def objective(trial):
    # Suggest hyperparameters
    # learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-2)
    # batch_size = trial.suggest_categorical('batch_size', [32, 64, 128])
    aug_translate_frac = trial.suggest_float("aug_translate_frac", 0.2, 0.7)

    # tc.learning_rate = learning_rate
    # tc.batch_size = batch_size
    tc.aug_translate_frac = aug_translate_frac

    neptune_logger = loggers.neptune_logger

    neptune_logger.experiment["source_files/train_config"].upload("train_config.py")

    # Log parameters before training
    neptune_logger.experiment["parameters"] = {
        "aug_translate_frac": aug_translate_frac,
    }

    torch_model = XceptionModel(num_classes=2)

    lit_model, train_loader, val_loader = create_standard_setup(torch_model)

    # Train model
    trainer = create_trainer(
        neptune_logger,
        using_optuna=True,
        optuna_trial=trial
    )
    trainer.fit(lit_model, train_loader, val_loader)

    return trainer.callback_metrics['val_loss'].item()