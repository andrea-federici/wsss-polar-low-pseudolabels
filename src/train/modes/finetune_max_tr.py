
from datetime import datetime

from models import Xception
from train.setup import create_max_tr_model_ckpt_setup, create_model_ckpt_setup
from trainers import create_trainer
import train.loggers as loggers
import train_config as tc
from src.train.helpers.max_tr_helper import batch_find_maximum_translations, average_bounding_box_area

import json


def run(base_model_ckpt_path: str):
    neptune_logger = loggers.neptune_logger

    neptune_logger.experiment["source_files/train_config"].upload("train_config.py")

    max_translations = None
    average_areas = []

    for iteration in range(0, tc.max_iterations+1):
        neptune_logger.experiment[f"iteration_{iteration}/start_time"] = datetime.now().isoformat() 

        torch_model = Xception(num_classes=2)
        if iteration == 0:
            lit_model, train_loader, val_loader = create_model_ckpt_setup(torch_model, base_model_ckpt_path)
        else:
            lit_model, train_loader, val_loader = create_max_tr_model_ckpt_setup(
                torch_model,
                base_model_ckpt_path,
                max_translations
            )

            trainer = create_trainer(neptune_logger)
            trainer.fit(lit_model, train_loader, val_loader)
        
        max_translations = batch_find_maximum_translations(lit_model, 'data/train/pos', tc.transform_prep) # TODO: fix this to dynamically change the path
        average_areas.append(average_bounding_box_area(max_translations, tc.resized_image_res))

        max_translation_file = f"max_translations_{iteration}.json"
        with open(max_translation_file, "w") as f:
            json.dump(max_translations, f)
        neptune_logger.experiment[f"iteration_{iteration}/max_translations_file"].upload(max_translation_file)

        neptune_logger.experiment[f"iteration_{iteration}/average_area"] = average_areas[-1]
        print(f"Average area for iteration {iteration}: {average_areas[-1]}")
