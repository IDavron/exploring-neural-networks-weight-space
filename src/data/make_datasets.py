import torch

from src.data.helpers import model_to_list
import numpy as np

import os
import csv
from tqdm import tqdm


def zoo_to_csv(input_dir: str, output_path: str, model) -> None:
    '''
    Save all model weights from a dataset into single csv file.

    Parameters:
        model_config (dict): The config file containing model structure used to train the zoo.
        input_dir (str): The directory where models are saved.
        output_dir (str): The directory where the csv file will be saved.
        file_name (str): Name of the csv file.
    Returns:
        bool: True if the csv file was created, False otherwise.
    '''
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    N = sum([np.prod(p.size()) for p in model_parameters])

    with(open(output_path, "w")) as f:
        fieldnames = [f"weight_{i}" for i in range(0, N)]
        fieldnames.insert(0, "model_name")
        fieldnames.append("angle")
        writer = csv.writer(f, lineterminator = '\n')
        writer.writerow(fieldnames)

        models = os.listdir(input_dir)
        for model_params in tqdm(models):
            angle = int(model_params.split("_")[1])
            model.load_state_dict(torch.load(f"{input_dir}/{model_params}"))
            weights = model_to_list(model)
            row = weights.tolist()
            row.append(angle)
            row.insert(0, model_params)
            writer.writerow(row)
