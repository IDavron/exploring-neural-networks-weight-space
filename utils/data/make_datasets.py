import torch

from utils.model.models import MLP
from utils.data.helpers import model_to_list

import os
import csv
import yaml
from tqdm import tqdm


def zoo_to_csv(input_dir: str, output_path: str) -> None:
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

    # Model config
    input_dim = 2
    hidden_dims = [10,10]
    output_dim = 1

    with(open(output_path, "w")) as f:
        fieldnames = [f"weight_{i}" for i in range(0, 151)]
        fieldnames.insert(0, "model_name")
        fieldnames.append("angle")
        writer = csv.writer(f, lineterminator = '\n')
        writer.writerow(fieldnames)

        models = os.listdir(input_dir)
        for model in tqdm(models):
            angle = int(model.split("_")[1])
            m = MLP(input_dim, hidden_dims, output_dim)
            m.load_state_dict(torch.load(f"{input_dir}/{model}"))
            weights = model_to_list(m)
            row = weights.tolist()
            row.append(angle)
            row.insert(0, model)
            writer.writerow(row)

if __name__ == "__main__":
    with open(r"configs/datasets/moons.yaml", "r") as file:
        config = yaml.safe_load(file)
        make_moons(config)