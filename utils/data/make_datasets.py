import torch

from utils.model.models import MLP
from utils.data.helpers import model_to_list
from utils.config.config import get_default_dirs

import os
import csv
import yaml
from tqdm import tqdm
from argparse import ArgumentParser


def models_to_csv(model_config, input_dir, output_dir = "", file_name="dataset.csv") -> bool:
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
    output_path = os.path.join(output_dir, file_name)
    with(open(output_path, "w")) as f:
        fieldnames = [f"weight_{i}" for i in range(0, 151)]
        fieldnames.insert(0, "model_name")
        fieldnames.append("angle")
        writer = csv.writer(f, lineterminator = '\n')
        writer.writerow(fieldnames)

        models = os.listdir(input_dir)
        for model in tqdm(models):
            angle = int(model.split("_")[1])
            m = MLP(model_config["INPUT_DIM"], model_config["HIDDEN_DIMS"], model_config["OUTPUT_DIM"])
            m.load_state_dict(torch.load(f"{input_dir}/{model}"))
            weights = model_to_list(m)
            row = weights.tolist()
            row.append(angle)
            row.insert(0, model)
            writer.writerow(row)
    return True

if __name__ == "__main__":
    parser = ArgumentParser(description='Save model weights and biases into single csv file.')
    parser.add_argument('-c', '--config', help='Path to the config file.', required=True)
    parser.add_argument('-i', '--input', help='Folder containing all models', required=True)
    parser.add_argument('-o', '--output', help='Path to the config file.')
    parser.add_argument('-n', '--name', help='Name of the file')
    args = parser.parse_args()

    default_dirs = get_default_dirs()
    config_path = args.config
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    
    model_config = config["MODEL"]
    
    if(args.name is None and args.output is None):
        models_to_csv(model_config, args.input)
    elif(args.name is None):
        models_to_csv(model_config, args.input, args.output)
    elif(args.output is None):
        models_to_csv(model_config, args.input, file_name=args.name)
    else:
        models_to_csv(model_config, args.input, args.output, args.name)