import torch
import numpy as np
from src.model.models import MLP
import sklearn.datasets

import json
import logging
import os
from collections import defaultdict
from pathlib import Path

from src.data.datasets import ModelDataset, Batch


def get_moons_dataset(n_samples: int = 1000, noise: float = 0.1, random_state=42, normalize: bool = True) -> tuple:
    X,y = sklearn.datasets.make_moons(n_samples=n_samples, noise=noise, random_state=random_state)
    if(normalize):
        X = (X - X.mean(axis=0)) / X.std(axis=0)
    return X, y

def rotate(X, angle: int):
    '''
    Rotate the dataset X by the angle.

    Parameters:
        X (np.array): The dataset.
        angle (float): The angle to rotate by.
    '''

    rad = np.radians(angle)
    R = np.array([[np.cos(rad), -np.sin(rad)], [np.sin(rad), np.cos(rad)]])
    X_rotated = X.dot(R)
    return X_rotated

def model_to_list(model) -> np.array:
    '''
    Takes all the weights and biases of a model and returns them as a list.

    Parameters:
        model (nn.Module): The model to extract the parameters from.
    '''
    trainable_parameters = np.array([])
    for param in model.parameters():
        trainable_parameters = np.append(trainable_parameters, param.data.flatten().numpy())
    
    return trainable_parameters


def list_to_model(model, list) -> None:
    '''
    Takes a list of weights and biases and assigns them to a model.

    Parameters:
        list (np.array): The list of weights and biases.
    
    Returns:
        model (nn.Module): The model with the weights and biases assigned.
    '''
    index = 0
    for param in model.parameters():
        parameters_from_list = list[index:index+param.numel()]
        param.data = torch.tensor(parameters_from_list, dtype=torch.float32).reshape(param.shape)
        index += param.numel()

def mlp_from_config(model_config: dict) -> MLP:
    input_dim = model_config["input_dim"]
    hidden_dims = model_config["hidden_dims"]
    output_dim = model_config["output_dim"]
    dropout = model_config.get("dropout", 0.0)
    use_batch_norm = model_config.get("use_batch_norm", False)
    output_activation = model_config.get("output_activation", "softmax")

    model = MLP(input_dim, hidden_dims, output_dim, dropout, use_batch_norm, output_activation)
    return model

def get_accuracy(model, X, y):
    '''
    Get the accuracy of a Moons classifier on a moons dataset.
    '''
    y_pred = model(torch.tensor(X).float()).squeeze().round().detach().numpy()
    correct = (y_pred == y).sum()
    accuracy = correct / len(y) * 100
    return accuracy

def generate_splits(data_path, save_path, name="dataset_splits.json", val_size=0):
    '''
    Generate a json file containing paths of all saved trained models. 
    This file is used to create a dataset and dataloader later.
    '''
    save_path = Path(save_path) / name
    inr_path = Path(data_path)
    data_split = defaultdict(lambda: defaultdict(list))
    for p in list(inr_path.glob("*.pth")):
        angle = p.stem.split("_")[-2]
        data_split["train"]["path"].append((os.getcwd() / p).as_posix())
        data_split["train"]["angle"].append(angle)

    logging.info(
        f"train size: {len(data_split['train']['path'])}, "
        f"val size: {len(data_split['val']['path'])}, test size: {len(data_split['test']['path'])}"
    )

    with open(save_path, "w") as file:
        json.dump(data_split, file)


def compute_stats(data_path: str, save_path: str, batch_size: int = 10000):
    '''
    Compute the mean and standard deviation of the weights and biases of a dataset. 
    Needed later to normalize the data.
    '''
    train_set = ModelDataset(path=data_path, split="train")
    train_loader = torch.utils.data.DataLoader(
        dataset=train_set, batch_size=batch_size, shuffle=True, num_workers=8
    )

    batch: Batch = next(iter(train_loader))
    weights_mean = [w.mean(0) for w in batch.weights]
    weights_std = [w.std(0) for w in batch.weights]
    biases_mean = [w.mean(0) for w in batch.biases]
    biases_std = [w.std(0) for w in batch.biases]

    statistics = {
        "weights": {"mean": weights_mean, "std": weights_std},
        "biases": {"mean": biases_mean, "std": biases_std},
    }

    out_path = Path(save_path)
    out_path.mkdir(exist_ok=True, parents=True)
    torch.save(statistics, out_path / "statistics.pth")


if __name__ == "__main__":
    # generate_splits("models/eight_angles_small", "data")
    compute_stats("data/dataset_splits.json", "data")
