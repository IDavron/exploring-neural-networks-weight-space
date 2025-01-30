import torch
import torch.nn as nn

import numpy as np
import sklearn.datasets

from src.model.models import MLP
from src.data.datasets import ModelDataset, Batch

import json
import logging
import os
from collections import defaultdict
from pathlib import Path
from tqdm import tqdm


def get_moons_dataset(angle=0, n_samples: int = 1000, noise: float = 0.1, random_state=42, normalize: bool = True) -> tuple:
    X, y = sklearn.datasets.make_moons(n_samples=n_samples, noise=noise, random_state=random_state)
    if(normalize):
        X = (X - X.mean(axis=0)) / X.std(axis=0)
    rad = np.radians(angle)
    R = np.array([[np.cos(rad), -np.sin(rad)], [np.sin(rad), np.cos(rad)]])
    X = X.dot(R)

    return X, y


def generate_model_zoo(zoo_name, input_dim, hidden_dims, output_dim, angles=(0, 45, 90, 135, 180, 225, 270, 315), models_per_angle=10000, learning_rate=0.05, epochs=60, accuracy_thershold=0.95, save_dir="../models/", seed=42, device=None):
    if(device is None):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Two-moons dataset
    criterion = nn.BCELoss()

    save_path = Path(save_dir) / zoo_name
    save_path.mkdir(exist_ok=True)

    torch.manual_seed(seed)
    model_accuracies = []

    for angle in tqdm(angles):
        X, y = get_moons_dataset(angle)

        X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
        y_tensor = torch.tensor(y, dtype=torch.float32).to(device)
        i = 0
        while i < models_per_angle:
            model = MLP(input_dim=input_dim, hidden_dims=hidden_dims, output_dim=output_dim)
            model.to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

            for _ in range(epochs):
                optimizer.zero_grad()
                y_pred = model(X_tensor).flatten()
                loss = criterion(y_pred, y_tensor)
                loss.backward()
                optimizer.step()
            
            model.eval()
            y_pred = model(X_tensor).flatten()
            correct = (y_pred.round() == y_tensor).sum().item()
            accuracy = correct / len(y)
            
            if(accuracy >= 0.95):
                model_name = f"model_{angle}_{i}.pth"
                model_path = save_path / model_name
                torch.save(model.state_dict(), model_path)
                model_accuracies.append((model_name, accuracy))
                i += 1
    
    return model_accuracies


# Source code from Equivariant Architectures for Learning in Deep Weight Spaces
# https://github.com/AvivNavon/DWSNets

def generate_splits(models_path, save_path, name="dataset_splits.json", total_size = 10000, val_size=0, test_size = 0):
    '''
    Generate a json file containing paths of all saved trained models. 
    This file is used to create a dataset and dataloader later.
    '''
    save_path = Path(save_path) / name
    models_path = Path(models_path)
    data_split = defaultdict(lambda: defaultdict(list))
    for i, p in enumerate(list(models_path.glob("*.pth"))):
        angle = p.stem.split("_")[-2]
        if(i % total_size >= total_size - val_size):
            s = "val"
        elif((i % total_size >= total_size - val_size - test_size) and (i % total_size < total_size - val_size)):
            s = "test"
        else:
            s = "train"

        data_split[s]["path"].append((os.getcwd() / p).as_posix())
        data_split[s]["angle"].append(angle)

    logging.info(
        f"train size: {len(data_split['train']['path'])}, "
        f"val size: {len(data_split['val']['path'])}, test size: {len(data_split['test']['path'])}"
    )

    with open(save_path, "w") as file:
        json.dump(data_split, file)

def compute_stats(data_path: str, save_path: str, batch_size: int = 10000, name="statistics.pth"):
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
    torch.save(statistics, out_path / name)
