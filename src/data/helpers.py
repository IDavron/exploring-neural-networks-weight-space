import torch
import torch.nn.functional as F

import numpy as np
import sklearn.datasets

from src.model.models import MLP, DBModel
from src.data.datasets import ModelDataset, Batch

import json
import logging
import os
import math
from collections import defaultdict
from pathlib import Path


gaussian = torch.distributions.multivariate_normal.MultivariateNormal(torch.zeros(33), math.sqrt(1) * torch.eye(33))

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

def get_accuracy(parameters, X, y):
    '''
    Get the accuracy of a Moons classifier on a moons dataset.
    '''
    model = DBModel(False)
    y_pred = model(parameters, X).squeeze().detach().round().numpy()
    correct = (y_pred == y).sum()
    accuracy = correct / len(y) * 100
    return accuracy


# Source code from Equivariant Architectures for Learning in Deep Weight Spaces
# https://github.com/AvivNavon/DWSNets

def generate_splits(data_path, save_path, name="dataset_splits.json", total_size = 10000, val_size=0, test_size = 0):
    '''
    Generate a json file containing paths of all saved trained models. 
    This file is used to create a dataset and dataloader later.
    '''
    save_path = Path(save_path) / name
    inr_path = Path(data_path)
    data_split = defaultdict(lambda: defaultdict(list))
    for i, p in enumerate(list(inr_path.glob("*.pth"))):
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


# For diffusion model
def add_noise(x_0, noise, alphas_cumprod, t):
    sqrt_alphas_cumprod = alphas_cumprod ** 0.5
    sqrt_one_minus_alphas_cumprod = (1 - alphas_cumprod) ** 0.5

    s1 = sqrt_alphas_cumprod[t]
    s2 = sqrt_one_minus_alphas_cumprod[t]

    s1 = s1.reshape(-1, 1)
    s2 = s2.reshape(-1, 1)

    return s1*x_0 + s2*noise

def reconstruct_xt(noise, x_t, t, betas):
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.)

        sqrt_inv_alphas_cumprod = torch.sqrt(1 / alphas_cumprod)
        sqrt_inv_alphas_cumprod_minus_one = torch.sqrt(1 / alphas_cumprod - 1)

        posterior_mean_coef1 = betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)
        posterior_mean_coef2 = (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod)

        s1 = sqrt_inv_alphas_cumprod[t]
        s2 = sqrt_inv_alphas_cumprod_minus_one[t]
        s1 = s1.reshape(-1, 1)
        s2 = s2.reshape(-1, 1)
        pred_x0 = s1 * x_t - s2 * noise

        posterior_mean_coef1 = betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)
        posterior_mean_coef2 = (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod)
        s1 = posterior_mean_coef1[t]
        s2 = posterior_mean_coef2[t]
        s1 = s1.reshape(-1, 1)
        s2 = s2.reshape(-1, 1)
        x_t = s1 * pred_x0 + s2 * x_t

        variance = 0
        if t > 0:
            variance = betas[t] * (1. - alphas_cumprod_prev[t]) / (1. - alphas_cumprod[t])
            variance = variance.clip(1e-20)
            noise = torch.randn_like(noise)
            variance = (variance ** 0.5) * noise

        pred_prev_sample = x_t + variance

        return pred_prev_sample


# Generation functions
def generate_flow(model, angle, prior_dim=33, prior_sd=1, num_iter=100):
    gaussian = torch.distributions.multivariate_normal.MultivariateNormal(torch.zeros(prior_dim), math.sqrt(prior_sd) * torch.eye(prior_dim))

    x = gaussian.sample((1,))
    angle = torch.tensor([angle*torch.pi/180])
    for i in np.linspace(0, 1, num_iter, endpoint=False):
        t = torch.tensor([i], dtype=torch.float32)
        sin = torch.sin(angle)
        cos = torch.cos(angle)
        path = model(torch.cat([x, t[:, None], sin[:, None], cos[:, None]], dim=-1))
        x += (0.01 * path)
    return x.detach()[0]

def generate_diffusion(model, angle, num_timesteps=1000, betas=None):
    if(betas is None):
        betas = torch.tensor(np.linspace(1e-4, 0.02, num_timesteps), dtype=torch.float32)
    else:
        betas = betas

    a = torch.tensor([angle * np.pi / 180])
    sin = torch.sin(a)
    cos = torch.cos(a)
    a = torch.cat([sin[None, :], cos[None, :]], dim=1)

    sample = torch.randn(1, 33)
    timesteps = list(range(num_timesteps))[::-1]
    for i, t in enumerate(timesteps):
        t = torch.from_numpy(np.repeat(t, 1)).long()
        with torch.no_grad():
            residual = model(sample, t, a)
        sample = reconstruct_xt(residual, sample, t[0], betas)

    return sample[0]

if __name__ == "__main__":
    # generate_splits("models/eight_angles_small", "data")
    compute_stats("data/dataset_splits.json", "data")
