import torch
import numpy as np
from src.model.models import MLP
import sklearn.datasets

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
