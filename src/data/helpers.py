import torch
import numpy as np
from src.model.models import MLP


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
    Get the accuracy of a model.
    
    Args:
        model (nn.Module): The model to evaluate (DBModels).
        X (torch.Tensor): Two-moons dataset coordinates.
        y (torch.Tensor): Two-moons dataset labels.
    '''
    y_pred = model(X).detach().squeeze().round().numpy()
    correct = (y_pred == y).sum()
    accuracy = correct / len(y) * 100
    return accuracy


# For diffusion model
def add_noise(x_0, noise, alphas_cumprod, t):
    sqrt_alphas_cumprod = alphas_cumprod ** 0.5
    sqrt_one_minus_alphas_cumprod = (1 - alphas_cumprod) ** 0.5

    s1 = sqrt_alphas_cumprod[t]
    s2 = sqrt_one_minus_alphas_cumprod[t]

    s1 = s1.reshape(-1, 1)
    s2 = s2.reshape(-1, 1)

    return s1*x_0 + s2*noise

def find_closest_vectors(X, Y):
    distances = torch.cdist(X, Y)
    closest_vals, closest_indices = distances.min(dim=1)
    closest_vectors = Y[closest_indices]

    return closest_vals, closest_vectors