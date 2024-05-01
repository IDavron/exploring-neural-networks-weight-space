import torch
import numpy as np
from utils.model.models import MLP

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


def list_to_model(list, model_config) -> MLP:
    '''
    Takes a list of weights and biases and assigns them to a model.

    Parameters:
        list (np.array): The list of weights and biases.
    
    Returns:
        model (nn.Module): The model with the weights and biases assigned.
    '''
    model = MLP(model_config["INPUT_DIM"], model_config["HIDDEN_DIMS"], model_config["OUTPUT_DIM"])
    index = 0
    for param in model.parameters():
        parameters_from_list = list[index:index+param.numel()]
        param.data = torch.tensor(parameters_from_list, dtype=torch.float32).reshape(param.shape)
        index += param.numel()

    return model
