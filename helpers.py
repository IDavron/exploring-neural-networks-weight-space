import numpy as np
import torch
import matplotlib.pyplot as plt
from models import MLP
import csv
import os

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

def plot_decision_boundary(model, X, y, steps=1000, color_map='Paired', device="cpu"):
    '''
    Plot the decision boundary of a model.

    Parameters:
        X (np.array): The dataset.
        y (np.array): The labels.
        steps (int): The number of steps to take in the meshgrid.
        color_map (str): The color map to use.
        device (str): The device to use.
    '''

    x_min = X[:, 0].min() - 1
    x_max = X[:, 0].max() + 1
    y_min = X[:, 1].min() - 1
    y_max = X[:, 1].max() + 1

    xx, yy = np.meshgrid(np.linspace(x_min, x_max, steps), np.linspace(y_min, y_max, steps))
    X_grid = np.c_[xx.ravel(), yy.ravel()]

    model.to(device)
    model.eval()
    y_boundary = model(torch.from_numpy(X_grid).float().to(device)).detach().numpy().round()
    y_boundary = np.array(y_boundary).reshape(xx.shape)

    color_map = plt.get_cmap(color_map)
    plt.contourf(xx, yy, y_boundary, cmap=color_map, alpha=0.5)
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)

    class_1 = [X[y==0,0], X[y==0,1]]
    class_2 = [X[y==1,0], X[y==1,1]]
    plt.scatter(class_1[0], class_1[1], color=color_map.colors[1], marker='o')
    plt.scatter(class_2[0], class_2[1], color=color_map.colors[11], marker='x')

    plt.legend(["0","1"])
    plt.show()

def load_model(dataset_name: str, angle: int, i: int, datasets_path: str = "Models"):
    '''
    Load model in given dataset, angle and return it.

    Parameters:
        dataset_name (str): The name of the dataset.
        angle (int): The angle on which the model is trained on.
        i (int): The index of the model.
        datasets_path (str): The path to the models.
    '''

    model = MLP()
    model_name = f"model_{angle}_{i}.pth"
    model_path = os.path.join(datasets_path, dataset_name, model_name)
    model.load_state_dict(torch.load(model_path))
    return model

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


def list_to_model(list) -> MLP:
    '''
    Takes a list of weights and biases and assigns them to a model.

    Parameters:
        list (np.array): The list of weights and biases.
    
    Returns:
        model (nn.Module): The model with the weights and biases assigned.
    '''
    model = MLP()
    index = 0
    for param in model.parameters():
        parameters_from_list = list[index:index+param.numel()]
        param.data = torch.tensor(parameters_from_list, dtype=torch.float32).reshape(param.shape)
        index += param.numel()

    return model
