import matplotlib.pyplot as plt
import numpy as np
import torch

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
    plt.close()