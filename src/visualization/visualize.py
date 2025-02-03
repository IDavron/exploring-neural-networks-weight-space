import matplotlib.pyplot as plt
import numpy as np
import torch

def plot_decision_boundary(model, X, y, steps=1000, color_map='Paired', axis=True, title=None, save_path=None, legend=True):
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
    X_grid = torch.from_numpy(X_grid).float()

    y_boundary = model(X_grid).detach().numpy().round()
    y_boundary = np.array(y_boundary).reshape(xx.shape)

    color_map = plt.get_cmap(color_map)
    plt.contourf(xx, yy, y_boundary, cmap=color_map, alpha=0.5)
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)

    class_1 = [X[y==0,0], X[y==0,1]]
    class_2 = [X[y==1,0], X[y==1,1]]
    plt.scatter(class_1[0], class_1[1], color=color_map.colors[1], marker='o')
    plt.scatter(class_2[0], class_2[1], color=color_map.colors[11], marker='x')

    if(legend):
        plt.legend(["0","1"])

    if title:
        plt.title(title)
    
    if not axis:
        # remove ticks
        plt.xticks([])
        plt.yticks([])

    if save_path:
        plt.savefig(save_path)
    plt.show()
    plt.close()


# def plot_interpolation(model, datapoint_1, datapoint_2, X, y, alpha):
#     parameters_1, angle_1 = datapoint_1
#     parameters_2, angle_2 = datapoint_2

#     parameters_1 = parameters_1.unsqueeze(0)
#     parameters_2 = parameters_2.unsqueeze(0)

#     latent_1 = model.encoder(parameters_1)
#     latent_2 = model.encoder(parameters_2)

#     latent = (1-alpha)*latent_1 + alpha*latent_2
#     w = model.decoder(latent).squeeze()

#     angle = (1-alpha)*angle_1 + alpha*angle_2
#     X_rotated = rotate(X, angle)
#     X_rotated = torch.tensor(X_rotated).float()

#     plot_decision_boundary(w, X_rotated, y)