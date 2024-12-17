#################################################################################################
#   Script to train MLP model zoo on moons dataset with different angles and save them.         #
#   Allows to log model accuracies into WandB. For that WANDB_PROJECT and WANDB_ENTITY fields   #
#   must be present in .env file.                                                               #
#################################################################################################

import torch
import torch.nn as nn

from tqdm import tqdm
import wandb

from src.model.models import MLP

from pathlib import Path
from src.data.helpers import rotate, get_moons_dataset

def train_zoo(angles: list, models_per_angle: int, output_dir: str) -> list:
    '''
    Train a zoo of MLP models.\n
    MLP structure:\n
        - input_dim = 2
        - hidden_dims = [10, 10]
        - output_dim = 1
    
    Parameters:
        angles (int): Angles of the moons dataset to train models on.
        models_per_angle (int): Number of models for each angle.
        output_dir (str): The directory to save trained models.
    
    Return:
        The list of (model_name, model_accuracy) for each model.
    '''
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    output_dir = Path(output_dir)

    # Hyperparameters
    epochs = 60
    learning_rate = 0.05
    seed = 42
    # Model config
    input_dim = 2
    hidden_dims = [10, 10]
    output_dim = 1

    # Logging
    torch.manual_seed(seed)
    model_accuracies = []

    # Data
    X,y = get_moons_dataset

    # Training
    print(f"STARTING TRAINING MODEL ZOO")
    print(f"Angles: {angles}")
    print(f"Models per angle: {models_per_angle}")
    for angle in tqdm(angles):
        X_rotated = rotate(X, angle)
        X_tensor = torch.tensor(X_rotated, dtype=torch.float32).to(device)
        y_tensor = torch.tensor(y, dtype=torch.float32).to(device)
        for i in range(models_per_angle):
            model = MLP(input_dim, hidden_dims, output_dim)
            model.to(device)
            criterion = nn.BCELoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

            model.train()
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

            model_name = f"model_{angle}_{i}.pth"
            torch.save(model.state_dict(), output_dir/model_name)
            model_accuracies.append((model_name, accuracy))

    return model_accuracies


def train_mlp(model: nn.Module, epochs: int, learning_rate: float, criterion, optimizer, dataloader_train, dataloader_valid=None, seed:int=None, log:bool=False) -> tuple:
    '''
    Train a MLP model on the given dataloaders.\n

    Parameters:
        model (nn.Module): The model to train.
        epochs (int): Number of epochs to train.
        learning_rate (float): The learning rate.
        criterion: The loss function.
        optimizer: The optimizer.
        dataloader_train: The dataloader for training.
        dataloader_valid: The dataloader for validating.
        seed (int): Optional seed to set pytorch seed for reproducibility.
        log (bool): Whether to log the training process.

    Return:
        Tuple of 2 lists of train_losses, valid_losses with loss values at each epoch.
    '''
    if(seed is not None):
        torch.manual_seed(seed)
        
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    train_losses = []
    valid_losses = []

    # Training
    for epoch in tqdm(range(epochs)):
        total_loss = 0
        model.train()
        for X, y in dataloader_train:
            X = X.to(device)
            y = y.to(device)

            y_pred = model(X)
            loss = criterion(y_pred, y)
            total_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        total_loss /= len(dataloader_train.dataset)
        train_losses.append(total_loss)
        if(log):
            wandb.log({"train_loss": total_loss})

        if(dataloader_valid is not None):
            total_loss = 0
            model.eval()
            for X, y in dataloader_valid:
                X = X.to(device)
                y = y.to(device)

                y_pred = model(X.float())
                loss = criterion(y_pred, y)
                total_loss += loss.item()
            
            total_loss /= len(dataloader_valid.dataset)
            valid_losses.append(total_loss)
            if(log):
                wandb.log({"valid_loss": total_loss})
    
    return train_losses, valid_losses

def get_accuracy(model: nn.Module, dataloader):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    total_correct = 0
    model.eval()
    for X, y in dataloader:
        X = X.to(device)
        y = y.to(device)
        y_pred = model(X.float())
        # Accuracy
        y = torch.argmax(y, dim=1)
        y_pred = torch.argmax(y_pred, dim=1)
        correct = (y_pred == y).sum()
        total_correct += correct

    accuracy_trained = total_correct / len(dataloader.dataset) * 100
    return accuracy_trained.item()

if __name__ == "__main__":    
    angles = [0, 90, 180, 270]
    model_accuracies = train_zoo(angles, 2000, "data/raw/moons.csv", "models/four_angles_zoo/")
    print(model_accuracies)
