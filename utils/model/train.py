#################################################################################################
#   Script to train MLP model zoo on moons dataset with different angles and save them.         #
#   Allows to log model accuracies into WandB. For that WANDB_PROJECT and WANDB_ENTITY fields   #
#   must be present in .env file.                                                               #
#################################################################################################


import torch
import torch.nn as nn
import sklearn.datasets

from tqdm import tqdm
import wandb

from utils.model.models import MLP, Classifier
from utils.data.helpers import get_moons_dataset, rotate

import os
from argparse import ArgumentParser


def train_zoo(config, output_dir, log=False):
    experiment_name = config["EXPERIMENT_NAME"]
    # Hyperparameters
    hyperparameters = config["HYPERPARAMETERS"]
    angles = hyperparameters["ANGLES"]
    models_per_angle = hyperparameters["MODELS_PER_ANGLE"]
    seed = hyperparameters["SEED"]

    model_config = config["MODEL"]

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    save_dir = os.path.join(output_dir, experiment_name)
    if(not os.path.exists(save_dir)):
        os.mkdir(save_dir)

    if(log):
        cfg = {"experiment_name": experiment_name, "hyperparameter": hyperparameters, "model": model_config}
        wandb.init(project=config["WANDB_PROJECT"], entity=config["WANDB_ENTITY"], config=cfg, name=experiment_name)
        columns = ["model", "accuracy"]
        accuracy_table = wandb.Table(columns=columns)

    # Dataset
    X,y = get_moons_dataset()

    # Logging
    torch.manual_seed(seed)

    # Training
    print(f"STARTING TRAINING MODEL ZOO")
    print(f"Angles: {angles}")
    print(f"Models per angle: {models_per_angle}")
    
    for angle in tqdm(angles):
        for i in tqdm(range(models_per_angle)):
            X_rotated = rotate(X, angle)
            X_tensor = torch.tensor(X_rotated, dtype=torch.float32).to(device)
            y_tensor = torch.tensor(y.reshape(-1,1), dtype=torch.float32).to(device)

            model = train_mlp(config, X_tensor, y_tensor)

            model.eval()
            y_pred = model(X_tensor).cpu().detach().numpy().round().flatten()
            correct = (y_pred == y).sum()
            accuracy = correct / len(y) * 100

            # Log accuracy into WandB
            model_name = f"model_{angle}_{i}.pth"

            if(log):
                accuracy_table.add_data(model_name, accuracy)

            # Save the model
            model_path = os.path.join(save_dir, model_name)
            torch.save(model.state_dict(), model_path)

    if(log):
        wandb.log({"accuracy_table": accuracy_table})
        wandb.finish()

def train_mlp(config, X, y) -> MLP:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Hyperparameters
    hyperparameters = config["HYPERPARAMETERS"]
    epochs = hyperparameters["EPOCHS"]
    learning_rate = hyperparameters["LEARNING_RATE"]

    model_config = config["MODEL"]
    input_dim = model_config["INPUT_DIM"]
    hidden_dims = model_config["HIDDEN_DIMS"]
    output_dim = model_config["OUTPUT_DIM"]

    model = MLP(input_dim, hidden_dims, output_dim).to(device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        y_pred = model(X)
        loss = criterion(y_pred, y)
        loss.backward()
        optimizer.step()
    
    return model

def train_classifier(config, dataloader_train, dataloader_valid, log=False, save=False, save_dir="") -> Classifier:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    experiment_name = config["EXPERIMENT_NAME"]
    # Hyperparameters
    hyperparameters = config["HYPERPARAMETERS"]
    epochs = hyperparameters["EPOCHS"]
    learning_rate = hyperparameters["LEARNING_RATE"]

    model_config = config["MODEL"]
    input_dim = model_config["INPUT_DIM"]
    hidden_dims = model_config["HIDDEN_DIMS"]
    output_dim = model_config["OUTPUT_DIM"]
    dropout = model_config["DROPOUT"]

    if(log):
        cfg = {"experiment_name": experiment_name, "hyperparameter": hyperparameters, "model": model_config}
        wandb.init(project=config["WANDB_PROJECT"], entity=config["WANDB_ENTITY"], config=cfg, name=experiment_name, group="angle_classifier")

    model = Classifier(input_dim, hidden_dims, output_dim, dropout)
    model.to(device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Training
    for epoch in tqdm(range(epochs)):
        total_loss = 0
        model.train()
        for X, y in dataloader_train:
            X = X.to(device)
            y = y.to(device)
            y_pred = model(X.float())
            y = torch.nn.functional.one_hot(y, output_dim).float()
            loss = criterion(y_pred, y)
            total_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        total_loss /= len(dataloader_train.dataset)
        if(log):
            wandb.log({"epoch": epoch, "train_loss": total_loss})

        total_loss = 0
        model.eval()
        for X, y in dataloader_valid:
            X = X.to(device)
            y = y.to(device)
            y_pred = model(X.float())
            y = torch.nn.functional.one_hot(y, num_classes=output_dim).float()
            loss = criterion(y_pred, y)
            total_loss += loss.item()
        
        total_loss /= len(dataloader_valid.dataset)
        if(log):
            wandb.log({"epoch": epoch, "valid_loss": total_loss})

    if(save):
        model_path = os.path.join(save_dir, f"{experiment_name}.pth")
        torch.save(model.state_dict(), model_path)

    wandb.finish()
    return model
# if __name__ == "__main__":
#     parser = ArgumentParser(description='Generate moons dataset classifier MLP zoo.')
#     parser.add_argument('-c', '--config', help='Path to the config file.', required=True)
#     args = parser.parse_args()
#     default_dirs = get_default_dirs()
#     config_path = args.config
#     with open(config_path, 'r') as file:
#         config = yaml.safe_load(file)
#     wandb_config = dotenv_values(default_dirs["ENV_PATH"])
#     config.update(wandb_config)
#     train_zoo(config, default_dirs["MODELS_DIR"])