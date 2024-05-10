import torch
import pandas as pd
import numpy as np

# Custom dataset with our model parameters
class ModelParamsDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_path, angle_change: int = 90):
        self.dataset_path = dataset_path
        self.dataset = pd.read_csv(dataset_path)
        self.angle_change = angle_change
        self.weights = self.dataset.drop(columns=["model_name", "angle"])
        self.angles = self.dataset["angle"]
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        weights = torch.tensor(self.weights.iloc[idx].astype('float32').values)
        angle = torch.tensor(self.angles.iloc[idx]/self.angle_change, dtype=torch.int64)
        angle = torch.nn.functional.one_hot(angle, num_classes=int(360/self.angle_change)).float()

        return weights, angle