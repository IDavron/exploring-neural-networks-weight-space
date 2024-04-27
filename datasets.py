import torch

class ModelParamsDataset(torch.utils.data.Dataset):
    '''
    Custom dataset with our model parameters
    '''
    def __init__(self, parameters, label):
        self.params = parameters
        self.label = label

    def __len__(self):
        return len(self.params)

    def __getitem__(self, idx):
        return self.params[idx], self.label[idx]