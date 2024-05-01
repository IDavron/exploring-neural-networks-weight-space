import torch.nn as nn
# Model
class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: list, output_dim: int):
        super(MLP, self).__init__()

        if(len(hidden_dims) == 0):
            raise ValueError("hidden_dims must have at least one element")
        
        self.fc1 = nn.Linear(input_dim, hidden_dims[0])

        # Add hidden layers
        self.hidden_layers = nn.ModuleList()
        for i in range(1, len(hidden_dims)):
            self.hidden_layers.append(nn.Linear(hidden_dims[i-1], hidden_dims[i]))

        self.fc2 = nn.Linear(hidden_dims[-1], output_dim)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        for hidden_layer in self.hidden_layers:
            x = self.relu(hidden_layer(x))
        x = self.fc2(x)
        return self.sigmoid(x)
    
class Classifier(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: list, output_dim: int):
        super(Classifier, self).__init__()

        if(len(hidden_dims) == 0):
            raise ValueError("hidden_dims must have at least one element")
        
        self.fc1 = nn.Linear(input_dim, hidden_dims[0])

        # Add hidden layers
        self.hidden_layers = nn.ModuleList()
        for i in range(1, len(hidden_dims)):
            self.hidden_layers.append(nn.Linear(hidden_dims[i-1], hidden_dims[i]))

        self.fc2 = nn.Linear(hidden_dims[-1], output_dim)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        for hidden_layer in self.hidden_layers:
            x = self.relu(hidden_layer(x))
        x = self.fc2(x)
        return self.softmax(x)