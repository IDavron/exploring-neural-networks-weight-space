import torch.nn as nn
# Model
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, dropout=0.0, use_batch_norm=False, output_activation="sigmoid"):
        super(MLP, self).__init__()
        if(len(hidden_dims) == 0):
            raise ValueError("hidden_dims must have at least one element")
        
        if(use_batch_norm):
            self.batch_norm = nn.BatchNorm1d
        else:
            self.batch_norm = nn.Identity

        self.fc1 = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.ReLU(),
            self.batch_norm(hidden_dims[0])
            )
        
        # Add hidden layers
        self.hidden_layers = nn.ModuleList()
        for i in range(1, len(hidden_dims)):
            self.hidden_layers.append(
                nn.Sequential(nn.Linear(hidden_dims[i-1], hidden_dims[i]), 
                              nn.ReLU(),
                              self.batch_norm(hidden_dims[i]))
                              )

        self.fc2 = nn.Linear(hidden_dims[-1], output_dim)
        self.dropout = nn.Dropout(dropout)
        self.output_activation = nn.Sigmoid() if output_activation == "sigmoid" else nn.Softmax(dim=1)
    
    def forward(self, x):
        x = self.dropout(self.fc1(x))
        for hidden_layer in self.hidden_layers:
            x = self.dropout(hidden_layer(x))
        x = self.fc2(x)
        return self.output_activation(x)
    

class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(33, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 16),
            nn.ReLU(),
            nn.Linear(16, 10),
        )

        self.decoder = nn.Sequential(
            nn.Linear(10, 16),
            nn.ReLU(),
            nn.Linear(16, 64),
            nn.ReLU(),
            nn.Linear(64, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 33),
        )
    
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x