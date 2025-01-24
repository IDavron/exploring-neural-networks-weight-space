import torch.nn as nn
import torch
import torch.nn.functional as F

from typing import Tuple
from src.model.layers.layers import BN, DownSampleDWSLayer, Dropout, DWSLayer, InvariantLayer, ReLU, NaiveInvariantLayer

import math

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
    
# Atuoencoder Models
class VAE(nn.Module):
    def __init__(self, in_out_dim = 33, latent_dim = 2, dropout=0.0) -> None:
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(
            torch.nn.Linear(in_out_dim, 1024),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(1024),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(1024, 512),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(512),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(512, 256),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(256),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(256, 128),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(128),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(128, latent_dim),
        )
        self.mu = nn.Linear(latent_dim, latent_dim)
        self.logvar = nn.Linear(latent_dim, latent_dim)
        self.decoder = nn.Sequential(
            torch.nn.Linear(latent_dim, 128),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(128),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(128, 256),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(256),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(256, 512),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(512),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(512, 1024),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(1024),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(1024, in_out_dim),
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def forward(self, x):
        h = self.encoder(x)
        mu = self.mu(h)
        logvar = self.logvar(h)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar


class Autoencoder(nn.Module):
    def __init__(self, in_out_dim = 33, latent_dim = 2, dropout=0.0) -> None:
        super(Autoencoder, self).__init__()
        
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(in_out_dim, 1024),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(1024),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(1024, 512),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(512),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(512, 256),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(256),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(256, 128),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(128),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(128, latent_dim),
        )

        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(latent_dim, 128),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(128),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(128, 256),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(256),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(256, 512),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(512),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(512, 1024),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(1024),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(1024, in_out_dim),
        )

    def forward(self, x):
        latent = self.encoder(x)
        output = self.decoder(latent)
        return output


class TSModel(nn.Module):
    def __init__(self, latent_dim, dropout=0.0) -> None:
        super(TSModel, self).__init__()
        
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(33, 1024),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(1024),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(1024, 512),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(512),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(512, 256),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(256),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(256, 128),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(128),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(128, latent_dim),
        )

        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(latent_dim, 128),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(128),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(128, 256),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(256),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(256, 512),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(512),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(512, 1024),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(1024),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(1024, 17),
        )

    def forward(self, x):
        latent = self.encoder(x)
        output = self.decoder(latent)
        return output

# Flow Matching Model
class Flow(nn.Module):
    def __init__(self, input_dim = 36, output_dim = 33, hidden_dim=512, n_hidden=2):
        super(Flow, self).__init__()


        self.layers = []
        self.layers.append(nn.Linear(input_dim, hidden_dim))
        self.layers.append(nn.SELU())

        for i in range(n_hidden):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
            self.layers.append(nn.SELU())
        
        self.layers.append(nn.Linear(hidden_dim, output_dim))

        self.layers = nn.Sequential(*self.layers)

    def forward(self, x):
        return self.layers(x)

# Diffusion Model
# Taken from Tiny Diffusion repository: https://github.com/tanelp/tiny-diffusion'''
class SinusoidalEmbedding(nn.Module):
    def __init__(self, size: int, scale: float = 1.0):
        super().__init__()
        self.size = size
        self.scale = scale

    def forward(self, x: torch.Tensor):
        x = x * self.scale
        half_size = self.size // 2
        emb = torch.log(torch.Tensor([10000.0])) / (half_size - 1)
        emb = torch.exp(-emb * torch.arange(half_size))
        emb = emb.to(x.device)
        emb = x.unsqueeze(-1) * emb.unsqueeze(0)
        emb = torch.cat((torch.sin(emb), torch.cos(emb)), dim=-1)
        return emb

    def __len__(self):
        return self.size
    

class Diffusion(nn.Module):
    def __init__(self, input_dim = 163, output_dim = 33, hidden_dim = 1024, n_hidden = 4):
        super().__init__()

        self.time_mlp = SinusoidalEmbedding(128)

        self.layers = []
        self.layers.append(nn.Linear(input_dim, hidden_dim))
        self.layers.append(nn.GELU())

        for i in range(n_hidden):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
            self.layers.append(nn.GELU())
        
        self.layers.append(nn.Linear(hidden_dim, output_dim))

        self.layers = nn.Sequential(*self.layers)

    def forward(self, x, t, angle):
        t_emb = self.time_mlp(t)
        x = torch.cat((x, t_emb, angle), dim=-1)
        
        x = self.layers(x)
        return x


# Decision Boundary Loss Models
class DBModelSmall(nn.Module):
    '''
    Model to classify the input with given parameters.
    Parameters:
        autoencoder (nn.Module): The autoencoder to use for the parameters.
        use_autoencoder (bool): Whether to use the autoencoder or not.
    '''
    def __init__(self, weights=None, batch_first=True) -> None:
        super(DBModelSmall, self).__init__()

        self.weights = weights
        self.batch_first = batch_first
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, input):
        parameters = self.weights
        if(self.batch_first):
            weights_1 = parameters[:, :16].reshape(-1, 8, 2).transpose(1, 2)
            bias_1 = parameters[:, 16:24].reshape(-1, 8)
            weights_2 = parameters[:, 24:32].reshape(-1, 1, 8).transpose(1, 2)
            bias_2 = parameters[:, 32:33].reshape(-1, 1)

            bias_1 = bias_1.unsqueeze(1).repeat(1, input.shape[1], 1)
            bias_2 = bias_2.unsqueeze(1).repeat(1, input.shape[1], 1)
        
            x = torch.bmm(input, weights_1) + bias_1
            x = self.relu(x)

            x = torch.bmm(x, weights_2) + bias_2
        else:
            weights_1 = parameters[:16].reshape(8, 2).T
            bias_1 = parameters[16:24].reshape(8)
            weights_2 = parameters[24:32].reshape(1,8).T
            bias_2 = parameters[32:33].reshape(1)

            x = torch.matmul(input, weights_1) + bias_1
            x = self.relu(x)

            x = torch.matmul(x, weights_2) + bias_2


        x = self.sigmoid(x) 
        return x
    
    def set_weights(self, weights):
        self.weights = weights

class DBModelMedium(nn.Module):
    '''
    Model to classify the input with given parameters.
    Parameters:
        autoencoder (nn.Module): The autoencoder to use for the parameters.
        use_autoencoder (bool): Whether to use the autoencoder or not.
    '''
    def __init__(self, weights=None, batch_first=True) -> None:
        super(DBModelMedium, self).__init__()

        self.weights = weights
        self.batch_first = batch_first
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, input):
        parameters = self.weights
        if(self.batch_first):
            weights_1 = parameters[:, :20].reshape(-1, 10, 2).transpose(1, 2)
            bias_1 = parameters[:, 20:30].reshape(-1, 10)
            weights_2 = parameters[:, 30:130].reshape(-1, 10, 10).transpose(1, 2)
            bias_2 = parameters[:, 130:140].reshape(-1, 10)
            weights_3 = parameters[:, 140:150].reshape(-1, 1, 10).transpose(1, 2)
            bias_3 = parameters[:, 150:115].reshape(-1, 1)

            bias_1 = bias_1.unsqueeze(1).repeat(1, input.shape[1], 1)
            bias_2 = bias_2.unsqueeze(1).repeat(1, input.shape[1], 1)
            bias_3 = bias_3.unsqueeze(1).repeat(1, input.shape[1], 1)
        
            x = torch.bmm(input, weights_1) + bias_1
            x = self.relu(x)
            x = torch.bmm(x, weights_2) + bias_2
            x = self.relu(x)

            x = torch.bmm(x, weights_3) + bias_3
        else:
            weights_1 = parameters[:20].reshape(10, 2).T
            bias_1 = parameters[20:30].reshape(10)
            weights_2 = parameters[30:130].reshape(10, 10).T
            bias_2 = parameters[130:140].reshape(10)
            weights_3 = parameters[140:150].reshape(1, 10).T
            bias_3 = parameters[150:151].reshape(1)

            x = torch.matmul(input, weights_1) + bias_1
            x = self.relu(x)
            x = torch.matmul(x, weights_2) + bias_2
            x = self.relu(x)

            x = torch.matmul(x, weights_3) + bias_3


        x = self.sigmoid(x) 
        return x

    def set_weights(self, weights):
        self.weights = weights

class DBModelBig(nn.Module):
    '''
    Model to classify the input with given parameters.
    Parameters:
        autoencoder (nn.Module): The autoencoder to use for the parameters.
        use_autoencoder (bool): Whether to use the autoencoder or not.
    '''
    def __init__(self, weights=None, batch_first=True) -> None:
        super(DBModelBig, self).__init__()

        self.weights = weights
        self.batch_first = batch_first
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, input):
        parameters = self.weights
        if(self.batch_first):
            weights_1 = parameters[:, :20].reshape(-1, 10, 2).transpose(1, 2)
            bias_1 = parameters[:, 20:30].reshape(-1, 10)
            weights_2 = parameters[:, 30:130].reshape(-1, 10, 10).transpose(1, 2)
            bias_2 = parameters[:, 130:140].reshape(-1, 10)
            weights_3 = parameters[:, 140:240].reshape(-1, 10, 10).transpose(1, 2)
            bias_3 = parameters[:, 240:250].reshape(-1, 10)
            weights_4 = parameters[:, 250:260].reshape(-1, 1, 10).transpose(1, 2)
            bias_4 = parameters[:, 260:261].reshape(-1, 1)

            bias_1 = bias_1.unsqueeze(1).repeat(1, input.shape[1], 1)
            bias_2 = bias_2.unsqueeze(1).repeat(1, input.shape[1], 1)
            bias_3 = bias_3.unsqueeze(1).repeat(1, input.shape[1], 1)
            bias_4 = bias_4.unsqueeze(1).repeat(1, input.shape[1], 1)
        
            x = torch.bmm(input, weights_1) + bias_1
            x = self.relu(x)
            x = torch.bmm(x, weights_2) + bias_2
            x = self.relu(x)
            x = torch.bmm(x, weights_3) + bias_3
            x = self.relu(x)

            x = torch.bmm(x, weights_4) + bias_4
        else:
            weights_1 = parameters[:20].reshape(10, 2).T
            bias_1 = parameters[20:30].reshape(10)
            weights_2 = parameters[30:130].reshape(10, 10).T
            bias_2 = parameters[130:140].reshape(10)
            weights_3 = parameters[140:240].reshape(10, 10).T
            bias_3 = parameters[240:250].reshape(10)
            weights_4 = parameters[250:260].reshape(1, 10).T
            bias_4 = parameters[260:261].reshape(1)

            x = torch.matmul(input, weights_1) + bias_1
            x = self.relu(x)
            x = torch.matmul(x, weights_2) + bias_2
            x = self.relu(x)
            x = torch.matmul(x, weights_3) + bias_3
            x = self.relu(x)

            x = torch.matmul(x, weights_4) + bias_4


        x = self.sigmoid(x) 
        return x
    
    def set_weights(self, weights):
        self.weights = weights


class SModel(nn.Module):
    def __init__(self, weights=None, batch_first=True) -> None:
        super(SModel, self).__init__()

        self.weights = weights
        self.batch_first = batch_first
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, input):
        parameters = self.weights
        if(self.batch_first):
            weights_1 = parameters[:, :8].reshape(-1, 4, 2).transpose(1, 2)
            bias_1 = parameters[:, 8:12].reshape(-1, 4)
            weights_2 = parameters[:, 12:16].reshape(-1, 1, 4).transpose(1, 2)
            bias_2 = parameters[:, 16:17].reshape(-1, 1)

            bias_1 = bias_1.unsqueeze(1).repeat(1, input.shape[1], 1)
            bias_2 = bias_2.unsqueeze(1).repeat(1, input.shape[1], 1)
        
            x = torch.bmm(input, weights_1) + bias_1
            x = self.relu(x)

            x = torch.bmm(x, weights_2) + bias_2
        else:
            weights_1 = parameters[:8].reshape(4, 2).T
            bias_1 = parameters[8:12].reshape(4)
            weights_2 = parameters[12:16].reshape(1, 4).T
            bias_2 = parameters[16:17].reshape(1)

            x = torch.matmul(input, weights_1) + bias_1
            x = self.relu(x)

            x = torch.matmul(x, weights_2) + bias_2


        x = self.sigmoid(x) 
        return x
    
    def set_weights(self, weights):
        self.weights = weights

# Source code from Equivariant Architectures for Learning in Deep Weight Spaces
# https://github.com/AvivNavon/DWSNets

class DWSModel(nn.Module):
    def __init__(
            self,
            weight_shapes: Tuple[Tuple[int, int], ...],
            bias_shapes: Tuple[
                Tuple[
                    int,
                ],
                ...,
            ],
            input_features,
            hidden_dim,
            n_hidden=2,
            output_features=None,
            reduction="max",
            bias=True,
            n_fc_layers=1,
            num_heads=8,
            set_layer="sab",
            input_dim_downsample=None,
            dropout_rate=0.0,
            add_skip=False,
            add_layer_skip=False,
            init_scale=1e-4,
            init_off_diag_scale_penalty=1.,
            bn=False,
            diagonal=False,
    ):
        super().__init__()
        assert len(weight_shapes) > 2, "the current implementation only support input networks with M>2 layers."

        self.input_features = input_features
        self.input_dim_downsample = input_dim_downsample
        if output_features is None:
            output_features = hidden_dim

        self.add_skip = add_skip
        if self.add_skip:
            self.skip = nn.Linear(
                input_features,
                output_features,
                bias=bias
            )
            with torch.no_grad():
                torch.nn.init.constant_(self.skip.weight, 1. / self.skip.weight.numel())
                torch.nn.init.constant_(self.skip.bias, 0.)

        if input_dim_downsample is None:
            layers = [
                DWSLayer(
                    weight_shapes=weight_shapes,
                    bias_shapes=bias_shapes,
                    in_features=input_features,
                    out_features=hidden_dim,
                    reduction=reduction,
                    bias=bias,
                    n_fc_layers=n_fc_layers,
                    num_heads=num_heads,
                    set_layer=set_layer,
                    add_skip=add_layer_skip,
                    init_scale=init_scale,
                    init_off_diag_scale_penalty=init_off_diag_scale_penalty,
                    diagonal=diagonal,
                ),
            ]
            for i in range(n_hidden):
                if bn:
                    layers.append(BN(hidden_dim, len(weight_shapes), len(bias_shapes)))

                layers.extend(
                    [

                        ReLU(),
                        Dropout(dropout_rate),
                        DWSLayer(
                            weight_shapes=weight_shapes,
                            bias_shapes=bias_shapes,
                            in_features=hidden_dim,
                            out_features=hidden_dim if i != (n_hidden - 1) else output_features,
                            reduction=reduction,
                            bias=bias,
                            n_fc_layers=n_fc_layers,
                            num_heads=num_heads if i != (n_hidden - 1) else 1,
                            set_layer=set_layer,
                            add_skip=add_layer_skip,
                            init_scale=init_scale,
                            init_off_diag_scale_penalty=init_off_diag_scale_penalty,
                            diagonal=diagonal,
                        ),
                    ]
                )
        else:
            layers = [
                DownSampleDWSLayer(
                    weight_shapes=weight_shapes,
                    bias_shapes=bias_shapes,
                    in_features=input_features,
                    out_features=hidden_dim,
                    reduction=reduction,
                    bias=bias,
                    n_fc_layers=n_fc_layers,
                    num_heads=num_heads,
                    set_layer=set_layer,
                    downsample_dim=input_dim_downsample,
                    add_skip=add_layer_skip,
                    init_scale=init_scale,
                    init_off_diag_scale_penalty=init_off_diag_scale_penalty,
                    diagonal=diagonal,
                ),
            ]
            for i in range(n_hidden):
                if bn:
                    layers.append(BN(hidden_dim, len(weight_shapes), len(bias_shapes)))

                layers.extend(
                    [
                        ReLU(),
                        Dropout(dropout_rate),
                        DownSampleDWSLayer(
                            weight_shapes=weight_shapes,
                            bias_shapes=bias_shapes,
                            in_features=hidden_dim,
                            out_features=hidden_dim if i != (n_hidden - 1) else output_features,
                            reduction=reduction,
                            bias=bias,
                            n_fc_layers=n_fc_layers,
                            num_heads=num_heads if i != (n_hidden - 1) else 1,
                            set_layer=set_layer,
                            downsample_dim=input_dim_downsample,
                            add_skip=add_layer_skip,
                            init_scale=init_scale,
                            init_off_diag_scale_penalty=init_off_diag_scale_penalty,
                            diagonal=diagonal,
                        ),
                    ]
                )
        self.layers = nn.Sequential(*layers)

    def forward(self, x: Tuple[Tuple[torch.tensor], Tuple[torch.tensor]]):
        out = self.layers(x)
        if self.add_skip:
            skip_out = tuple(self.skip(w) for w in x[0]), tuple(
                self.skip(b) for b in x[1]
            )
            weight_out = tuple(ws + w for w, ws in zip(out[0], skip_out[0]))
            bias_out = tuple(bs + b for b, bs in zip(out[1], skip_out[1]))
            out = weight_out, bias_out
        return out


class DWSModelForClassification(nn.Module):
    def __init__(
        self,
        weight_shapes: Tuple[Tuple[int, int], ...],
        bias_shapes: Tuple[
            Tuple[
                int,
            ],
            ...,
        ],
        input_features,
        hidden_dim,
        n_hidden=2,
        n_classes=10,
        reduction="max",
        bias=True,
        n_fc_layers=1,
        num_heads=8,
        set_layer="sab",
        n_out_fc=1,
        dropout_rate=0.0,
        input_dim_downsample=None,
        init_scale=1.,
        init_off_diag_scale_penalty=1.,
        bn=False,
        add_skip=False,
        add_layer_skip=False,
        equiv_out_features=None,
        diagonal=False,
    ):
        super().__init__()
        self.layers = DWSModel(
            weight_shapes=weight_shapes,
            bias_shapes=bias_shapes,
            input_features=input_features,
            hidden_dim=hidden_dim,
            n_hidden=n_hidden,
            reduction=reduction,
            bias=bias,
            output_features=equiv_out_features,
            n_fc_layers=n_fc_layers,
            num_heads=num_heads,
            set_layer=set_layer,
            dropout_rate=dropout_rate,
            input_dim_downsample=input_dim_downsample,
            init_scale=init_scale,
            init_off_diag_scale_penalty=init_off_diag_scale_penalty,
            bn=bn,
            add_skip=add_skip,
            add_layer_skip=add_layer_skip,
            diagonal=diagonal,
        )
        self.dropout = Dropout(dropout_rate)
        self.relu = ReLU()
        self.clf = InvariantLayer(
            weight_shapes=weight_shapes,
            bias_shapes=bias_shapes,
            in_features=hidden_dim
            if equiv_out_features is None
            else equiv_out_features,
            out_features=n_classes,
            reduction=reduction,
            n_fc_layers=n_out_fc,
        )

    def forward(
        self, x: Tuple[Tuple[torch.tensor], Tuple[torch.tensor]], return_equiv=False
    ):
        x = self.layers(x)
        out = self.clf(self.dropout(self.relu(x)))
        if return_equiv:
            return out, x
        else:
            return out
        

# Source code from Set Transformers paper
# https://github.com/juho-lee/set_transformer
class MAB(nn.Module):
    def __init__(self, dim_Q, dim_K, dim_V, num_heads, ln=False):
        super(MAB, self).__init__()
        self.dim_V = dim_V
        self.num_heads = num_heads
        self.fc_q = nn.Linear(dim_Q, dim_V)
        self.fc_k = nn.Linear(dim_K, dim_V)
        self.fc_v = nn.Linear(dim_K, dim_V)
        if ln:
            self.ln0 = nn.LayerNorm(dim_V)
            self.ln1 = nn.LayerNorm(dim_V)
        self.fc_o = nn.Linear(dim_V, dim_V)

    def forward(self, Q, K):
        Q = self.fc_q(Q)
        K, V = self.fc_k(K), self.fc_v(K)

        dim_split = self.dim_V // self.num_heads
        Q_ = torch.cat(Q.split(dim_split, 2), 0)
        K_ = torch.cat(K.split(dim_split, 2), 0)
        V_ = torch.cat(V.split(dim_split, 2), 0)

        A = torch.softmax(Q_.bmm(K_.transpose(1,2))/math.sqrt(self.dim_V), 2)
        O = torch.cat((Q_ + A.bmm(V_)).split(Q.size(0), 0), 2)
        O = O if getattr(self, 'ln0', None) is None else self.ln0(O)
        O = O + F.relu(self.fc_o(O))
        O = O if getattr(self, 'ln1', None) is None else self.ln1(O)
        return O

class SAB(nn.Module):
    def __init__(self, dim_in, dim_out, num_heads, ln=False):
        super(SAB, self).__init__()
        self.mab = MAB(dim_in, dim_in, dim_out, num_heads, ln=ln)

    def forward(self, X):
        return self.mab(X, X)

class ISAB(nn.Module):
    def __init__(self, dim_in, dim_out, num_heads, num_inds, ln=False):
        super(ISAB, self).__init__()
        self.I = nn.Parameter(torch.Tensor(1, num_inds, dim_out))
        nn.init.xavier_uniform_(self.I)
        self.mab0 = MAB(dim_out, dim_in, dim_out, num_heads, ln=ln)
        self.mab1 = MAB(dim_in, dim_out, dim_out, num_heads, ln=ln)

    def forward(self, X):
        H = self.mab0(self.I.repeat(X.size(0), 1, 1), X)
        return self.mab1(X, H)

class PMA(nn.Module):
    def __init__(self, dim, num_heads, num_seeds, ln=False):
        super(PMA, self).__init__()
        self.S = nn.Parameter(torch.Tensor(1, num_seeds, dim))
        nn.init.xavier_uniform_(self.S)
        self.mab = MAB(dim, dim, dim, num_heads, ln=ln)

    def forward(self, X):
        return self.mab(self.S.repeat(X.size(0), 1, 1), X)


class SetTransformer(nn.Module):
    def __init__(self, dim_input, num_outputs, dim_output,
            num_inds=32, dim_hidden=128, num_heads=4, ln=False):
        super(SetTransformer, self).__init__()
        self.enc = nn.Sequential(
                ISAB(dim_input, dim_hidden, num_heads, num_inds, ln=ln),
                ISAB(dim_hidden, dim_hidden, num_heads, num_inds, ln=ln))
        self.dec = nn.Sequential(
                PMA(dim_hidden, num_heads, num_outputs, ln=ln),
                SAB(dim_hidden, dim_hidden, num_heads, ln=ln),
                SAB(dim_hidden, dim_hidden, num_heads, ln=ln),
                nn.Linear(dim_hidden, dim_output),
                nn.Softmax(dim=1))

    def forward(self, X):
        return self.dec(self.enc(X)).squeeze()