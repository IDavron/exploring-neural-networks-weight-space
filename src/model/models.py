import torch.nn as nn
import torch

from typing import Tuple
from src.model.layers.layers import BN, DownSampleDWSLayer, Dropout, DWSLayer, InvariantLayer, ReLU

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
    


class Variational(torch.nn.Module):
    def __init__(self) -> None:
        super(Variational, self).__init__()
        
        self.encoder = self.encoder = torch.nn.Sequential(
            torch.nn.Linear(33, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 16),
            torch.nn.ReLU(),
            torch.nn.Linear(16, 10),
        )

        self.decoder = self.decoder = torch.nn.Sequential(
            torch.nn.Linear(10, 16),
            torch.nn.ReLU(),
            torch.nn.Linear(16, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 33),
        )

        self.mean = torch.nn.Linear(10, 10)
        self.log_var = torch.nn.Linear(10, 10)
    
    def reparameterize(self, mean, log_var):
        var = torch.exp(0.5*log_var)
        epsilon = torch.randn_like(var)     
        latent = mean + var*epsilon
        return latent

    def forward(self, x):
        latent = self.encoder(x)
        mean = self.mean(latent)
        log_var = self.log_var(latent)
        x = self.reparameterize(mean, log_var)
        output = self.decoder(x)
        return output, mean, log_var
    


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