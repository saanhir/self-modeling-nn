import torch
import torch.nn as nn

from functools import reduce


class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dims=[100]):
        super(MLP, self).__init__()

        hidden_dims = [in_dim] + hidden_dims
        layers = [(nn.Linear(i, j), nn.Sigmoid()) for i, j in zip(hidden_dims, hidden_dims[1:])]
        self.mlp = nn.Sequential(*reduce(lambda a, b: a + b, layers))
        self.fc_out = nn.Linear(hidden_dims[-1], out_dim)

    def forward(self, x):
        aux = self.mlp(x)
        out = self.fc_out(aux)
        return out, aux


class MLP_SM(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dims=[100], aux_layer=1):
        super(MLP_SM, self).__init__()

        hidden_dims = [in_dim] + hidden_dims
        layers = [(nn.Linear(i, j), nn.Sigmoid()) for i, j in zip(hidden_dims, hidden_dims[1:])]
        self.mlp = nn.Sequential(*reduce(lambda a, b: a + b, layers))
        self.fc_out = nn.Linear(hidden_dims[-1], out_dim + hidden_dims[aux_layer])

    def forward(self, x):
        aux = self.mlp(x)
        out = self.fc_out(aux)
        return out, aux

