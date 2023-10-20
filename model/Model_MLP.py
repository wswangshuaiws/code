import torch.nn as nn
import torch
import torch.nn.functional as F


class Model_MLP(nn.Module):
    def __init__(self, dropout_rate):
        super(Model_MLP, self).__init__()

        self.layers = nn.Sequential(
            nn.Linear(3, 9),
            nn.Dropout(dropout_rate),
            nn.PReLU(),
            nn.Linear(9, 6),
            nn.Dropout(dropout_rate),
            nn.PReLU(),
            nn.Linear(6, 3),
            nn.Dropout(dropout_rate),
            nn.PReLU(),
        )

    def forward(self, x):
        return self.layers(x)
