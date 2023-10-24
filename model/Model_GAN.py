import torch.nn as nn
import torch
import torch.nn.functional as F


class Model_GAN_GEN(nn.Module):
    def __init__(self, dropout_rate):
        super(Model_GAN_GEN, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(3, 16),
            nn.Dropout(dropout_rate),
            nn.Linear(16, 6),
            nn.Dropout(dropout_rate),
            nn.Linear(6, 3),
        )

    def forward(self, x):
        return self.model(x)


class Model_GAN_DISC(nn.Module):
    def __init__(self, dropout_rate):
        super(Model_GAN_DISC, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(3, 64),
            nn.Dropout(dropout_rate),
            nn.ReLU(),
            nn.Linear(64, 2),
        )

    def forward(self, x):
        return self.model(x)
