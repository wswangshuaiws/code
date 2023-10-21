import torch.nn as nn
import torch
import torch.nn.functional as F


class Model_GAN_GEN(nn.Module):
    def __init__(self, dropout_rate):
        super(Model_GAN_GEN, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.Linear(64, 3),
        )

    def forward(self, x):
        return self.model(x)


class Model_GAN_DISC(nn.Module):
    def __init__(self, dropout_rate):
        super(Model_GAN_DISC, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)
