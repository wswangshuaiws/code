import torch.nn as nn
import torch
import torch.nn.functional as F


class Model_GAN_GEN(nn.Module):
    def __init__(self):
        super(Model_GAN_GEN, self).__init__()

    def forward(self, x):
        return x


class Model_GAN_DISC(nn.Module):
    def __init__(self):
        super(Model_GAN_DISC, self).__init__()

    def forward(self, x):
        return x