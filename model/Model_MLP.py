import torch.nn as nn
import torch
import torch.nn.functional as F


class Model_MLP(nn.Module):
    def __init__(self):
        super(Model_MLP, self).__init__()

    def forward(self, x):
        return x
