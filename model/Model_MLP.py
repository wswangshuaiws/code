import torch.nn as nn


class Model_MLP(nn.Module):
    def __init__(self, dropout_rate):
        super(Model_MLP, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(3, 16),
            nn.Dropout(dropout_rate),
            nn.Linear(16, 6),
            nn.Dropout(dropout_rate),
            nn.Linear(6, 3),
        )

    def forward(self, x):
        return self.model(x)
