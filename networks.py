import torch
from torch import nn


class SmallConv(nn.Module):
    def __init__(self, classes=10):
        super().__init__()
        self.stack = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.fc_in = nn.Linear(32 * 7 * 7, 128)
        self.fc_out = nn.Linear(128, classes)
        self.drop = nn.Dropout(0.3)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        z = self.stack(x).flatten(1)
        z = self.drop(self.relu(self.fc_in(z)))
        return self.fc_out(z)


_REGISTRY = {
    "small_conv": SmallConv,
}


def build(name="small_conv", **kwargs):
    if name not in _REGISTRY:
        raise KeyError(f"unknown architecture: {name}; have {sorted(_REGISTRY)}")
    return _REGISTRY[name](**kwargs)


def available():
    return sorted(_REGISTRY)